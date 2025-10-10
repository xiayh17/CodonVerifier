"""
Convert UniProt TSV dataset to JSONL format for surrogate training.

This module handles the conversion of UniProt-style TSV data with the following columns:
- Entry: UniProt ID
- Reviewed: Review status
- Entry Name: UniProt identifier
- Protein names: Protein name
- Gene Names: Gene names
- Organism: Source organism (E. coli, Human, Mouse, etc.)
- Length: Protein sequence length
- Subcellular location [CC]: Subcellular localization
- RefSeq_id: NCBI RefSeq identifier (or "embl" for EMBL sequences)
- Genome_id: NCBI genome ID or EMBL ID
- RefSeq_nn: Coding sequence (nucleotide)
- RefSeq_aa: Protein sequence (amino acid)
"""

import argparse
import json
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Organism name mapping to standardized host names
ORGANISM_MAP = {
    "Escherichia coli": "E_coli",
    "Homo sapiens": "Human",
    "Mus musculus": "Mouse",
    "Pichia pastoris": "P_pastoris",
    "Saccharomyces cerevisiae": "S_cerevisiae",
}

# Reverse mapping for convenience
HOST_TO_ORGANISM = {v: k for k, v in ORGANISM_MAP.items()}


def normalize_sequence(seq: str) -> str:
    """Remove whitespace and asterisks (stop codons) from sequence."""
    return seq.replace(" ", "").replace("*", "").replace("\n", "").upper()


def estimate_expression_from_metadata(
    reviewed: str,
    subcellular_location: str,
    protein_length: int,
    organism: str
) -> Tuple[float, str]:
    """
    Estimate expression level based on metadata.
    
    This is a heuristic approach since we don't have direct expression data.
    Returns (expression_value, confidence_level)
    
    Note: For enhanced estimation with Evo2 model features, use the
    expression_estimator module instead.
    """
    # Base score
    score = 50.0
    confidence = "low"
    
    # Reviewed proteins from SwissProt tend to be better characterized
    if reviewed.lower() == "reviewed":
        score += 20.0
        confidence = "medium"
    
    # Subcellular location hints at expression level
    if subcellular_location:
        location_lower = subcellular_location.lower()
        # Cytoplasmic proteins often have higher expression
        if "cytoplasm" in location_lower or "cytosol" in location_lower:
            score += 15.0
        # Ribosomal proteins typically highly expressed
        if "ribosome" in location_lower:
            score += 30.0
            confidence = "medium"
        # Membrane proteins may have lower expression
        if "membrane" in location_lower:
            score -= 10.0
        # Secreted proteins
        if "secreted" in location_lower or "extracellular" in location_lower:
            score -= 5.0
    
    # Protein length factor (very long or very short may be less expressed)
    if 100 <= protein_length <= 500:
        score += 10.0
    elif protein_length < 50 or protein_length > 1000:
        score -= 10.0
    
    # Ensure non-negative
    score = max(10.0, score)
    
    return score, confidence


def parse_tsv_row(row: Dict[str, str], source_file: str) -> Optional[Dict]:
    """
    Parse a single TSV row into JSONL format.
    
    Returns None if the row is invalid or missing critical fields.
    """
    try:
        # Extract fields
        entry = row.get("Entry", "").strip()
        reviewed = row.get("Reviewed", "").strip()
        entry_name = row.get("Entry Name", "").strip()
        protein_names = row.get("Protein names", "").strip()
        gene_names = row.get("Gene Names", "").strip()
        organism = row.get("Organism", "").strip()
        length_str = row.get("Length", "").strip()
        subcellular = row.get("Subcellular location [CC]", "").strip()
        refseq_id = row.get("RefSeq_id", "").strip()
        genome_id = row.get("Genome_id", "").strip()
        refseq_nn = row.get("RefSeq_nn", "").strip()
        refseq_aa = row.get("RefSeq_aa", "").strip()
        
        # Validate critical fields
        if not entry or not refseq_nn or not refseq_aa:
            missing_fields = []
            if not entry:
                missing_fields.append("Entry")
            if not refseq_nn:
                missing_fields.append("RefSeq_nn")
            if not refseq_aa:
                missing_fields.append("RefSeq_aa")
            logger.warning(f"Skipping row: missing critical fields: {', '.join(missing_fields)} (Entry={entry or 'N/A'})")
            return None
        
        # Normalize sequences
        sequence = normalize_sequence(refseq_nn)
        protein_aa = normalize_sequence(refseq_aa)
        
        # Validate sequence
        if not sequence or len(sequence) % 3 != 0:
            logger.warning(f"Skipping {entry}: invalid nucleotide sequence length")
            return None
        
        if not protein_aa:
            logger.warning(f"Skipping {entry}: empty protein sequence")
            return None
        
        # Check sequence consistency
        expected_aa_len = len(sequence) // 3
        if abs(len(protein_aa) - expected_aa_len) > 1:  # Allow 1 codon difference for stop
            logger.warning(
                f"Skipping {entry}: sequence length mismatch "
                f"(DNA={len(sequence)}, expected AA={expected_aa_len}, actual AA={len(protein_aa)})"
            )
            return None
        
        # Map organism to host
        host = ORGANISM_MAP.get(organism, None)
        if not host:
            # Try partial matching
            for org_key, host_val in ORGANISM_MAP.items():
                if org_key.lower() in organism.lower():
                    host = host_val
                    break
            if not host:
                # Smart fallback based on organism type
                organism_lower = organism.lower()
                if any(keyword in organism_lower for keyword in ["yeast", "pichia", "saccharomyces", "candida", "komagataella", "hansenula", "scheffersomyces", "wickerhamomyces", "cyberlindnera", "ogataea", "millerozyma"]):
                    fallback_host = "S_cerevisiae"
                    logger.warning(f"Unknown organism: {organism}, using S_cerevisiae as fallback (yeast)")
                elif any(keyword in organism_lower for keyword in ["human", "homo", "sapiens"]):
                    fallback_host = "Human"
                    logger.warning(f"Unknown organism: {organism}, using Human as fallback")
                elif any(keyword in organism_lower for keyword in ["mouse", "mus", "musculus"]):
                    fallback_host = "Mouse"
                    logger.warning(f"Unknown organism: {organism}, using Mouse as fallback")
                else:
                    fallback_host = "E_coli"
                    logger.warning(f"Unknown organism: {organism}, using E_coli as fallback (bacteria)")
                host = fallback_host
        
        # Parse length
        try:
            length = int(length_str) if length_str else len(protein_aa)
        except ValueError:
            length = len(protein_aa)
        
        # Estimate expression
        expr_value, confidence = estimate_expression_from_metadata(
            reviewed, subcellular, length, organism
        )
        
        # Build JSONL record
        record = {
            # Stable identifier for downstream joins
            "protein_id": entry or refseq_id or genome_id,
            "sequence": sequence,
            "protein_aa": protein_aa,
            "host": host,
            "expression": {
                "value": expr_value,
                "unit": "estimated",
                "assay": "metadata_heuristic",
                "confidence": confidence
            },
            "extra_features": {
                "length": length,
                "reviewed": reviewed.lower() == "reviewed",
                "source_file": source_file,
            },
            "metadata": {
                "uniprot_id": entry,
                "entry_name": entry_name,
                "protein_names": protein_names,
                "gene_names": gene_names,
                "organism": organism,
                "subcellular_location": subcellular,
                "refseq_id": refseq_id,
                "genome_id": genome_id,
            }
        }
        
        return record
        
    except Exception as e:
        logger.error(f"Error parsing row: {e}")
        return None


def convert_tsv_to_jsonl(
    tsv_path: str,
    output_jsonl: str,
    max_records: Optional[int] = None,
    filter_reviewed: bool = False
) -> Dict[str, int]:
    """
    Convert a TSV file to JSONL format.
    
    Args:
        tsv_path: Path to input TSV file
        output_jsonl: Path to output JSONL file
        max_records: Maximum number of records to convert (None = all)
        filter_reviewed: If True, only include reviewed entries
    
    Returns:
        Statistics dictionary
    """
    stats = {
        "total_rows": 0,
        "valid_records": 0,
        "skipped": 0,
        "filtered": 0,
    }
    
    source_file = Path(tsv_path).name
    
    with open(tsv_path, 'r', encoding='utf-8') as infile, \
         open(output_jsonl, 'w', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile, delimiter='\t')
        
        for row in reader:
            stats["total_rows"] += 1
            
            record = parse_tsv_row(row, source_file)
            
            if record is None:
                stats["skipped"] += 1
                continue
            
            # Filter reviewed if requested
            if filter_reviewed and not record["extra_features"].get("reviewed", False):
                stats["filtered"] += 1
                continue
            
            # Write record
            outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
            stats["valid_records"] += 1
            
            # Check max records
            if max_records and stats["valid_records"] >= max_records:
                logger.info(f"Reached max_records limit: {max_records}")
                break
    
    logger.info(f"Conversion complete: {stats}")
    return stats


def convert_dataset_directory(
    dataset_dir: str,
    output_dir: str,
    max_per_file: Optional[int] = None,
    filter_reviewed: bool = False,
    merge_output: bool = False
) -> Dict[str, Dict[str, int]]:
    """
    Convert all TSV files in a directory to JSONL format.
    
    Args:
        dataset_dir: Directory containing TSV files
        output_dir: Output directory for JSONL files
        max_per_file: Maximum records per file
        filter_reviewed: Only include reviewed entries
        merge_output: If True, merge all outputs into a single file
    
    Returns:
        Statistics for each file
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    tsv_files = list(dataset_path.glob("*.tsv"))
    
    if not tsv_files:
        logger.warning(f"No TSV files found in {dataset_dir}")
        return {}
    
    all_stats = {}
    merged_file = None
    
    if merge_output:
        merged_file = output_path / "merged_dataset.jsonl"
        merged_file.unlink(missing_ok=True)  # Remove if exists
    
    for tsv_file in tsv_files:
        logger.info(f"Processing {tsv_file.name}...")
        
        if merge_output:
            output_jsonl = merged_file
            # Append mode for merged file
            with open(tsv_file, 'r', encoding='utf-8') as infile, \
                 open(output_jsonl, 'a', encoding='utf-8') as outfile:
                
                reader = csv.DictReader(infile, delimiter='\t')
                stats = {"total_rows": 0, "valid_records": 0, "skipped": 0, "filtered": 0}
                
                for row in reader:
                    stats["total_rows"] += 1
                    record = parse_tsv_row(row, tsv_file.name)
                    
                    if record is None:
                        stats["skipped"] += 1
                        continue
                    
                    if filter_reviewed and not record["extra_features"].get("reviewed", False):
                        stats["filtered"] += 1
                        continue
                    
                    outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
                    stats["valid_records"] += 1
                    
                    if max_per_file and stats["valid_records"] >= max_per_file:
                        break
        else:
            output_jsonl = output_path / f"{tsv_file.stem}.jsonl"
            stats = convert_tsv_to_jsonl(
                str(tsv_file),
                str(output_jsonl),
                max_records=max_per_file,
                filter_reviewed=filter_reviewed
            )
        
        all_stats[tsv_file.name] = stats
    
    # Summary
    total_valid = sum(s["valid_records"] for s in all_stats.values())
    total_skipped = sum(s["skipped"] for s in all_stats.values())
    logger.info(f"\n=== Summary ===")
    logger.info(f"Total valid records: {total_valid}")
    logger.info(f"Total skipped: {total_skipped}")
    
    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert UniProt TSV dataset to JSONL format for codon optimization training"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input TSV file or directory containing TSV files"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file or directory"
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Maximum number of records per file (default: all)"
    )
    parser.add_argument(
        "--filter-reviewed",
        action="store_true",
        help="Only include reviewed (SwissProt) entries"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all files into a single JSONL (when input is a directory)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_dir():
        # Convert entire directory
        convert_dataset_directory(
            str(input_path),
            args.output,
            max_per_file=args.max_records,
            filter_reviewed=args.filter_reviewed,
            merge_output=args.merge
        )
    elif input_path.is_file():
        # Convert single file
        stats = convert_tsv_to_jsonl(
            str(input_path),
            args.output,
            max_records=args.max_records,
            filter_reviewed=args.filter_reviewed
        )
        logger.info(f"Conversion complete: {stats}")
    else:
        logger.error(f"Input path does not exist: {input_path}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

