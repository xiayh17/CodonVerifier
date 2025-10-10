#!/usr/bin/env python3
"""
Add default Evo2 features to existing enhanced JSONL file

This script adds 8 default Evo2 features to sequences that only have
38 enhanced features (struct + evo + ctx), bringing the total to 46.

Author: CodonVerifier Team
Date: 2025-10-07
"""

import json
import argparse
import sys
from pathlib import Path


def add_evo2_features(input_file: str, output_file: str, limit: int = None):
    """
    Add default Evo2 features to sequences
    
    Args:
        input_file: Input JSONL file
        output_file: Output JSONL file  
        limit: Optional limit on number of records
    """
    processed = 0
    added = 0
    skipped = 0
    
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    
    with open(input_file) as f_in, open(output_file, 'w') as f_out:
        for line_num, line in enumerate(f_in, 1):
            if limit and processed >= limit:
                print(f"Reached limit of {limit} records")
                break
            
            if not line.strip():
                continue
            
            try:
                record = json.loads(line.strip())
                
                # Check if extra_features exist
                if 'extra_features' not in record or not record['extra_features']:
                    print(f"Warning: Line {line_num} has no extra_features, skipping")
                    skipped += 1
                    continue
                
                extra_features = record['extra_features']
                
                # Check if Evo2 features already exist
                has_evo2 = any(k.startswith('evo2_') for k in extra_features.keys())
                
                if has_evo2:
                    # Already has Evo2, just write as-is
                    f_out.write(json.dumps(record) + '\n')
                    processed += 1
                    continue
                
                # Calculate Evo2 defaults based on sequence
                sequence = record.get('sequence', '')
                if sequence:
                    gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
                    # Approximate codon entropy
                    codon_entropy = min(3.0, len(set(sequence)) / 20.0 * 5.0)
                else:
                    gc_content = 0.5
                    codon_entropy = 2.5
                
                # Add 8 Evo2 features
                extra_features.update({
                    'evo2_avg_confidence': 0.75,
                    'evo2_max_confidence': 0.95,
                    'evo2_min_confidence': 0.55,
                    'evo2_std_confidence': 0.12,
                    'evo2_avg_loglik': -2.5,
                    'evo2_perplexity': 15.0,
                    'evo2_gc_content': gc_content,
                    'evo2_codon_entropy': codon_entropy
                })
                
                # Write updated record
                f_out.write(json.dumps(record) + '\n')
                processed += 1
                added += 1
                
                if processed % 1000 == 0:
                    print(f"Processed {processed} records ({added} updated)...")
            
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                skipped += 1
                continue
    
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Total processed: {processed}")
    print(f"  Evo2 features added: {added}")
    print(f"  Skipped: {skipped}")
    print("="*60)
    
    return {
        'processed': processed,
        'added': added,
        'skipped': skipped
    }


def main():
    parser = argparse.ArgumentParser(
        description="Add default Evo2 features to enhanced JSONL file"
    )
    
    parser.add_argument('--input', required=True, help="Input JSONL file")
    parser.add_argument('--output', required=True, help="Output JSONL file")
    parser.add_argument('--limit', type=int, help="Limit number of records (for testing)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process
    stats = add_evo2_features(args.input, args.output, args.limit)
    
    print(f"\n✓ Complete! Output saved to: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

