#!/usr/bin/env python3
"""
Test script for the complete feature generation pipeline

This script tests the new generate_complete_features.py script
with a small sample to ensure it works correctly.
"""

import subprocess
import sys
from pathlib import Path

# Ensure project root is on sys.path so imports/modules resolve when run from anywhere
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
import tempfile
import json

def test_complete_features():
    """Test the complete features generation with a small sample"""
    
    print("🧪 Testing Complete Feature Generation Pipeline")
    print("=" * 60)
    
    # Check if we have a test TSV file
    test_tsv = "data/2025_bio-os_data/dataset/Ec.tsv"
    if not Path(test_tsv).exists():
        print(f"❌ Test TSV file not found: {test_tsv}")
        print("Please ensure the test data exists before running this test.")
        return False
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        output_file = f.name
    
    try:
        print(f"📁 Input: {test_tsv}")
        print(f"📁 Output: {output_file}")
        print(f"🔢 Limit: 10 records (for testing)")
        print()
        
        # Run the complete features generation with limit
        cmd = [
            "python", "scripts/generate_complete_features.py",
            "--input", test_tsv,
            "--output", output_file,
            "--limit", "10",
            "--use-docker"
        ]
        
        print(f"🚀 Running: {' '.join(cmd)}")
        print()
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()))
        
        if result.returncode != 0:
            print("❌ Pipeline failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
        
        print("✅ Pipeline completed successfully!")
        print()
        print("STDOUT:")
        print(result.stdout)
        
        # Check output file
        if Path(output_file).exists():
            print(f"📊 Output file created: {output_file}")
            
            # Count records
            with open(output_file, 'r') as f:
                records = [json.loads(line) for line in f if line.strip()]
            
            print(f"📈 Records generated: {len(records)}")
            
            if records:
                # Check first record structure
                first_record = records[0]
                print("🔍 First record structure:")
                print(f"   - protein_id: {first_record.get('protein_id', 'N/A')}")
                print(f"   - sequence length: {len(first_record.get('sequence', ''))}")
                print(f"   - host: {first_record.get('host', 'N/A')}")
                print(f"   - expression: {first_record.get('expression', 'N/A')}")
                
                # Check extra_features
                extra_features = first_record.get('extra_features', {})
                print(f"   - extra_features count: {len(extra_features)}")
                
                # Show some feature categories
                struct_features = [k for k in extra_features.keys() if k.startswith('struct_')]
                evo_features = [k for k in extra_features.keys() if k.startswith('evo_')]
                evo2_features = [k for k in extra_features.keys() if k.startswith('evo2_')]
                ctx_features = [k for k in extra_features.keys() if k.startswith('ctx_')]
                
                print(f"   - struct_ features: {len(struct_features)}")
                print(f"   - evo_ features: {len(evo_features)}")
                print(f"   - evo2_ features: {len(evo2_features)}")
                print(f"   - ctx_ features: {len(ctx_features)}")
                
                if struct_features:
                    print(f"     Example struct: {struct_features[0]}")
                if evo_features:
                    print(f"     Example evo: {evo_features[0]}")
                if evo2_features:
                    print(f"     Example evo2: {evo2_features[0]}")
                if ctx_features:
                    print(f"     Example ctx: {ctx_features[0]}")
            
            print()
            print("🎉 Test completed successfully!")
            return True
        else:
            print("❌ Output file was not created!")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False
    
    finally:
        # Clean up
        if Path(output_file).exists():
            Path(output_file).unlink()
            print(f"🧹 Cleaned up test file: {output_file}")


if __name__ == "__main__":
    success = test_complete_features()
    sys.exit(0 if success else 1)
