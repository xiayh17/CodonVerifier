#!/usr/bin/env python3
"""
MMseqs2 Command Generator
Generate and test MMseqs2 commands with different parameters
"""

import os
import subprocess
import time
from datetime import datetime

class MMseqsCommandGenerator:
    def __init__(self):
        self.database = "/data/mmseqs_db/production/UniRef50"
        self.default_params = {
            'threads': 20,
            'gpu_id': 0,
            'use_gpu': True,
            'max_seqs': 1000,
            'sensitivity': 7.5,
            'evalue': 0.001,
            'min_seq_id': 0.3,
            'coverage': 0.5,
            'memory_limit': 12288
        }
        self.current_params = self.default_params.copy()
    
    def generate_command(self, query_db, result_db, tmp_dir):
        """Generate MMseqs2 search command"""
        cmd = [
            'mmseqs', 'search',
            query_db,
            self.database,
            result_db,
            tmp_dir,
            '--threads', str(self.current_params['threads']),
            '-e', str(self.current_params['evalue']),
            '--min-seq-id', str(self.current_params['min_seq_id']),
            '-c', str(self.current_params['coverage']),
            '--alignment-mode', '3',
            '--max-seqs', str(self.current_params['max_seqs']),
            '-s', str(self.current_params['sensitivity'])
        ]
        
        if self.current_params['use_gpu']:
            cmd.extend([
                '--gpu', str(self.current_params['gpu_id']),
                '--split-memory-limit', str(self.current_params['memory_limit'])
            ])
        
        return cmd
    
    def print_command(self, query_db, result_db, tmp_dir):
        """Print formatted command"""
        cmd = self.generate_command(query_db, result_db, tmp_dir)
        print("=== Generated MMseqs2 Command ===")
        print(" ".join(cmd))
        print()
        return cmd
    
    def create_test_data(self, num_sequences, output_dir):
        """Create test FASTA data"""
        os.makedirs(output_dir, exist_ok=True)
        fasta_file = os.path.join(output_dir, "test_queries.fasta")
        
        with open(fasta_file, 'w') as f:
            for i in range(1, num_sequences + 1):
                f.write(f">test_sequence_{i}\n")
                f.write("MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL\n")
        
        return fasta_file
    
    def run_test(self, num_sequences, test_name="Test"):
        """Run a complete test"""
        print(f"=== {test_name} ===")
        print(f"Sequences: {num_sequences}")
        print(f"Parameters: {self.current_params}")
        print()
        
        # Create test directory
        test_dir = f"/tmp/mmseqs_test_{int(time.time())}"
        os.makedirs(test_dir, exist_ok=True)
        
        try:
            # Create test data
            fasta_file = self.create_test_data(num_sequences, test_dir)
            print(f"Created test data: {fasta_file}")
            
            # Create query database
            query_db = os.path.join(test_dir, "query_db")
            print("Creating query database...")
            subprocess.run(['mmseqs', 'createdb', fasta_file, query_db], check=True)
            
            # Prepare paths
            result_db = os.path.join(test_dir, "result_db")
            tmp_dir = os.path.join(test_dir, "tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            
            # Generate and show command
            cmd = self.print_command(query_db, result_db, tmp_dir)
            
            # Ask for confirmation
            response = input("Run this command? (y/n): ").lower()
            if response != 'y':
                print("Test cancelled.")
                return
            
            # Run the command
            print("Running MMseqs2 search...")
            print(f"Start time: {datetime.now()}")
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            end_time = time.time()
            
            if result.returncode == 0:
                print("✓ Search completed successfully!")
                print(f"Duration: {end_time - start_time:.2f} seconds")
                
                # Convert results
                result_tsv = os.path.join(test_dir, "result.tsv")
                print("Converting results to TSV...")
                subprocess.run([
                    'mmseqs', 'convertalis',
                    query_db, self.database, result_db, result_tsv,
                    '--format-output', 'query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits'
                ], check=True)
                
                # Show results
                if os.path.exists(result_tsv):
                    with open(result_tsv, 'r') as f:
                        lines = f.readlines()
                    print(f"Total hits: {len(lines)}")
                    print(f"Results saved to: {result_tsv}")
                
            else:
                print("✗ Search failed!")
                print(f"Error: {result.stderr}")
            
            print(f"Test directory: {test_dir}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    def quick_test(self, num_sequences=5):
        """Quick test with current parameters"""
        self.run_test(num_sequences, f"Quick Test ({num_sequences} sequences)")
    
    def parameter_test(self, param_name, values, num_sequences=5):
        """Test different values for a parameter"""
        original_value = self.current_params[param_name]
        
        print(f"=== Testing {param_name} ===")
        for value in values:
            print(f"\nTesting {param_name} = {value}")
            self.current_params[param_name] = value
            self.run_test(num_sequences, f"{param_name}={value}")
            
            response = input("Continue to next value? (y/n): ").lower()
            if response != 'y':
                break
        
        # Restore original value
        self.current_params[param_name] = original_value
    
    def batch_size_test(self):
        """Test different batch sizes"""
        batch_sizes = [1, 5, 10, 25, 50, 100]
        self.parameter_test('max_seqs', batch_sizes, 10)
    
    def sensitivity_test(self):
        """Test different sensitivity values"""
        sensitivities = [5.0, 6.0, 7.0, 7.5, 8.0, 9.0]
        self.parameter_test('sensitivity', sensitivities, 5)
    
    def gpu_cpu_test(self):
        """Test GPU vs CPU performance"""
        print("=== GPU vs CPU Performance Test ===")
        
        # Test CPU
        print("\n1. Testing CPU performance...")
        self.current_params['use_gpu'] = False
        self.run_test(10, "CPU Test")
        
        # Test GPU
        print("\n2. Testing GPU performance...")
        self.current_params['use_gpu'] = True
        self.run_test(10, "GPU Test")
    
    def show_menu(self):
        """Show interactive menu"""
        while True:
            print("\n=== MMseqs2 Command Generator ===")
            print("1. Show current parameters")
            print("2. Modify parameters")
            print("3. Quick test (5 sequences)")
            print("4. Test different batch sizes")
            print("5. Test different sensitivity values")
            print("6. Test GPU vs CPU performance")
            print("7. Generate command only")
            print("8. Custom test")
            print("9. Exit")
            
            choice = input("\nSelect option (1-9): ").strip()
            
            if choice == '1':
                print("\nCurrent parameters:")
                for key, value in self.current_params.items():
                    print(f"  {key}: {value}")
            
            elif choice == '2':
                self.modify_parameters()
            
            elif choice == '3':
                self.quick_test()
            
            elif choice == '4':
                self.batch_size_test()
            
            elif choice == '5':
                self.sensitivity_test()
            
            elif choice == '6':
                self.gpu_cpu_test()
            
            elif choice == '7':
                self.print_command("/tmp/query_db", "/tmp/result_db", "/tmp/tmp")
            
            elif choice == '8':
                num_seq = int(input("Number of sequences: "))
                self.run_test(num_seq, f"Custom Test ({num_seq} sequences)")
            
            elif choice == '9':
                print("Exiting...")
                break
            
            else:
                print("Invalid option. Please select 1-9.")
    
    def modify_parameters(self):
        """Modify parameters interactively"""
        print("\n=== Modify Parameters ===")
        print("Enter new values (press Enter to keep current):")
        
        for key, current_value in self.current_params.items():
            new_value = input(f"{key} [{current_value}]: ").strip()
            if new_value:
                if key in ['threads', 'gpu_id', 'max_seqs', 'memory_limit']:
                    self.current_params[key] = int(new_value)
                elif key in ['sensitivity', 'evalue', 'min_seq_id', 'coverage']:
                    self.current_params[key] = float(new_value)
                elif key == 'use_gpu':
                    self.current_params[key] = new_value.lower() == 'true'
                else:
                    self.current_params[key] = new_value
        
        print("Parameters updated!")

if __name__ == "__main__":
    generator = MMseqsCommandGenerator()
    generator.show_menu()
