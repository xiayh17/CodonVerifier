#!/bin/bash
# MMseqs2 Direct Testing Script
# This script allows you to directly test MMseqs2 commands with different parameters

echo "=== MMseqs2 Direct Testing Script ==="
echo "This script helps you test MMseqs2 commands directly with different parameters"
echo

# Default parameters
DATABASE="/data/mmseqs_db/production/UniRef50"
THREADS=20
GPU_ID=0
USE_GPU=true
BATCH_SIZE=10
MAX_SEQS=1000
SENSITIVITY=7.5
EVALUE=0.001
MIN_SEQ_ID=0.3
COVERAGE=0.5
MEMORY_LIMIT=12288

# Function to show current parameters
show_params() {
    echo "=== Current Parameters ==="
    echo "Database: $DATABASE"
    echo "Threads: $THREADS"
    echo "GPU ID: $GPU_ID"
    echo "Use GPU: $USE_GPU"
    echo "Batch Size: $BATCH_SIZE"
    echo "Max Sequences: $MAX_SEQS"
    echo "Sensitivity: $SENSITIVITY"
    echo "E-value: $EVALUE"
    echo "Min Seq ID: $MIN_SEQ_ID"
    echo "Coverage: $COVERAGE"
    echo "Memory Limit: $MEMORY_LIMIT"
    echo
}

# Function to generate MMseqs2 command
generate_command() {
    local query_db="$1"
    local result_db="$2"
    local tmp_dir="$3"
    
    echo "=== Generated MMseqs2 Command ==="
    echo "mmseqs search \\"
    echo "  $query_db \\"
    echo "  $DATABASE \\"
    echo "  $result_db \\"
    echo "  $tmp_dir \\"
    echo "  --threads $THREADS \\"
    echo "  -e $EVALUE \\"
    echo "  --min-seq-id $MIN_SEQ_ID \\"
    echo "  -c $COVERAGE \\"
    echo "  --alignment-mode 3 \\"
    echo "  --max-seqs $MAX_SEQS \\"
    echo "  -s $SENSITIVITY"
    
    if [ "$USE_GPU" = true ]; then
        echo "  --gpu $GPU_ID \\"
        echo "  --split-memory-limit $MEMORY_LIMIT"
    fi
    echo
}

# Function to run a test
run_test() {
    local test_name="$1"
    local num_sequences="$2"
    
    echo "=== Running Test: $test_name ==="
    echo "Creating test data with $num_sequences sequences..."
    
    # Create test directory
    local test_dir="/tmp/mmseqs_test_$(date +%s)"
    mkdir -p "$test_dir"
    
    # Create test FASTA file
    local test_fasta="$test_dir/test_queries.fasta"
    echo "Creating test FASTA file: $test_fasta"
    
    # Generate test sequences (simple example)
    for i in $(seq 1 $num_sequences); do
        echo ">test_sequence_$i" >> "$test_fasta"
        echo "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL" >> "$test_fasta"
    done
    
    # Create query database
    local query_db="$test_dir/query_db"
    echo "Creating query database..."
    mmseqs createdb "$test_fasta" "$query_db"
    
    # Create result database path
    local result_db="$test_dir/result_db"
    local tmp_dir="$test_dir/tmp"
    mkdir -p "$tmp_dir"
    
    # Generate and show command
    generate_command "$query_db" "$result_db" "$tmp_dir"
    
    # Ask for confirmation
    echo "Do you want to run this command? (y/n)"
    read -r response
    if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
        echo "Running MMseqs2 search..."
        echo "Start time: $(date)"
        
        # Build the command
        local cmd="mmseqs search $query_db $DATABASE $result_db $tmp_dir --threads $THREADS -e $EVALUE --min-seq-id $MIN_SEQ_ID -c $COVERAGE --alignment-mode 3 --max-seqs $MAX_SEQS -s $SENSITIVITY"
        
        if [ "$USE_GPU" = true ]; then
            cmd="$cmd --gpu $GPU_ID --split-memory-limit $MEMORY_LIMIT"
        fi
        
        # Run the command with timing
        time $cmd
        
        echo "End time: $(date)"
        echo "Test completed!"
        
        # Convert results to TSV
        local result_tsv="$test_dir/result.tsv"
        echo "Converting results to TSV..."
        mmseqs convertalis "$query_db" "$DATABASE" "$result_db" "$result_tsv" --format-output "query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits"
        
        # Show results summary
        echo "=== Results Summary ==="
        echo "Total hits: $(wc -l < "$result_tsv")"
        echo "Results saved to: $result_tsv"
        echo "Test directory: $test_dir"
        
    else
        echo "Test cancelled."
    fi
    
    echo
}

# Function to show menu
show_menu() {
    echo "=== MMseqs2 Test Menu ==="
    echo "1. Show current parameters"
    echo "2. Modify parameters"
    echo "3. Run quick test (1 sequence)"
    echo "4. Run small test (5 sequences)"
    echo "5. Run medium test (10 sequences)"
    echo "6. Run large test (50 sequences)"
    echo "7. Generate command only (no execution)"
    echo "8. Test different batch sizes"
    echo "9. Test different sensitivity values"
    echo "10. Test GPU vs CPU performance"
    echo "11. Exit"
    echo
}

# Function to modify parameters
modify_params() {
    echo "=== Modify Parameters ==="
    echo "Current parameters:"
    show_params
    
    echo "Enter new values (press Enter to keep current):"
    
    echo -n "Threads [$THREADS]: "
    read -r input
    if [ -n "$input" ]; then THREADS="$input"; fi
    
    echo -n "GPU ID [$GPU_ID]: "
    read -r input
    if [ -n "$input" ]; then GPU_ID="$input"; fi
    
    echo -n "Use GPU (true/false) [$USE_GPU]: "
    read -r input
    if [ -n "$input" ]; then USE_GPU="$input"; fi
    
    echo -n "Max Sequences [$MAX_SEQS]: "
    read -r input
    if [ -n "$input" ]; then MAX_SEQS="$input"; fi
    
    echo -n "Sensitivity [$SENSITIVITY]: "
    read -r input
    if [ -n "$input" ]; then SENSITIVITY="$input"; fi
    
    echo -n "E-value [$EVALUE]: "
    read -r input
    if [ -n "$input" ]; then EVALUE="$input"; fi
    
    echo -n "Memory Limit [$MEMORY_LIMIT]: "
    read -r input
    if [ -n "$input" ]; then MEMORY_LIMIT="$input"; fi
    
    echo "Parameters updated!"
    show_params
}

# Function to test different batch sizes
test_batch_sizes() {
    echo "=== Testing Different Batch Sizes ==="
    
    local sizes=(1 5 10 25 50 100)
    
    for size in "${sizes[@]}"; do
        echo "Testing batch size: $size"
        run_test "Batch Size $size" "$size"
        
        echo "Continue to next batch size? (y/n)"
        read -r response
        if [ "$response" != "y" ] && [ "$response" != "Y" ]; then
            break
        fi
    done
}

# Function to test different sensitivity values
test_sensitivity() {
    echo "=== Testing Different Sensitivity Values ==="
    
    local sensitivities=(5.0 6.0 7.0 7.5 8.0 9.0)
    
    for sens in "${sensitivities[@]}"; do
        echo "Testing sensitivity: $sens"
        local old_sens="$SENSITIVITY"
        SENSITIVITY="$sens"
        run_test "Sensitivity $sens" 5
        SENSITIVITY="$old_sens"
        
        echo "Continue to next sensitivity? (y/n)"
        read -r response
        if [ "$response" != "y" ] && [ "$response" != "Y" ]; then
            break
        fi
    done
}

# Function to test GPU vs CPU
test_gpu_cpu() {
    echo "=== Testing GPU vs CPU Performance ==="
    
    echo "Testing CPU performance..."
    USE_GPU=false
    run_test "CPU Test" 10
    
    echo "Testing GPU performance..."
    USE_GPU=true
    run_test "GPU Test" 10
    
    echo "GPU vs CPU test completed!"
}

# Main menu loop
while true; do
    show_menu
    echo -n "Select option (1-11): "
    read -r choice
    
    case $choice in
        1)
            show_params
            ;;
        2)
            modify_params
            ;;
        3)
            run_test "Quick Test" 1
            ;;
        4)
            run_test "Small Test" 5
            ;;
        5)
            run_test "Medium Test" 10
            ;;
        6)
            run_test "Large Test" 50
            ;;
        7)
            generate_command "/tmp/query_db" "/tmp/result_db" "/tmp/tmp"
            ;;
        8)
            test_batch_sizes
            ;;
        9)
            test_sensitivity
            ;;
        10)
            test_gpu_cpu
            ;;
        11)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid option. Please select 1-11."
            ;;
    esac
    
    echo
    echo "Press Enter to continue..."
    read -r
done
