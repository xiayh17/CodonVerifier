#!/bin/bash
# Quick MMseqs2 Test Script
# Simple script to test MMseqs2 commands directly

echo "=== Quick MMseqs2 Test ==="
echo "This script helps you test MMseqs2 commands quickly"
echo

# Default parameters
DATABASE="/data/mmseqs_db/production/UniRef50"
THREADS=20
GPU_ID=0
USE_GPU=true
MAX_SEQS=1000
SENSITIVITY=7.5
EVALUE=0.001
MIN_SEQ_ID=0.3
COVERAGE=0.5
MEMORY_LIMIT=12288

# Function to create test data
create_test_data() {
    local num_seq=$1
    local output_file=$2
    
    echo "Creating test data with $num_seq sequences..."
    > "$output_file"
    
    for i in $(seq 1 $num_seq); do
        echo ">test_sequence_$i" >> "$output_file"
        echo "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL" >> "$output_file"
    done
    
    echo "Test data created: $output_file"
}

# Function to run MMseqs2 test
run_mmseqs_test() {
    local num_seq=$1
    local test_name=$2
    
    echo "=== $test_name ==="
    echo "Sequences: $num_seq"
    echo "Parameters: threads=$THREADS, gpu=$USE_GPU, max_seqs=$MAX_SEQS, sensitivity=$SENSITIVITY"
    echo
    
    # Create test directory
    local test_dir="/tmp/mmseqs_quick_$(date +%s)"
    mkdir -p "$test_dir"
    
    # Create test data
    local fasta_file="$test_dir/test.fasta"
    create_test_data "$num_seq" "$fasta_file"
    
    # Create query database
    local query_db="$test_dir/query_db"
    echo "Creating query database..."
    mmseqs createdb "$fasta_file" "$query_db"
    
    # Prepare paths
    local result_db="$test_dir/result_db"
    local tmp_dir="$test_dir/tmp"
    mkdir -p "$tmp_dir"
    
    # Build command
    local cmd="mmseqs search $query_db $DATABASE $result_db $tmp_dir --threads $THREADS -e $EVALUE --min-seq-id $MIN_SEQ_ID -c $COVERAGE --alignment-mode 3 --max-seqs $MAX_SEQS -s $SENSITIVITY"
    
    if [ "$USE_GPU" = true ]; then
        cmd="$cmd --gpu $GPU_ID --split-memory-limit $MEMORY_LIMIT"
    fi
    
    echo "Command: $cmd"
    echo
    
    # Ask for confirmation
    echo "Run this command? (y/n)"
    read -r response
    if [ "$response" != "y" ] && [ "$response" != "Y" ]; then
        echo "Test cancelled."
        return
    fi
    
    # Run the command
    echo "Running MMseqs2 search..."
    echo "Start time: $(date)"
    
    start_time=$(date +%s)
    $cmd
    end_time=$(date +%s)
    
    if [ $? -eq 0 ]; then
        echo "✓ Search completed successfully!"
        echo "Duration: $((end_time - start_time)) seconds"
        
        # Convert results
        local result_tsv="$test_dir/result.tsv"
        echo "Converting results to TSV..."
        mmseqs convertalis "$query_db" "$DATABASE" "$result_db" "$result_tsv" --format-output "query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits"
        
        # Show results
        if [ -f "$result_tsv" ]; then
            local hits=$(wc -l < "$result_tsv")
            echo "Total hits: $hits"
            echo "Results saved to: $result_tsv"
        fi
        
    else
        echo "✗ Search failed!"
    fi
    
    echo "Test directory: $test_dir"
    echo
}

# Function to show menu
show_menu() {
    echo "=== Quick MMseqs2 Test Menu ==="
    echo "1. Quick test (1 sequence)"
    echo "2. Small test (5 sequences)"
    echo "3. Medium test (10 sequences)"
    echo "4. Large test (50 sequences)"
    echo "5. Test different batch sizes"
    echo "6. Test different sensitivity values"
    echo "7. Test GPU vs CPU"
    echo "8. Modify parameters"
    echo "9. Show current parameters"
    echo "10. Exit"
    echo
}

# Function to modify parameters
modify_params() {
    echo "=== Modify Parameters ==="
    echo "Current parameters:"
    echo "  threads: $THREADS"
    echo "  gpu_id: $GPU_ID"
    echo "  use_gpu: $USE_GPU"
    echo "  max_seqs: $MAX_SEQS"
    echo "  sensitivity: $SENSITIVITY"
    echo "  evalue: $EVALUE"
    echo "  min_seq_id: $MIN_SEQ_ID"
    echo "  coverage: $COVERAGE"
    echo "  memory_limit: $MEMORY_LIMIT"
    echo
    
    echo "Enter new values (press Enter to keep current):"
    
    echo -n "threads [$THREADS]: "
    read -r input
    if [ -n "$input" ]; then THREADS="$input"; fi
    
    echo -n "gpu_id [$GPU_ID]: "
    read -r input
    if [ -n "$input" ]; then GPU_ID="$input"; fi
    
    echo -n "use_gpu (true/false) [$USE_GPU]: "
    read -r input
    if [ -n "$input" ]; then USE_GPU="$input"; fi
    
    echo -n "max_seqs [$MAX_SEQS]: "
    read -r input
    if [ -n "$input" ]; then MAX_SEQS="$input"; fi
    
    echo -n "sensitivity [$SENSITIVITY]: "
    read -r input
    if [ -n "$input" ]; then SENSITIVITY="$input"; fi
    
    echo -n "evalue [$EVALUE]: "
    read -r input
    if [ -n "$input" ]; then EVALUE="$input"; fi
    
    echo -n "min_seq_id [$MIN_SEQ_ID]: "
    read -r input
    if [ -n "$input" ]; then MIN_SEQ_ID="$input"; fi
    
    echo -n "coverage [$COVERAGE]: "
    read -r input
    if [ -n "$input" ]; then COVERAGE="$input"; fi
    
    echo -n "memory_limit [$MEMORY_LIMIT]: "
    read -r input
    if [ -n "$input" ]; then MEMORY_LIMIT="$input"; fi
    
    echo "Parameters updated!"
}

# Function to test different batch sizes
test_batch_sizes() {
    echo "=== Testing Different Batch Sizes ==="
    
    local sizes=(100 500 750 1000 1500 2000)
    
    for size in "${sizes[@]}"; do
        echo "Testing max_seqs: $size"
        local old_max_seqs="$MAX_SEQS"
        MAX_SEQS="$size"
        run_mmseqs_test 5 "Batch Size $size"
        MAX_SEQS="$old_max_seqs"
        
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
        run_mmseqs_test 5 "Sensitivity $sens"
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
    run_mmseqs_test 10 "CPU Test"
    
    echo "Testing GPU performance..."
    USE_GPU=true
    run_mmseqs_test 10 "GPU Test"
    
    echo "GPU vs CPU test completed!"
}

# Main menu loop
while true; do
    show_menu
    echo -n "Select option (1-10): "
    read -r choice
    
    case $choice in
        1)
            run_mmseqs_test 1 "Quick Test"
            ;;
        2)
            run_mmseqs_test 5 "Small Test"
            ;;
        3)
            run_mmseqs_test 10 "Medium Test"
            ;;
        4)
            run_mmseqs_test 50 "Large Test"
            ;;
        5)
            test_batch_sizes
            ;;
        6)
            test_sensitivity
            ;;
        7)
            test_gpu_cpu
            ;;
        8)
            modify_params
            ;;
        9)
            echo "Current parameters:"
            echo "  threads: $THREADS"
            echo "  gpu_id: $GPU_ID"
            echo "  use_gpu: $USE_GPU"
            echo "  max_seqs: $MAX_SEQS"
            echo "  sensitivity: $SENSITIVITY"
            echo "  evalue: $EVALUE"
            echo "  min_seq_id: $MIN_SEQ_ID"
            echo "  coverage: $COVERAGE"
            echo "  memory_limit: $MEMORY_LIMIT"
            ;;
        10)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid option. Please select 1-10."
            ;;
    esac
    
    echo
    echo "Press Enter to continue..."
    read -r
done
