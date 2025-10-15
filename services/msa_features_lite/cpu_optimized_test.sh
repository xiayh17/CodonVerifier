#!/bin/bash
# CPU-Optimized MMseqs2 Test Script
# Since MMseqs2 was compiled without CUDA support, optimize for CPU performance

echo "=== CPU-Optimized MMseqs2 Test ==="
echo "MMseqs2 was compiled without CUDA support, optimizing for CPU performance"
echo

# CPU-optimized parameters
DATABASE="/data/mmseqs_db/production/UniRef50"
THREADS=32  # Use more threads for CPU
USE_GPU=false  # Disable GPU
MAX_SEQS=1000
SENSITIVITY=7.5
EVALUE=0.001
MIN_SEQ_ID=0.3
COVERAGE=0.5

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

# Function to run CPU-optimized MMseqs2 test
run_cpu_test() {
    local num_seq=$1
    local test_name=$2
    
    echo "=== $test_name ==="
    echo "Sequences: $num_seq"
    echo "Parameters: threads=$THREADS, cpu_only=true, max_seqs=$MAX_SEQS, sensitivity=$SENSITIVITY"
    echo
    
    # Create test directory
    local test_dir="/tmp/mmseqs_cpu_$(date +%s)"
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
    
    # Build CPU-optimized command (no GPU parameters)
    local cmd="mmseqs search $query_db $DATABASE $result_db $tmp_dir --threads $THREADS -e $EVALUE --min-seq-id $MIN_SEQ_ID -c $COVERAGE --alignment-mode 3 --max-seqs $MAX_SEQS -s $SENSITIVITY"
    
    echo "Command: $cmd"
    echo
    
    # Ask for confirmation
    echo "Run this CPU-optimized command? (y/n)"
    read -r response
    if [ "$response" != "y" ] && [ "$response" != "Y" ]; then
        echo "Test cancelled."
        return
    fi
    
    # Run the command
    echo "Running MMseqs2 search (CPU only)..."
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

# Function to test different thread counts
test_thread_counts() {
    echo "=== Testing Different Thread Counts ==="
    
    local thread_counts=(8 16 24 32 40 48)
    
    for threads in "${thread_counts[@]}"; do
        echo "Testing threads: $threads"
        local old_threads="$THREADS"
        THREADS="$threads"
        run_cpu_test 5 "Threads $threads"
        THREADS="$old_threads"
        
        echo "Continue to next thread count? (y/n)"
        read -r response
        if [ "$response" != "y" ] && [ "$response" != "Y" ]; then
            break
        fi
    done
}

# Function to test different batch sizes
test_batch_sizes() {
    echo "=== Testing Different Batch Sizes ==="
    
    local batch_sizes=(100 500 750 1000 1500 2000)
    
    for batch_size in "${batch_sizes[@]}"; do
        echo "Testing max_seqs: $batch_size"
        local old_max_seqs="$MAX_SEQS"
        MAX_SEQS="$batch_size"
        run_cpu_test 10 "Batch Size $batch_size"
        MAX_SEQS="$old_max_seqs"
        
        echo "Continue to next batch size? (y/n)"
        read -r response
        if [ "$response" != "y" ] && [ "$response" != "Y" ]; then
            break
        fi
    done
}

# Function to show menu
show_menu() {
    echo "=== CPU-Optimized MMseqs2 Test Menu ==="
    echo "1. Quick test (1 sequence)"
    echo "2. Small test (5 sequences)"
    echo "3. Medium test (10 sequences)"
    echo "4. Large test (50 sequences)"
    echo "5. Test different thread counts"
    echo "6. Test different batch sizes"
    echo "7. Test different sensitivity values"
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
    echo "  max_seqs: $MAX_SEQS"
    echo "  sensitivity: $SENSITIVITY"
    echo "  evalue: $EVALUE"
    echo "  min_seq_id: $MIN_SEQ_ID"
    echo "  coverage: $COVERAGE"
    echo
    
    echo "Enter new values (press Enter to keep current):"
    
    echo -n "threads [$THREADS]: "
    read -r input
    if [ -n "$input" ]; then THREADS="$input"; fi
    
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
    
    echo "Parameters updated!"
}

# Main menu loop
while true; do
    show_menu
    echo -n "Select option (1-10): "
    read -r choice
    
    case $choice in
        1)
            run_cpu_test 1 "Quick Test"
            ;;
        2)
            run_cpu_test 5 "Small Test"
            ;;
        3)
            run_cpu_test 10 "Medium Test"
            ;;
        4)
            run_cpu_test 50 "Large Test"
            ;;
        5)
            test_thread_counts
            ;;
        6)
            test_batch_sizes
            ;;
        7)
            echo "Testing different sensitivity values..."
            local sensitivities=(5.0 6.0 7.0 7.5 8.0 9.0)
            for sens in "${sensitivities[@]}"; do
                echo "Testing sensitivity: $sens"
                local old_sens="$SENSITIVITY"
                SENSITIVITY="$sens"
                run_cpu_test 5 "Sensitivity $sens"
                SENSITIVITY="$old_sens"
                
                echo "Continue to next sensitivity? (y/n)"
                read -r response
                if [ "$response" != "y" ] && [ "$response" != "Y" ]; then
                    break
                fi
            done
            ;;
        8)
            modify_params
            ;;
        9)
            echo "Current parameters:"
            echo "  threads: $THREADS"
            echo "  max_seqs: $MAX_SEQS"
            echo "  sensitivity: $SENSITIVITY"
            echo "  evalue: $EVALUE"
            echo "  min_seq_id: $MIN_SEQ_ID"
            echo "  coverage: $COVERAGE"
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
