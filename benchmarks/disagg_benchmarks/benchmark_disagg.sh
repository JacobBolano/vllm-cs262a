set -ex
# 1. Set initial GPU cluster configuration (3P 1D, 2P 2D, 1P 3D, 4 CP)
# 1a) For Chunked Prefill - vary the Max num batch tokens: 4k, 8k, 16k
# 2. Vary different benchmark datasets: Random, ShareGPT
# 2a) For Random we can change the Input len–Output len: 1k-100, 1k-1k, 100-1k, 8k-32
# 2b) For ShareGPT we can change the output len: 32, 100, 1000
# 3. QPS: 2, 8, 14
# 4. Number of Requests: 10, 100, 1000

# Start with Random Dataset
main() {
    chunked_prefill_max=(4000 8000 16000)
    input_output=(
    "1000 100"
    "1000 1000"
    "100 1000"
    "8000 32"
    )
    for pair in "${pairs[@]}"; do
        read input output <<< "$pair"
        for qps in 2,8,14; do
            for num_requests in 10, 100, 1000; do
                results_folder="random-2p2d-qps"$qps"-numreq"$num_requests"-io"$input":"$output
                rm -rf results_disagg
                mkdir results_disagg
                ./batch_2p2d.sh --dataset random \
                                --qps qps \
                                --num_requests num_requests \
                                --random-input-len input \
                                --random-output-len output \
                                --results_folder results_folder
            done
        done
    done

    # Next Run ShareGPT
    for input in 32, 100, 1000; do
        for qps in 2,8,14; do
            for num_requests in 10, 100, 1000; do
                ./batch_2p2d.sh --dataset shareGPT \
                                --qps qps \
                                --num_requests num_requests \
                                --sharegpt-input-len input \
            done
        done
    done
}

main "$@"
