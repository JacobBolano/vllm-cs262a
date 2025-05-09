#!/bin/bash

# Requirement: 2x GPUs.


# Model: meta-llama/Meta-Llama-3.1-8B-Instruct
# Query: 1024 input tokens, 6 output tokens, QPS 2/4/6/8, 100 requests
# Resource: 2x GPU
# Approaches:
# 2. Chunked prefill: 2 vllm instance with tp=4, equivalent to 1 tp=4 instance with QPS 4
# 3. Disaggregated prefill: 1 prefilling instance and 1 decoding instance
# Prefilling instance: max_output_token=1
# Decoding instance: force the input tokens be the same across requests to bypass prefilling

set -ex

kill_gpu_processes() {
  # kill all processes on GPU.
  pgrep pt_main_thread | xargs -r kill -9
  pgrep python3 | xargs -r kill -9
  for port in 8000 8100 8101 8102 8200; do lsof -t -i:$port | xargs -r kill -9; done
  sleep 1
}

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

launch_disagg_prefill() {
  model="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8" 
  # disagg prefill
  CUDA_VISIBLE_DEVICES=1 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8100 \
    --max-model-len 15000 \
    --gpu-memory-utilization 0.6 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":4,"kv_buffer_size":4e9}' &

    CUDA_VISIBLE_DEVICES=2 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8101 \
    --max-model-len 15000 \
    --gpu-memory-utilization 0.6 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":1,"kv_parallel_size":4,"kv_buffer_size":4e9}' &

  CUDA_VISIBLE_DEVICES=3 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8102 \
    --max-model-len 15000 \
    --gpu-memory-utilization 0.6 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":2,"kv_parallel_size":4,"kv_buffer_size":4e9}' &

    CUDA_VISIBLE_DEVICES=4 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8200 \
    --max-model-len 15000 \
    --gpu-memory-utilization 0.6 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":3,"kv_parallel_size":4,"kv_buffer_size":4e9}' &

  wait_for_server 8100
  wait_for_server 8101
  wait_for_server 8102
  wait_for_server 8200
  python3 disagg_3p1d_proxy_server.py &
  sleep 1
}


benchmark() {
  results_folder=$4
  model="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"
  dataset_name="random"
  dataset_path="r"
  num_prompts=$6
  qps=$1
  input_len=$5
  output_len=$2
  tag=$3
  
  python3 ../benchmark_serving.py \
          --backend vllm \
          --model $model \
          --dataset-name $dataset_name \
          --dataset-path $dataset_path \
          --random-input-len $input_len \
          --random-output-len "$output_len" \
          --num-prompts $num_prompts \
          --port 8000 \
          --save-result \
          --result-dir $results_folder \
          --result-filename "$tag"_inputLen:"$input_len"_outputLen:"$output_len".json \
          --ignore-eos \
          --request-rate "$qps"

  sleep 2
}


main() {

  (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
  (which jq) || (apt-get -y install jq)
  (which socat) || (apt-get -y install socat)
  (which lsof) || (apt-get -y install lsof)

  pip install quart httpx matplotlib aiohttp datasets

  cd "$(dirname "$0")"

  rm -rf 3p1d_results_50req_8qps_newRatio
  mkdir 3p1d_results_50req_8qps_newRatio

  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

  start_point=0
  counter=0

   # Define input/output pairs as an indexed array of strings
  input_output_pairs=(
    "10 1000"
    "50 1000"
    "250 1000"
    "1000 800"
    "1000 160"
    "1000 32"
    "625 4"
  )

  # launch_disagg_prefill
  # Loop through the QPS values, input/output pairs, and num_reqs
  for qps in 8; do
    for pair in "${input_output_pairs[@]}"; do
      # Split the pair string into input_len and output_len
      input_len_i=$(echo $pair | awk '{print $1}')
      output_len_i=$(echo $pair | awk '{print $2}')

      for num_reqs in 50; do
        counter=$((counter + 1))

        # if [ $counter -gt $start_point ]; then
          launch_disagg_prefill
          echo "Running iteration $counter: qps=$qps, input_len=$input_len_i, output_len=$output_len_i, num_reqs=$num_reqs"
          benchmark $qps $output_len_i gloo "./3p1d_results_50req_8qps_newRatio" $input_len_i $num_reqs
          kill_gpu_processes
          sleep 5
        # else
          # echo "Skipping iteration $counter"
        # fi
      done
    done
  done
  # kill_gpu_processes


  # python3 visualize_benchmark_results.py

}


main "$@"
