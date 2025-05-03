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
  for port in 8000 8001 8100 8200 8300 8400; do lsof -t -i:$port | xargs -r kill -9; done
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


# launch_chunked_prefill() {
#   model="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"
#   # disagg prefill
#   local max_batch_tokens=$1
#   CUDA_VISIBLE_DEVICES=6 python3 \
#     -m vllm.entrypoints.openai.api_server \
#     --model $model \
#     --port 8100 \
#     --max-model-len 15000 \
#     --enable-chunked-prefill \
#     --max_num_batched_tokens $max_batch_tokens \
#     --gpu-memory-utilization 0.6 &
#   CUDA_VISIBLE_DEVICES=7 python3 \
#     -m vllm.entrypoints.openai.api_server \
#     --model $model \
#     --port 8200 \
#     --max-model-len 15000 \
#     --enable-chunked-prefill \
#     --max_num_batched_tokens $max_batch_tokens \
#     --gpu-memory-utilization 0.6 &
#   CUDA_VISIBLE_DEVICES=8 python3 \
#     -m vllm.entrypoints.openai.api_server \
#     --model $model \
#     --port 8300 \
#     --max-model-len 15000 \
#     --enable-chunked-prefill \
#     --max_num_batched_tokens $max_batch_tokens \
#     --gpu-memory-utilization 0.6 &
#     CUDA_VISIBLE_DEVICES=9 python3 \
#     -m vllm.entrypoints.openai.api_server \
#     --model $model \
#     --port 8400 \
#     --max-model-len 15000 \
#     --enable-chunked-prefill \
#     --max_num_batched_tokens $max_batch_tokens \
#     --gpu-memory-utilization 0.6 &
#   wait_for_server 8100
#   wait_for_server 8200
#   wait_for_server 8300
#   wait_for_server 8400
#   python3 round_robin_proxy.py &
#   sleep 1
# }


launch_chunked_prefill() {
  local max_batch_tokens=$1; shift
  local ports=("$@")
  local model="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"

  # launch one vLLM server per port, cycling GPUs 6,7,8,...
  for i in "${!ports[@]}"; do
    local port=${ports[i]}
    local gpu=$((6 + i))
    echo "Starting vLLM on GPU $gpu → port $port"
    CUDA_VISIBLE_DEVICES=$gpu python3 -m vllm.entrypoints.openai.api_server \
      --model "$model" \
      --port "$port" \
      --max-model-len 15000 \
      --enable-chunked-prefill \
      --max_num_batched_tokens "$max_batch_tokens" \
      --gpu-memory-utilization 0.6 &
  done

  # wait for all of them to come up
  for port in "${ports[@]}"; do
    wait_for_server "$port"
  done

  # finally launch the proxy
  echo "Starting round-robin proxy on localhost:8001"
  python3 round_robin_proxy.py --ports "${ports[@]}" &
  sleep 1
  echo "All services up!"
}



benchmark() {
  results_folder=$4
  model="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"
  dataset_name="random"
  dataset_path="r"
  num_prompts=5
  qps=$1
  input_len=1000
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
          --port 8001 \
          --save-result \
          --result-dir $results_folder \
          --result-filename "$tag"-qps-"$qps".json \
          --ignore-eos \
          --request-rate "$qps"

  sleep 2
}


main() {
  export VLLM_USE_V1=0
  (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
  (which jq) || (apt-get -y install jq)
  (which socat) || (apt-get -y install socat)
  (which lsof) || (apt-get -y install lsof)

  pip install quart httpx matplotlib aiohttp datasets

  cd "$(dirname "$0")"

  default_output_len=100

  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
  
  for max_token in 2048 ; do  #2048 4096 16384
  launch_chunked_prefill $max_token 8100 8200 8300 8400
  rm -rf "./4cp_results_${max_token}_14000_2"
  mkdir "./4cp_results_${max_token}_14000_2"
  for qps in 14 ; do  # 8 14
  benchmark $qps $default_output_len chunked_prefill "./4cp_results_${max_token}_14000_2"
  done
  kill_gpu_processes
  done

  # launch_disagg_prefill
  # for qps in 14; do
  # benchmark $qps $default_output_len disagg_prefill "./results_disagg"
  # done
  # kill_gpu_processes

  # python3 visualize_benchmark_results.py

}


main "$@"
