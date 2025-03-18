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
  for port in 8000 8100 8200; do lsof -t -i:$port | xargs -r kill -9; done
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


launch_chunked_prefill() {
  model="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"
  # disagg prefill
  local max_batch_tokens=$1
  CUDA_VISIBLE_DEVICES=0 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8100 \
    --max-model-len 10000 \
    --enable-chunked-prefill \
    --max_num_batched_tokens $max_batch_tokens \
    --gpu-memory-utilization 0.6 &
  CUDA_VISIBLE_DEVICES=1 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8200 \
    --max-model-len 10000 \
    --enable-chunked-prefill \
    --max_num_batched_tokens $max_batch_tokens \
    --gpu-memory-utilization 0.6 &
  wait_for_server 8100
  wait_for_server 8200
  python3 round_robin_proxy.py &
  sleep 1
}


launch_disagg_prefill() {
  model="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8" 
  # disagg prefill
  CUDA_VISIBLE_DEVICES=3 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8100 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.6 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":5e9}' &

  CUDA_VISIBLE_DEVICES=4 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8200 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.6 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":5e9}' &

  wait_for_server 8100
  wait_for_server 8200
  python3 disagg_prefill_proxy_server.py &
  sleep 1
}


benchmark() {
  results_folder=$4
  model="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"
  dataset_name="sharegpt"
  dataset_path="../ShareGPT_V3_unfiltered_cleaned_split.json"
  num_prompts=100
  qps=$1
  tag=$2
  
  python3 ../benchmark_serving.py \
          --backend vllm \
          --model $model \
          --dataset-name $dataset_name \
          --dataset-path $dataset_path \
          --num-prompts $num_prompts \
          --port 8000 \
          --save-result \
          --result-dir $results_folder \
          --result-filename "$tag"-qps-"$qps".json \
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

  rm -rf results_disagg
  mkdir results_disagg

  default_output_len=6

  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
  
  for max_token in 10000 16384 32768; do
  launch_chunked_prefill $max_token
  rm -rf "./results_${max_token}"
  mkdir "./results_${max_token}"
  for qps in 2 4 6 8 10 12 14; do
  benchmark $qps chunked_prefill "./results_${max_token}"
  done
  done
  kill_gpu_processes

  # launch_disagg_prefill
  #for qps in 2 4 6 8 10 12 14; do
  #benchmark $qps $default_output_len disagg_prefill "./results_disagg"
  #done
  #kill_gpu_processes

  #python3 visualize_benchmark_results.py

}


main "$@"
