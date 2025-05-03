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
  for port in 8000 8100 8200 50001 8209; do lsof -t -i:$port | xargs -r kill -9; done
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
  kill_gpu_processes

  # adding go 
  export PATH=$PATH:/usr/local/go/bin
  # adding config file for mooncake
  export MOONCAKE_CONFIG_PATH=/home/ubuntu/cs262a/fresh_copy/vllm-cs262a/benchmarks/disagg_benchmarks/mooncake_config.json
  # adding 
  export PATH=$PATH:/home/ubuntu/cs262a/etcd/bin

  # # 1. Start the etcd server
  etcd --listen-client-urls http://172.27.28.94:2379 --advertise-client-urls http://172.27.28.94:2379 &

  # 2. Start the mooncake_master server
  LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH \
  mooncake_master --port 50001 &


  # disagg prefill for production level serving
  # kv_producer role
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH \
MOONCAKE_CONFIG_PATH=./mooncake_config.json \
vllm serve neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
		--port 8100 \
		--trust-remote-code \
		--max-model-len 10000 \
		--gpu-memory-utilization 0.8 \
		--kv-transfer-config '{"kv_connector":"MooncakeStoreConnector", "kv_role":"kv_producer", "kv_rank":0, "kv_parallel_size":2}'  \
    --enforce-eager &


# kv_consumer role
CUDA_VISIBLE_DEVICES=1 \
LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH \
MOONCAKE_CONFIG_PATH=./mooncake_config.json \
vllm serve neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
		--port 8200 \
		--trust-remote-code \
		--max-model-len 10000 \
		--gpu-memory-utilization 0.8 \
		--kv-transfer-config '{"kv_connector":"MooncakeStoreConnector", "kv_rank":1, "kv_role":"kv_consumer", "kv_parallel_size":2}' \
    --enforce-eager &


  wait_for_server 8100
  wait_for_server 8200
 
  #python3 disagg_prefill_proxy_server.py &
  # 4. Run round robin proxy server (added from https://github.com/vllm-project/vllm/pull/16020)
  python3 ../../examples/online_serving/disagg_examples/disagg_proxy_demo.py --model neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 --prefill localhost:8100  --decode localhost:8200 --port 8000 & 
  sleep 1
}


benchmark() {
  results_folder=$4
  model="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"
  dataset_name="random"
  dataset_path="r"
  num_prompts=10
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
          --port 8000 \
          --save-result \
          --result-dir $results_folder \
          --result-filename "$tag"-qps-"$qps".json \
          --ignore-eos \
          --request-rate "$qps"

  sleep 2
}


main() {
  # adding go 
  export PATH=$PATH:/usr/local/go/bin
  # adding config file for mooncake
  export MOONCAKE_CONFIG_PATH=/home/ubuntu/cs262a/fresh_copy/vllm-cs262a/benchmarks/disagg_benchmarks/mooncake_config.json
  # adding 
  export PATH=$PATH:/home/ubuntu/cs262a/etcd/bin

  (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
  (which jq) || (apt-get -y install jq)
  (which socat) || (apt-get -y install socat)
  (which lsof) || (apt-get -y install lsof)

  pip install quart httpx matplotlib aiohttp datasets

  cd "$(dirname "$0")"

  rm -rf results_disagg
  mkdir results_disagg

  default_output_len=100

  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
  
  # for max_token in 512 2048 4096 16384; do
  # launch_chunked_prefill $max_token
  # rm -rf "./results_${max_token}"
  # mkdir "./results_${max_token}"
  # for qps in 2 8 14; do
  # benchmark $qps $default_output_len chunked_prefill "./results_${max_token}"
  # done
  # done
  # kill_gpu_processes

  launch_disagg_prefill
  for qps in 14; do
  benchmark $qps $default_output_len disagg_prefill "./results_disagg"
  done
  kill_gpu_processes

  # python3 visualize_benchmark_results.py

}


main "$@"
