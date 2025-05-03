# 0. added imports and kills
# adding go 
export PATH=$PATH:/usr/local/go/bin
# adding config file for mooncake
export MOONCAKE_CONFIG_PATH=/home/ubuntu/cs262a/fresh_copy/vllm-cs262a/benchmarks/disagg_benchmarks/mooncake_config.json
# adding 
export PATH=$PATH:/home/ubuntu/cs262a/etcd/bin

# kill all processes on GPU.
kill_gpu_processes() {
  # kill all processes on GPU.
  pgrep pt_main_thread | xargs -r kill -9
  pgrep python3 | xargs -r kill -9
  for port in 8000 8100 8200 8201 50001 8209; do lsof -t -i:$port | xargs -r kill -9; done
  sleep 1
}
kill_gpu_processes


echo "finished step 0"
# 1. Start the etcd server
etcd --listen-client-urls http://172.27.28.94:2379 --advertise-client-urls http://172.27.28.94:2379 &

echo "finished step 1"
# 2. Start the mooncake_master server
LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH \
mooncake_master --port 50001 &

echo "finished step 2"

# 3. Run multiple vllm instances
# kv_producer role
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH \
MOONCAKE_CONFIG_PATH=./mooncake_config.json \
vllm serve neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
		--port 8100 \
		--trust-remote-code \
		--max-model-len 10000 \
		--gpu-memory-utilization 0.8 \
		--kv-transfer-config '{"kv_connector":"MooncakeStoreConnector", "kv_role":"kv_producer", "kv_rank":0, "kv_parallel_size":2}'  &


# kv_consumer role
CUDA_VISIBLE_DEVICES=1 \
LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH \
MOONCAKE_CONFIG_PATH=./mooncake_config.json \
vllm serve neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
		--port 8200 \
		--trust-remote-code \
		--max-model-len 10000 \
		--gpu-memory-utilization 0.8 \
		--kv-transfer-config '{"kv_connector":"MooncakeStoreConnector", "kv_rank":1, "kv_role":"kv_consumer", "kv_rank":1, "kv_parallel_size":2}' &

CUDA_VISIBLE_DEVICES=2 \
LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH \
MOONCAKE_CONFIG_PATH=./mooncake_config.json \
vllm serve neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
		--port 8201 \
		--trust-remote-code \
		--max-model-len 10000 \
		--gpu-memory-utilization 0.8 \
		--kv-transfer-config '{"kv_connector":"MooncakeStoreConnector", "kv_rank":1, "kv_role":"kv_consumer", "kv_rank":1, "kv_parallel_size":2}' &

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}
wait_for_server 8100
wait_for_server 8200
wait_for_server 8201

echo "finished step 3"


# 4. Run round robin proxy server
python3 ../../examples/online_serving/disagg_examples/disagg_proxy_demo.py --model neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 --prefill localhost:8100  --decode localhost:8200 localhost:8201  --port 8000



echo "finished step 4"





#5. Send request 
curl -v -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
  "model": "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
  "prompt": "San Francisco is a",
  "max_tokens": 100}'

echo "finished step 5"

# kill_gpu_processes

echo "finished step 6"