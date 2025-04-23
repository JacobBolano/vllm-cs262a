import time
import os
import json
import torch
from collections import defaultdict

log_buffer = []

# Adds to memory
def log_event(log_string):
    log_buffer.append(
        log_string
    )

# Flush to the designated log path:
def flush_log_buffer(log_path="rohan_logging.txt"):
    if not log_buffer:
        return  # nothing to write

    with open(log_path, "a") as f:  # "a" to append, not overwrite
        for entry in log_buffer:
            f.write(entry + "\n")

    print(f"Flushed {len(log_buffer)} log entries to {log_path}")
    log_buffer.clear()



# Shankar: added for cuda timing the send for prefill
cuda_timing_events = defaultdict(list)

def add_cuda_event(cuda_event_start, cuda_event_end, pid):

    cuda_timing_events[pid].append((cuda_event_start, cuda_event_end))
    # print("CUDA event start: ", cuda_event_start)
    # print("CUDA event end: ", cuda_event_end)
    print(f"CUDA event list PID: {pid} length: ", len(cuda_timing_events[pid]))
    print(f"CUDA event list PID {pid}: ", cuda_timing_events)

def cuda_flush(log_path="rohan_logging.txt"):
    
    # synchronize device at end
    # gpu_device_0 = torch.device(f"cuda:{0}")
    # gpu_device_1 = torch.device(f"cuda:{1}")

    torch.cuda.synchronize()
    print("SHANKAR cuda timing events: ", cuda_timing_events, "length: ", len(cuda_timing_events))
    with open(log_path, "a") as f:  # "a" to append, not overwrite
        
        for pid, events in cuda_timing_events.items():
            i = 0
            for start_evt, end_evt in events:
                ms = start_evt.elapsed_time(end_evt)
                        
            
                f.write(f"SHANKAR (cuda logging) PID: {pid}, [{i}] " + str(ms) + "\n")
                i += 1