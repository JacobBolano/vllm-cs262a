import time
import os
import json

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

