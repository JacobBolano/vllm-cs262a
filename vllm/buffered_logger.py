import time
import os
import json

_log_buffer = []

# Adds to memory
def log_event(log_string):
    _log_buffer.append(
        log_string
    )

# Flush to the designated log path:
def flush_log_buffer(log_path="shankar_cp_timing.txt"):
    if not _log_buffer:
        return  # nothing to write

    with open(log_path, "a") as f:  # "a" to append, not overwrite
        for entry in _log_buffer:
            f.write(entry + "\n")

    _log_buffer.clear()

