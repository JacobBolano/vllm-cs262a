import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import ast
import argparse
import os
import sys

# Define all regex patterns
patterns = {
    "prefill_receive": re.compile(r'SHRI Prefill Request (\S+)'),
    "prefill_start": re.compile(r'ROHAN Start of Engine Step Request ID: (\S+)'),
    "prefill_async_start": re.compile(r'ROHAN Prefill Begins Drop Select Handler: (\S+)'),
    "prefill_async_end": re.compile(r'ROHAN Prefill Drop Select Handler End: (\S+)'),
    
    "decode_receive": re.compile(r'SHRI Decode Request (\S+) Number (\d+)'),
    "decode_start": re.compile(r'ROHAN Start of Engine Step Request ID: (\S+) ON PID: \d+ at: \S+ in batch size (\d+)'),
    "decode_async_start": re.compile(r'ROHAN Decode Drop Select Start: (\S+)'),
    "decode_async_end": re.compile(r'ROHAN Decode Drop Select End: (\S+)'),

    "end": re.compile(r'SHRI Finished Request ID (\S+)')

}

# Define expected phases per request type
# required_phases = {
#     'Prefill': ['prefill_receive', 'prefill_start', 'prefill_async_start', 'prefill_async_end', 'end'],
#     'Decode': ['decode_receive', 'decode_start', 'decode_async_start', 'decode_async_end', 'end'],
# }

required_phases = {
    'Prefill': ['prefill_receive', 'prefill_start', 'end'],
    'Decode': ['decode_receive', 'decode_start', 'end'],
}


def extract_phase_timings(log_file_path):
    request_data = defaultdict(dict)
    request_types = {}  # Map Request ID to type (Prefill or Decode)
    request_comm = defaultdict()
    prefill_comm = 0
    decode_comm = 0

    request_number = {} # Map Request (str) to number (int)
    request_batch_sizes = defaultdict(list) # map request to list of batch sizes (for each token)
    request_itls = defaultdict(list) # map request to list of itls (for each token)

    with open(log_file_path, 'r') as f:
        for line in f:
            if "jacob flushing to buffer start. " in line:
                continue
            timestamp_match = re.search(r'(\d+\.\d+)', line)
            if not timestamp_match:
                continue

            timestamp = float(timestamp_match.group(1))

            for phase, pattern in patterns.items():
                match = pattern.search(line)
                if match:
                    if phase == 'prefill_async_start':
                        request_comm["Prefill Async Start Req #" + str(prefill_comm)] = timestamp
                        break
                    elif phase == 'prefill_async_end':
                        request_comm["Prefill Async End Req #" + str(prefill_comm)] = timestamp #- request_comm["Prefill #" + str(prefill_comm)]
                        prefill_comm += 1
                        break
                    elif phase == 'decode_async_start':
                        request_comm["Decode Async Start Req #" + str(decode_comm)] = timestamp
                        break
                    elif phase == 'decode_async_end':
                        request_comm["Decode Async End Req #" + str(decode_comm)] = timestamp #- request_comm["Decode #" + str(decode_comm)]
                        decode_comm += 1
                        break
                    
                    req_id = match.group(1)
                    number = None

                    if req_id in request_types and request_types[req_id] == 'Decode' and phase.startswith('prefill'):
                        continue 
                    if req_id in request_types and request_types[req_id] == 'Prefill' and phase.startswith('decode'):
                        continue 

                    if phase == "decode_receive":
                        number = match.group(2)
                        request_number[req_id] = number

                    batch_size = None
                    if phase == "decode_start":
                        batch_size = match.group(2)
                        request_batch_sizes[request_number[req_id]].append(batch_size)
                    
                    
                    if req_id in request_data and phase in request_data[req_id]:
                        continue  # Skip if already recorded
                    # print(f"Found {phase} for Request ID {req_id} at {timestamp:.6f}")
                    request_data[req_id][phase] = timestamp

                    if phase.startswith('prefill'):
                        request_types[req_id] = 'Prefill'
                    elif phase.startswith('decode'):
                        request_types[req_id] = 'Decode'
                    break

    # Compute durations
    durations = []
    for req_id, times in request_data.items():
        req_type = request_types.get(req_id)
        if not req_type:
            continue
        phases = required_phases[req_type]

        if all(p in times for p in phases):
            if req_type == 'Prefill':
                durations.append({
                    'id': req_id,
                    'type': req_type,
                    'start_to_end': times['end'] - times['prefill_start'],
                    'start': times['prefill_start'],
                    'end': times['end'],
                    # 'async_duration': times['prefill_async_end'] - times['prefill_async_start'],
                    # 'engine_to_async_start': times['prefill_async_start'] - times['prefill_start'],
                })
            elif req_type == 'Decode':
                durations.append({
                    'id': req_id,
                    'type': req_type,
                    'start_to_end': times['end'] - times['decode_start'],
                    'start': times['decode_start'],
                    'end': times['end'],
                    # 'async_duration': times['decode_async_end'] - times['decode_async_start'],
                    # 'engine_to_async_start': times['decode_async_start'] - times['decode_start'],
                    # 'async_end_to_finish': times['end'] - times['decode_async_end']
                })

    # Aggregates ITL for each request
    with open(log_file_path, 'r') as f:
        for line in f:
            if "jacob flushing to buffer start. " in line:
                continue
            itl_match = re.search(r'ITL #(\d+): (.+)', line)
            if itl_match:
                i = itl_match.group(1)
                output_itl_str = itl_match.group(2)

                output_itl_list = ast.literal_eval(output_itl_str)
                request_itls[i] = output_itl_list

    
    return durations, request_comm, request_batch_sizes, request_itls

def naive_visualize_phase_durations(durations):
    prefill = [d for d in durations if d['type'] == 'Prefill']
    decode = [d for d in durations if d['type'] == 'Decode']

    def plot_phase(data, label):
        ids = range(len(data))
        start_to_end = [d['start_to_end'] for d in data]
        # async_dur = [d['async_duration'] for d in data]
        # to_async = [d['engine_to_async_start'] for d in data]
        # from_async = [d['async_end_to_finish'] for d in data]

        plt.figure(figsize=(12, 6))
        plt.plot(ids, start_to_end, label='Total Duration')
        # plt.plot(ids, async_dur, label='Async Duration')
        # plt.plot(ids, to_async, label='Start to Async Start')
        # plt.plot(ids, from_async, label='Async End to Finish')
        plt.title(f'{label} Request Timing Breakdown')
        plt.xlabel('Request Index')
        plt.ylabel('Duration (seconds)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f'TimePlots/{label.lower()}_phases.png', dpi=300)
        plt.show()

    if prefill:
        plot_phase(prefill, 'Prefill')
    if decode:
        plot_phase(decode, 'Decode')

def print_metrics(durations):
    offset = durations[0]['start']
    for d in durations:
        print(f"{d['type']} ID {d['id']}: Total={d['start_to_end']:.6f}s, Async={d['async_duration']:.6f}s, "
               f"ToAsync={d['engine_to_async_start']:.6f}s, FromAsync={d['async_end_to_finish']:.6f}s")

        print(f"{d['type']} ID {d['id']}: Start={d['start']-offset:.6f}s, End={d['end']-offset:.6f}s")

    for req, time in request_comm.items():
        print(f"{req} communication timestamp: {time-offset:.6f}s")

def visualize_itl_batch(request_batch_sizes, request_itls, folder_name):
    if len(request_batch_sizes) != len(request_itls):
        print("Error: the number for requests between request_batch_sizes and request_itls is not the same")
        return

    for i in range(len(request_itls)):

        batch_size = list(map(int, request_batch_sizes[str(i)]))
        inner_token_latency = request_itls[str(i)]

        # Generate x-axis values for batch size (whole numbers)
        num_tokens_batch = np.arange(1, len(batch_size) + 1)

        # Generate x-axis values for ITL (offset by 0.5)
        num_tokens_itl = np.arange(1.5, len(inner_token_latency) + 1.5)

        # Create a figure with a single plot and two y-axes
        fig, ax1 = plt.subplots(figsize=(14, 6))

        # Plot Batch Size on the left y-axis
        ax1.plot(num_tokens_batch, batch_size, 'o-', color='blue', alpha=0.8, linewidth=2, label='Batch Size')
        ax1.set_xlabel('Number of Tokens Generated')
        ax1.set_ylabel('Batch Size', color='blue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Create a second y-axis for the ITL values
        ax2 = ax1.twinx()
        ax2.plot(num_tokens_itl, inner_token_latency, 'o-', color='red', alpha=0.8, linewidth=2, label='Inner Token Latency')
        ax2.set_ylabel('Inner Token Latency (seconds)', color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')

        # Add a title and legend
        plt.title('Batch Size and Inner Token Latency per Token Generated', fontsize=14)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        # Adjust layout and save
        fig.tight_layout()
        plt.savefig(f'{folder_name}/batch_size_itl_offset_plot{i}.png', dpi=300, bbox_inches='tight')
        plt.close()
    print(f"All plots have been generated and saved to {folder_name}")

# Run the timing script
if __name__ == "__main__":
    # --------------------------------------------
    # USAGE:
    # Run this script from the command line like:
    #
    # python timing.py --log_path mylog.txt --itl_batch_folder 100requests_qps14
    #
    # Arguments:
    #   --log_path:     Path to the input log file (must be a .txt file)
    #   --itl_batch_folder:  Directory where itl_batch_folder plots will be saved (must already exist)
    # --------------------------------------------
    parser = argparse.ArgumentParser(description="Process log file and generate visualizations.")
    parser.add_argument("--log_path", type=str, required=True, help="Path to the log file (.txt)")
    parser.add_argument("--itl_batch_folder", type=str, required=True, help="Directory to save plots")

    args = parser.parse_args()

    log_path = args.log_path
    itl_batch_folder = args.itl_batch_folder

    # Validate log file
    if not os.path.isfile(log_path) or not log_path.endswith(".txt"):
        print(f"Error: '{log_path}' does not exist or is not a .txt file.")
        sys.exit(1)

    # Validate output directory
    if not os.path.isdir(itl_batch_folder):
        print(f"Error: Directory '{itl_batch_folder}' does not exist.")
        sys.exit(1)

    # Run the extraction function
    durations, request_comm, request_batch_sizes, request_itls = extract_phase_timings(log_path)
    print(f"\nTotal completed requests with all phase timings: {len(durations)}")

    # Uncomment to print to terminal all durations:
    # print_metrics(durations)
    
    # Uncomment to visualize the phase durations
    # naive_visualize_phase_durations(durations)

    # Uncomment to print ITL-Batch size graphs:
    visualize_itl_batch(request_batch_sizes, request_itls, itl_batch_folder)
