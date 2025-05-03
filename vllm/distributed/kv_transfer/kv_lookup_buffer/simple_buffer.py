# # SPDX-License-Identifier: Apache-2.0
# """
#     Implements a distributed key-value (KV) cache transfer mechanism.

#     Key Features:
#     - Distributed KV cache transmission using PyNccl pipes.
#     - Non-blocking `insert`, blocking `drop_select`.
#     - Use CPU signal pipe to avoid racing condition
#     - Handles buffer size constraints and provide backpressure mechanism to
#       stop the prefill instance when the decode instance is slow.
# """

# # cs262a imports
# import datetime
# import time
# from vllm import buffered_logger
# import os
# import asyncio


# import threading
# from collections import deque
# from typing import Deque, List, Optional, Union

# import torch

# from vllm.distributed.kv_transfer.kv_lookup_buffer.base import (
#     KVLookupBufferBase)
# from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
# from vllm.logger import init_logger

# logger = init_logger(__name__)


# class SimpleBuffer(KVLookupBufferBase):

#     def __init__(self, signal_pipe: KVPipeBase, data_pipe: KVPipeBase,
#                  buffer_size_thresh: float):
#         """
#         signal_pipe: on CPU

#         NOTE: on-device recv will block all threads in the process, making the
#         KV cache producer unable to listen to new request while transmitting
#         KV cache. Luckily CPU recv only blocks the current thread so we use
#         CPU recv to listen to new request.

#         data_pipe: on device (e.g. GPU)
#         """

#         self.buffer: Deque[List[torch.Tensor]] = deque()

#         self.buffer_size = 0
#         self.buffer_size_threshold = buffer_size_thresh
#         self.buffer_cv = threading.Condition()
#         self.signal_pipe = signal_pipe
#         self.data_pipe = data_pipe
#         self.request_handling_thread: Optional[threading.Thread] = []

#         self.normal_signal = torch.tensor([0], device="cpu")
#         self.end_signal = None

#     def _matches(self, tokens_roi_sender: List[torch.Tensor],
#                  tokens_roi_recver: List[torch.Tensor]):

#         # tokens_roi_sender: tokens and roi of the producer (in the buffer)
#         # tokens_roi_recver: tokens and roi of the consumer (query)

#         tokens_sender = tokens_roi_sender[0]
#         tokens_recver = tokens_roi_recver[0]
#         roi_sender = tokens_roi_sender[1]
#         roi_recver = tokens_roi_recver[1]

#         if tokens_recver is None:
#             # consumer sends an empty request
#             # semantics: DROP SELECT * LIMIT 1
#             # so any of the data in the buffer can be drop-selected
#             return True

#         # Assuming that roi is a binary mask on tokens
#         tokens_sender = tokens_sender[roi_sender]
#         tokens_recver = tokens_recver[roi_recver]

#         # simple common prefix matching
#         min_length = min(len(tokens_sender), len(tokens_recver))
#         if torch.allclose(tokens_sender[:min_length],
#                           tokens_recver[:min_length]):
#             return min_length

#         return 0

#     def _send_tensor_and_dec_size(self,
#                                   target_rank,
#                                   tensor: Optional[torch.Tensor],
#                                   is_real_data=True) -> None:

#         assert tensor is not None, "Use self.data_pipe.send(None) instead"

#         if is_real_data:
#             # buffered_logger.log_event("SHRI CLEARING KV BUFFER BEFORE SEND")
#             self.buffer_size -= tensor.element_size() * tensor.numel()
            
#         if tensor.dtype == torch.bool:
#             # buffered_logger.log_event("ROI SHOULD GET HERE")
#             tensor = tensor.float()
#         # timestamp = time.time()  # Unix timestamp (synchronized)
#         # utc_time = datetime.datetime.utcnow().isoformat()  # Readable time
#         # logger.info(f"Shri {os.getpid()} Sending Tensor of type: {tensor.dtype} with dimension: {tensor.size()} at timestamp: {timestamp}, utc_time: {utc_time}")
#         # buffered_logger.flush_log_buffer()

#         self.data_pipe.send_tensor(tensor, target_rank)

#     def _get_element_size(self, data: Optional[Union[List, torch.Tensor]]):

#         if isinstance(data, torch.Tensor):
#             return data.element_size() * data.numel()
#         if not data:
#             # cannot perform `not data` on a tensor
#             # so this check needs to go after the check above
#             return 0

#         raise AssertionError(f"Unknown data type {type(data)}")

#     def _add_to_buffer(self, input_tokens: torch.Tensor, roi: torch.Tensor,
#                        key: torch.Tensor, value: torch.Tensor,
#                        hidden: torch.Tensor):

#         if isinstance(input_tokens, torch.Tensor):
#             input_tokens = input_tokens.clone()
#         if isinstance(roi, torch.Tensor):
#             roi = roi.clone()
#         if isinstance(key, torch.Tensor):
#             key = key.clone()
#         if isinstance(value, torch.Tensor):
#             value = value.clone()
#         if isinstance(hidden, torch.Tensor):
#             hidden = hidden.clone()

#         buffer_item = [input_tokens, roi, key, value, hidden]
#         data_size = sum([self._get_element_size(data) for data in buffer_item])

#         with self.buffer_cv:
#             if self.buffer_size + data_size > self.buffer_size_threshold:
#                 # log outside the while loop to avoid this message being logged
#                 # repeatedly.
#                 logger.debug("KV transfer buffer is full. Handling...")
#                 while self.buffer_size + data_size > self.buffer_size_threshold:
#                     self.buffer_cv.wait()

#             self.buffer_size += data_size
#             self.buffer.append(buffer_item)
#             self.buffer_cv.notify()

#     def _is_end_signal(self, signal):
#         return signal is None

#     def drop_select_handler(self, recv_rank: int = -1):
#         #used by prefill instance

#         if recv_rank == -1:
#             logger.info("Shri failed to set recv rank in drop select handler")

#         try:
#             while True:
#                 signal = self.signal_pipe.recv_tensor(recv_rank)
                
#                 # timestamp = time.time()  # Unix timestamp (synchronized)
#                 # utc_time = datetime.datetime.utcnow().isoformat()  # Readable time
#                 # buffered_logger.log_event(f"ROHAN Prefill {os.getpid()} Begins Drop Select Handler to decode {recv_rank}: {timestamp} (Unix), {utc_time} (UTC)")

#                 if self._is_end_signal(signal):
#                     logger.info("Received end signal!")
#                     break

#                 input_tokens = self.data_pipe.recv_tensor(recv_rank)

#                 roi = self.data_pipe.recv_tensor(recv_rank)

#                 # timestamp = time.time()  # Unix timestamp (synchronized)
#                 # utc_time = datetime.datetime.utcnow().isoformat()  # Readable time
#                 # buffered_logger.log_event(f"ROHAN Prefill Drop Select Handler Received Tensors From Decode {recv_rank}: {timestamp} (Unix), {utc_time} (UTC)")

#                 assert roi is not None, "Please provide the roi when sending "\
#                     "drop-select request"
#                 roi = (roi > 0.5)
#                 tokens_roi_recver = [input_tokens, roi]

#                 def is_buffer_available(
#                     tokens_roi_recver: List[torch.Tensor], ) -> bool:
#                     # perform input tokens and roi matching
#                     # FIXME: this matching is O(n), ideally it should be O(1)
#                     # but this buffer size won't (and shouldn't) be too large so
#                     # the fix is not urgent.
#                     for _ in range(len(self.buffer)):
#                         if self._matches(self.buffer[0],
#                                          tokens_roi_recver) > 0:
#                             return True
#                         # rotate the element we just accessed to the end
#                         self.buffer.rotate(-1)
#                     return False

#                 with self.buffer_cv:
#                     # while not is_buffer_available(tokens_roi_recver):
#                     #     logger.debug(
#                     #         "KV transfer buffer is not available. Waiting...")
#                     #     self.buffer_cv.wait()
#                     # need to clone the tensor
#                     # in case the tensor is freed before sending finishes
#                     if is_buffer_available(tokens_roi_recver):
#                         # buffered_logger.log_event(f"Shri {os.getpid()} buffer is available")
#                         # buffered_logger.flush_log_buffer()
#                         # self._send_tensor_and_dec_size(recv_rank, torch.tensor([1]), False)
#                         matched_item = self.buffer.popleft()
#                         for i, tensor in enumerate(matched_item):
#                             # buffered_logger.log_event(f"Shri Sending Tensor of type: {tensor.dtype} with dimension: {tensor.size()}")
#                             self._send_tensor_and_dec_size(recv_rank, tensor)
#                             # if i == 0:
#                             #     logger.info(f"SEND input tokens {tensor[:3]}")
#                             # elif i == 1:
#                             #     logger.info(f"SEND ROI {tensor.size()}")
#                             # elif i == 2:
#                             #     logger.info(f"SEND key {tensor.size()}")
#                             # elif i == 3:
#                             #     logger.info(f"SEND values {tensor.size()}")
#                             # elif i == 4:
#                             #     logger.info(f"SEND hidden {tensor.size()}")
                        
#                     else:
#                         # buffered_logger.log_event(f"Shri {os.getpid()} Signaled KV cache miss to decode instance")
#                         # buffered_logger.flush_log_buffer()

#                         self.data_pipe.send_tensor(torch.empty(0), recv_rank)
#                         self.data_pipe.send_tensor(torch.empty(0), recv_rank)
#                         self.data_pipe.send_tensor(torch.empty(0), recv_rank)
#                         self.data_pipe.send_tensor(torch.empty(0), recv_rank)
#                         self.data_pipe.send_tensor(torch.empty(0), recv_rank)



#                     # self.buffer_cv.notify()
                    
#                     timestamp = time.time()  # Unix timestamp (synchronized)
#                     utc_time = datetime.datetime.utcnow().isoformat()  # Readable time
#                     logger.info("ROHAN Prefill Drop Select Handler End: %f (Unix), %s (UTC)", timestamp, utc_time)
#                     # buffered_logger.log_event(f"ROHAN Prefill Drop Select Handler End: {timestamp} (Unix), {utc_time} (UTC)")

#         except RuntimeError as e:
#             if 'Connection closed by peer' not in str(e):
#                 raise e

#         logger.debug("Closing drop_select_handler")
#         # buffered_logger.flush_log_buffer()

#     def drop_select(
#             self, input_tokens: Optional[torch.Tensor],
#             roi: Optional[torch.Tensor]) -> List[Optional[torch.Tensor]]:

#         assert len(self.request_handling_thread) == 0, \
#             "drop_select should be called by the KV cache consumer "\
#             "(e.g. the decode vLLM instance)"

#         if isinstance(input_tokens, torch.Tensor):
#             original_input_tokens = input_tokens.clone()
#         if isinstance(roi, torch.Tensor):
#             original_roi = roi.clone().float()

#         # timestamp = time.time()  # Unix timestamp (synchronized)
#         # utc_time = datetime.datetime.utcnow().isoformat()  # Readable time
#         # logger.info("ROHAN Decode Drop Select Start: %f (Unix), %s (UTC)", timestamp, utc_time)
#         # buffered_logger.log_event(f"ROHAN Decode Drop Select Start: {timestamp} (Unix), {utc_time} (UTC)")
#         # buffered_logger.flush_log_buffer()

#         found = False
#         target_ranks = [0, 1]
#         t_idx = 0

#         while not found:

#             self.signal_pipe.send_tensor(self.normal_signal, target_ranks[t_idx])
#             self.data_pipe.send_tensor(original_input_tokens, target_ranks[t_idx])
#             self.data_pipe.send_tensor(original_roi, target_ranks[t_idx])

#             logger.info(f"Drop Select Trying To Receive From: {target_ranks[t_idx]}")

#             input_tokens = self.data_pipe.recv_tensor(target_ranks[t_idx])

#             roi = self.data_pipe.recv_tensor(target_ranks[t_idx])

#             if roi is not None:
#                 # convert from float tensor to bool tensor
#                 # as PyNccl does not support sending bool tensor
#                 roi = (roi > 0.5)

#             key = self.data_pipe.recv_tensor(target_ranks[t_idx])

#             value = self.data_pipe.recv_tensor(target_ranks[t_idx])

#             hidden = self.data_pipe.recv_tensor(target_ranks[t_idx])

#             if input_tokens.numel() != 0:

          
#                 logger.info("Shri KV cache found")
#                 # buffered_logger.flush_log_buffer()
#                 found = True

#                 # logger.info(f"First three elements: {input_tokens[0].item()}, {input_tokens[1].item()}, {input_tokens[2].item()}")
#                 # logger.info(f"CONNECTOR ROI: {roi.size()}")
#                 # logger.info(f"CONNECTOR KEYS: {key.size()}")
#                 # logger.info(f"CONNECTOR VALUES: {value.size()}")
#                 # logger.info(f"CONNECTOR HIDDEN: {hidden.size()}")

#                 # timestamp = time.time()  # Unix timestamp (synchronized)
#                 # utc_time = datetime.datetime.utcnow().isoformat()  # Readable time
#             else:
#                 t_idx += 1
#                 if t_idx >= len(target_ranks):
#                     t_idx = 0
#                 #     buffered_logger.log_event("Shri could not find KV cache in all prefill instances, relooping")

#                 logger.info("Shri KV cache not found - checking another node")
#                 # asyncio.run(asyncio.sleep(0.25))
                
#                 # buffered_logger.flush_log_buffer()
            

#         # timestamp = time.time()  # Unix timestamp (synchronized)
#         # utc_time = datetime.datetime.utcnow().isoformat()  # Readable time
#         # logger.info("ROHAN Decode Drop Select End: %f (Unix), %s (UTC)", timestamp, utc_time)
#         # buffered_logger.log_event(f"ROHAN Decode Drop Select End: {timestamp} (Unix), {utc_time} (UTC)")


#         return [input_tokens, roi, key, value, hidden]

#     def insert(self, input_tokens: torch.Tensor, roi: torch.Tensor,
#                key: torch.Tensor, value: torch.Tensor,
#                hidden: torch.Tensor) -> None:

#         self._add_to_buffer(input_tokens, roi, key, value, hidden)

#         recv_ranks = [2]

#         # when calling the insert, the current process is a sender
#         # need to launch the request handler and start listening to request.
#         if len(self.request_handling_thread) == 0:
#             for i, r in enumerate(recv_ranks):
#                 self.request_handling_thread.append(threading.Thread(
#                     target=self.drop_select_handler,
#                     args=(r,)))
#                 self.request_handling_thread[i].start()

#     def close(self):
#         if hasattr(self, "request_handling_thread"
#                    ) and len(self.request_handling_thread) > 0:
#             for thr in self.request_handling_thread:
#                 thr.join()

#         else:
#             # TODO: have a explicit close signal and have a explicit way to
#             # check if it's requester
#             prod_ranks = [0, 1]
#             cons_ranks = [2]
#             # If consumer then send end signal to produceers
#             if len(self.request_handling_thread) == 0:
#                 for pr in prod_ranks:
#                     self.signal_pipe.send_tensor(self.end_signal, pr)
#             else:
#                 for cr in cons_ranks:
#                     self.signal_pipe.send_tensor(self.end_signal, cr)





# SPDX-License-Identifier: Apache-2.0
"""
    Implements a distributed key-value (KV) cache transfer mechanism.

    Key Features:
    - Distributed KV cache transmission using PyNccl pipes.
    - Non-blocking `insert`, blocking `drop_select`.
    - Use CPU signal pipe to avoid racing condition
    - Handles buffer size constraints and provide backpressure mechanism to
      stop the prefill instance when the decode instance is slow.
"""
# cs262a imports
import datetime
import time
from vllm import buffered_logger

import threading
from collections import deque
from typing import Deque, List, Optional, Union

import torch

from vllm.distributed.kv_transfer.kv_lookup_buffer.base import (
    KVLookupBufferBase)
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from vllm.logger import init_logger

logger = init_logger(__name__)


class SimpleBuffer(KVLookupBufferBase):

    def __init__(self, signal_pipe: KVPipeBase, data_pipe: KVPipeBase,
                 buffer_size_thresh: float):
        """
        signal_pipe: on CPU

        NOTE: on-device recv will block all threads in the process, making the
        KV cache producer unable to listen to new request while transmitting
        KV cache. Luckily CPU recv only blocks the current thread so we use
        CPU recv to listen to new request.

        data_pipe: on device (e.g. GPU)
        """

        self.buffer: Deque[List[torch.Tensor]] = deque()

        self.buffer_size = 0
        self.buffer_size_threshold = buffer_size_thresh
        self.buffer_cv = threading.Condition()
        self.signal_pipe = signal_pipe
        self.data_pipe = data_pipe
        self.request_handling_thread: Optional[threading.Thread] = None

        self.normal_signal = torch.tensor([0], device="cpu")
        self.end_signal = None

    def _matches(self, tokens_roi_sender: List[torch.Tensor],
                 tokens_roi_recver: List[torch.Tensor]):

        # tokens_roi_sender: tokens and roi of the producer (in the buffer)
        # tokens_roi_recver: tokens and roi of the consumer (query)

        tokens_sender = tokens_roi_sender[0]
        tokens_recver = tokens_roi_recver[0]
        roi_sender = tokens_roi_sender[1]
        roi_recver = tokens_roi_recver[1]

        if tokens_recver is None:
            # consumer sends an empty request
            # semantics: DROP SELECT * LIMIT 1
            # so any of the data in the buffer can be drop-selected
            return True

        # Assuming that roi is a binary mask on tokens
        tokens_sender = tokens_sender[roi_sender]
        tokens_recver = tokens_recver[roi_recver]

        # simple common prefix matching
        min_length = min(len(tokens_sender), len(tokens_recver))
        if torch.allclose(tokens_sender[:min_length],
                          tokens_recver[:min_length]):
            return min_length

        return 0

    def _send_tensor_and_dec_size(self,
                                  tensor: Optional[torch.Tensor]) -> None:

        assert tensor is not None, "Use self.data_pipe.send(None) instead"
        self.buffer_size -= tensor.element_size() * tensor.numel()
        if tensor.dtype == torch.bool:
            tensor = tensor.float()
        self.data_pipe.send_tensor(tensor)

    def _get_element_size(self, data: Optional[Union[List, torch.Tensor]]):

        if isinstance(data, torch.Tensor):
            return data.element_size() * data.numel()
        if not data:
            # cannot perform `not data` on a tensor
            # so this check needs to go after the check above
            return 0

        raise AssertionError(f"Unknown data type {type(data)}")

    def _add_to_buffer(self, input_tokens: torch.Tensor, roi: torch.Tensor,
                       key: torch.Tensor, value: torch.Tensor,
                       hidden: torch.Tensor):

        if isinstance(input_tokens, torch.Tensor):
            input_tokens = input_tokens.clone()
        if isinstance(roi, torch.Tensor):
            roi = roi.clone()
        if isinstance(key, torch.Tensor):
            key = key.clone()
        if isinstance(value, torch.Tensor):
            value = value.clone()
        if isinstance(hidden, torch.Tensor):
            hidden = hidden.clone()

        buffer_item = [input_tokens, roi, key, value, hidden]
        data_size = sum([self._get_element_size(data) for data in buffer_item])

        with self.buffer_cv:
            if self.buffer_size + data_size > self.buffer_size_threshold:
                # log outside the while loop to avoid this message being logged
                # repeatedly.
                logger.debug("KV transfer buffer is full. Handling...")
                while self.buffer_size + data_size > self.buffer_size_threshold:
                    self.buffer_cv.wait()

            self.buffer_size += data_size
            self.buffer.append(buffer_item)
            self.buffer_cv.notify()

    def _is_end_signal(self, signal):
        return signal is None

    def drop_select_handler(self):
        # TODO : LOG HERE FOR TIMING PREFILL COMMUNICATION
        try:

            while True:
                signal = self.signal_pipe.recv_tensor()
                timestamp = time.time()  # Unix timestamp (synchronized)
                utc_time = datetime.datetime.utcnow().isoformat()  # Readable time
                buffered_logger.log_event(f"ROHAN Prefill Begins Drop Select Handler: {timestamp} (Unix), {utc_time} (UTC)")
                print("ROHAN Prefill Begins Drop Select Handler")

                if self._is_end_signal(signal):
                    logger.info("Received end signal!")
                    break

                input_tokens = self.data_pipe.recv_tensor()

                roi = self.data_pipe.recv_tensor()

                # timestamp = time.time()  # Unix timestamp (synchronized)
                # utc_time = datetime.datetime.utcnow().isoformat()  # Readable time
                # logger.info("ROHAN Prefill Drop Select Handler Received Tensors From Decode: %f (Unix), %s (UTC)", timestamp, utc_time)

                assert roi is not None, "Please provide the roi when sending "\
                    "drop-select request"
                roi = (roi > 0.5)
                tokens_roi_recver = [input_tokens, roi]

                def is_buffer_available(
                    tokens_roi_recver: List[torch.Tensor], ) -> bool:
                    # perform input tokens and roi matching
                    # FIXME: this matching is O(n), ideally it should be O(1)
                    # but this buffer size won't (and shouldn't) be too large so
                    # the fix is not urgent.
                    for _ in range(len(self.buffer)):
                        if self._matches(self.buffer[0],
                                         tokens_roi_recver) > 0:
                            return True
                        # rotate the element we just accessed to the end
                        self.buffer.rotate(-1)
                    return False

                with self.buffer_cv:
                    while not is_buffer_available(tokens_roi_recver):
                        logger.debug(
                            "KV transfer buffer is not available. Waiting...")
                        self.buffer_cv.wait()
                    # need to clone the tensor
                    # in case the tensor is freed before sending finishes
                    matched_item = self.buffer.popleft()
                    for tensor in matched_item:
                        self._send_tensor_and_dec_size(tensor)
                    self.buffer_cv.notify()

                    timestamp = time.time()  # Unix timestamp (synchronized)
                    utc_time = datetime.datetime.utcnow().isoformat()  # Readable time
                    # logger.info("ROHAN Prefill Drop Select Handler End: %f (Unix), %s (UTC)", timestamp, utc_time)
                    buffered_logger.log_event(f"ROHAN Prefill Drop Select Handler End: {timestamp} (Unix), {utc_time} (UTC)")

        except RuntimeError as e:
            if 'Connection closed by peer' not in str(e):
                raise e

        logger.debug("Closing drop_select_handler")

    def drop_select(
        # TODO : LOG HERE FOR TIMING DECODE COMMUNICATION
            self, input_tokens: Optional[torch.Tensor],
            roi: Optional[torch.Tensor]) -> List[Optional[torch.Tensor]]:

        assert self.request_handling_thread is None, \
            "drop_select should be called by the KV cache consumer "\
            "(e.g. the decode vLLM instance)"

        if isinstance(input_tokens, torch.Tensor):
            input_tokens = input_tokens.clone()
        if isinstance(roi, torch.Tensor):
            roi = roi.clone().float()

        # timestamp = time.time()  # Unix timestamp (synchronized)
        # utc_time = datetime.datetime.utcnow().isoformat()  # Readable time
        # logger.info("ROHAN Decode Drop Select Start: %f (Unix), %s (UTC)", timestamp, utc_time)
        # buffered_logger.log_event(f"ROHAN Decode Drop Select Start: {timestamp} (Unix), {utc_time} (UTC)")

        self.signal_pipe.send_tensor(self.normal_signal)
        self.data_pipe.send_tensor(input_tokens)
        self.data_pipe.send_tensor(roi)

        input_tokens = self.data_pipe.recv_tensor()
        roi = self.data_pipe.recv_tensor()
        if roi is not None:
            # convert from float tensor to bool tensor
            # as PyNccl does not support sending bool tensor
            roi = (roi > 0.5)
        key = self.data_pipe.recv_tensor()
        value = self.data_pipe.recv_tensor()
        hidden = self.data_pipe.recv_tensor()

        # timestamp = time.time()  # Unix timestamp (synchronized)
        # utc_time = datetime.datetime.utcnow().isoformat()  # Readable time
        # logger.info("ROHAN Decode Drop Select End: %f (Unix), %s (UTC)", timestamp, utc_time)
        # buffered_logger.log_event(f"ROHAN Decode Drop Select End: {timestamp} (Unix), {utc_time} (UTC)")

        return [input_tokens, roi, key, value, hidden]

    def insert(self, input_tokens: torch.Tensor, roi: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor) -> None:

        self._add_to_buffer(input_tokens, roi, key, value, hidden)

        # when calling the insert, the current process is a sender
        # need to launch the request handler and start listening to request.
        if self.request_handling_thread is None:
            self.request_handling_thread = threading.Thread(
                target=self.drop_select_handler)
            self.request_handling_thread.start()

    def close(self):

        if hasattr(self, "request_handling_thread"
                   ) and self.request_handling_thread is not None:
            self.request_handling_thread.join()

        else:
            # TODO: have a explicit close signal and have a explicit way to
            # check if it's requester
            self.signal_pipe.send_tensor(self.end_signal)