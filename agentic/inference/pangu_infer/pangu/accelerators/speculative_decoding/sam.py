# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Union
import torch
from typing import List, Tuple, Dict
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
import pickle
import multiprocessing as mp
import queue
import time
import threading
from numba import jit
from vllm.config import VllmConfig, SpeculativeConfig
from pangu_infer.pangu.adaptor_manager.adaptor_manager import register, get_config
import importlib.util


def has_samfast_package():
    return importlib.util.find_spec("samfast") is not None

if has_samfast_package():
    import samfast


class SAM:
    """
    SAM (Suffix Automaton) data structure. SAM is used to store the prefix of the prompt and generated text.
    """
    #... SAM
    @dataclass
    class SAMState:
        next: dict[int, int]
        link: int
        length: int
        min_endpos: int

    def __init__(self, 
                n_predicts: int = 3,
                device: str = "cpu"):
        self.n_predicts = n_predicts
        self.states: List[SAM.SAMState] = [SAM.SAMState(next={}, link=-1, length=0, min_endpos=0)]
        self.input_ids: List[int] = [-1]
        self.last = 0
        self.max_length = 0
        self.device = device
        
        # params needed to be reset for each query
        self.cur_index = 0
        self.cur_length = 0
    
    def reset(self):
        self.states: List[SAM.SAMState] = [SAM.SAMState(next={}, link=-1, length=0, min_endpos=0)]
        self.input_ids: List[int] = [-1]
        self.last = 0
        self.max_length = 0
        self.cur_index = 0
        self.cur_length = 0
    
    def expand_state(self, state: SAMState):
        new_index = len(self.states)
        self.states.append(state)
        return new_index

    def add_state(self, token: int):
        self.max_length += 1
        cur = self.expand_state(
            SAM.SAMState(
                next={}, link=-1, 
                length=self.max_length, 
                min_endpos=self.max_length
            )
        )
        p = self.last
        while p != -1 and token not in self.states[p].next:
            self.states[p].next[token] = cur
            p = self.states[p].link
        if p == -1:
            self.states[cur].link = 0
        else:
            q = self.states[p].next[token]
            if self.states[p].length + 1 == self.states[q].length:
                self.states[cur].link = q
            else:
                clone = self.expand_state(deepcopy(self.states[q]))
                self.states[clone].length = self.states[p].length + 1
                while p != -1 and self.states[p].next[token] == q:
                    self.states[p].next[token] = clone
                    p = self.states[p].link
                self.states[q].link = self.states[cur].link = clone
        self.last = cur
           
    def transfer_state(self, index: int, length: int, token: int):
        while index != 0 and token not in self.states[index].next:
            index = self.states[index].link
            length = self.states[index].length
        if token in self.states[index].next:
            index = self.states[index].next[token]
            length += 1
        else:
            index = length = 0
        return index, length
    
    def transfer_cur_state(self, token: int):
        self.cur_index, self.cur_length = \
            self.transfer_state(self.cur_index, self.cur_length, token)
    
    def add_tokens(self, tokens: Union[List[int], np.ndarray]):
        for token in tokens:
            self.transfer_cur_state(token)
            self.add_state(token)
        self.input_ids.extend(tokens)
    
    def transfer_tokens(self, tokens: Union[List[int], np.ndarray]):
        for token in tokens:
            self.transfer_cur_state(token)

    def lookup(self, token: int):
        index, length = \
            self.transfer_state(self.cur_index, self.cur_length, token)
        return index, length

    def to_anc(self, index: int):
        if index != 0:
            length_to_end = self.max_length - self.states[index].min_endpos
            while self.states[index].link != 0 and self.n_predicts > length_to_end:
                index = self.states[index].link
                length_to_end = self.max_length - self.states[index].min_endpos
        return index

    def gen_draft(self, index: int, start_token: int):
        index = self.to_anc(index)
        endpos = self.states[index].min_endpos
        pred_ids = []
        if endpos != 0:
            pred_ids = self.input_ids[endpos + 1:endpos + self.n_predicts+1]
        return pred_ids

    def gen_dyn_draft(self, index: int, match_length: int, start_token: int):
        n = min(self.n_predicts, 1 + int(match_length * self.alpha))
        endpos = self.states[index].min_endpos
        seq = [start_token] + self.input_ids[endpos + 1:endpos + n]
        seq_position_ids = torch.arange(0, len(seq), dtype=torch.long, device=self.device).unsqueeze(0)
        return seq, {"seq_position_ids": seq_position_ids}

    def propose(self, context_token_ids: np.ndarray,) -> Optional[np.ndarray]:
        self.reset()
        self.add_tokens(context_token_ids[:-1])
        query_token = context_token_ids[-1]
        index_dyn, _ = self.lookup(query_token)
        seq = self.gen_draft(index_dyn, query_token)  # query token是seq中的第一个token
        return np.array(seq)
            

class SAMProposer:
    """
    Stores SAMs corresponding to different request_ids.
    self.n_predicts: int, number of speculative tokens
    self.all_proposers: dict[int, SAM], a dictionary containing request_id and its SAM.
    """
    def __init__(self, speculative_config: SpeculativeConfig, asynchronous_sam=False):
        self.n_predicts = speculative_config.num_speculative_tokens
        self.all_proposers: dict = {} # dict[request_id, SAM]
        self.waiting_queue = queue.Queue()
        self._setup_thread = None
        self.asynchronous_sam = asynchronous_sam
        if asynchronous_sam:
            self._setup_thread = threading.Thread(target=self._setup_sam_loop, daemon=True)
            self._setup_thread.start()

    def propose(self, 
                request_id: int, 
                old_token_ids, 
                new_token_ids, 
                num_sampled_ids):
        if self.asynchronous_sam:
            return self.propose_async(request_id, old_token_ids, new_token_ids, num_sampled_ids)
        else:
            return self.propose_sync(request_id, old_token_ids, new_token_ids, num_sampled_ids)

    def propose_sync(self, 
                request_id: int, 
                old_token_ids, 
                new_token_ids, 
                num_sampled_ids):
        """
        Propose draft tokens with the synchronous version SAM.
        If SAM corresponding the request is abscent, then build SAM synchronously.
        """
        if self.all_proposers.get(request_id, None) is None:
            self.all_proposers[request_id] = SAM(n_predicts=self.n_predicts)
            self.all_proposers[request_id].add_tokens(old_token_ids)
        if num_sampled_ids > 1:
            self.all_proposers[request_id].add_tokens(new_token_ids[:-1])
        query_token = new_token_ids[-1]
        index_dyn, _ = self.all_proposers[request_id].lookup(query_token)
        seq = self.all_proposers[request_id].gen_draft(index_dyn, query_token)
        self.all_proposers[request_id].add_tokens(new_token_ids[-1:])
        return seq
    
    def propose_async(self, 
                request_id: int, 
                old_token_ids, 
                new_token_ids, 
                num_sampled_ids):
        """
        Propose draft tokens with the asynchronous version SAM. 
        If SAM corresponding the request is abscent, then return an empty list, 
        and launch asynchronous SAM construction.
        This function requires the installation of samfast package at 
        Pangu_infer/pangu_infer/pangu/accelerators/speculative_decoding/samfast.
        """
        if request_id not in self.all_proposers:
            self.waiting_queue.put((request_id, np.concatenate((old_token_ids, new_token_ids))))
            return []
        else:
            seq = self.query_sam(request_id, new_token_ids)
            return seq
    
    def _setup_sam_loop(self):
        while True:
            if self.waiting_queue.empty():
                time.sleep(0.1)
            else:
                try:
                    request_id, tokens = self.waiting_queue.get_nowait()
                    sam = self.build_sam(tokens, n_predicts=self.n_predicts)
                    self.all_proposers[request_id] = sam
                except Exception as e:
                    print(f"request id {request_id} SAM built failure due to exception: {e}")

    def load_model(self):
        pass

    def build_sam(self, old_token_ids, n_predicts=3):
        """
        The function is used to build SAM asynchronously.
        """
        sam = samfast.SAM(n_predicts=n_predicts)
        sam.add_tokens(old_token_ids)
        return sam
    
    def query_sam(self, request_id, tokens):
        """
        The function is used to query and append SAM asynchronously.
        """
        num_sampled_ids = len(tokens)
        if num_sampled_ids > 1:
            self.all_proposers[request_id].add_tokens(tokens[:-1])
        query_token = tokens[-1]
        index_dyn, _ = self.all_proposers[request_id].lookup(query_token)
        seq = self.all_proposers[request_id].gen_draft(index_dyn, query_token)
        self.all_proposers[request_id].add_tokens(tokens[-1:])
        return seq

    def cleanup(self, request_id):
        self.all_proposers.pop(request_id, None)

    def shutdown(self):
        self.parent_conn.send({"cmd": "shutdown"})
        self.proc.join(timeout=1)