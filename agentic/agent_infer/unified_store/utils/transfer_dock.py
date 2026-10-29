# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import copy
import logging
import threading
import time
from abc import ABC
from operator import itemgetter
from typing import List, Dict, Union, Optional

import ray
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


class TimeoutException(Exception):
    """Custom Timeout Exception"""

    def __init__(self, message="TIMEOUT: Time Sleep Too Long"):
        super().__init__(message)


DEFAULT_TOPIC = "default"


class TransferDock(ABC):
    """
    TransferDock is a data structure class that serves as the base class for GRPOTransferDock,
    providing data storage and retrieval functions.
    """

    def __init__(
            self,
            prompts_num: int,
            n_samples_per_prompt: int,
            experience_columns: Union[List[str], None],
            timeout: Union[int, None],
            timeout_interval: Union[int, None]
    ) -> None:
        """TransferDock initialize.

        Args:
            prompts_num: The number of prompts loaded from the dataset.
            n_samples_per_prompt: The number of responses for a prompt.
            experience_columns: Data columns in TransferDock.
            timeout: The waiting time for over time printing
            timeout_interval: Time interval for timeout printing
        """
        super().__init__()

        self.prompts_num = prompts_num
        self.n_samples_per_prompt = n_samples_per_prompt
        self.max_len = prompts_num * n_samples_per_prompt

        print(f'{prompts_num=} {n_samples_per_prompt=} {self.max_len=}')

        self.experience_columns = (
            experience_columns if experience_columns is not None else []
        )
        self.experience_data = {
            key: [None for _ in range(self.max_len)]
            for key in self.experience_columns
        }
        self.experience_data_status = {
            key: torch.zeros(self.max_len, dtype=torch.int32)
            for key in self.experience_columns
        }

        self.topic2data = {DEFAULT_TOPIC: {
            key: [None for _ in range(self.max_len)]
            for key in self.experience_columns
        }}

        self.topic2data_status = {DEFAULT_TOPIC: {
            key: torch.zeros(self.max_len, dtype=torch.int32)
            for key in self.experience_columns
        }}

        self.timeout = timeout if timeout is not None else 300  # wait over 300s to log
        self.timeout_interval = timeout_interval if timeout_interval is not None else 5  # logger every 5s
        self.logger = logging.getLogger(__name__)

    def _build_topic_data(self, topic: str):
        self.topic2data[topic] = {
            key: [None for _ in range(self.max_len)]
            for key in self.experience_columns
        }

        self.topic2data_status[topic] = {
            key: torch.zeros(self.max_len, dtype=torch.int32)
            for key in self.experience_columns
        }

    def _put(
            self,
            experience_columns: List[str],
            experience: List[List[torch.Tensor]],
            indexes: List[int] = None,
            topic: str = DEFAULT_TOPIC
    ):
        """Put data into specified columns and rows.

        Args:
            experience_columns: Columns to put data in.
                ['prompts', 'attention_mask']
            experience: Data for the corresponding columns.
                [
                    [
                        tensor([1, 1, 1, 1]),
                        tensor([2, 2, 2, 2]),
                        tensor([3, 3, 3, 3]),
                        tensor([4, 4, 4, 4])
                    ],
                    [
                        tensor([1]),
                        tensor([2, 2]),
                        tensor([3, 3, 3]),
                        tensor([4, 4, 4, 4])
                    ]
                ]
            indexes: Rows to put data in.
                [0, 1, 2, 4]

        Returns: None

        """
        # If experience_columns not in TD, raise ValueError
        for experience_column in experience_columns:
            if experience_column not in self.experience_columns:
                raise ValueError(
                    f"put experience ERROR: {experience_column} not in TD experience_column {self.experience_columns}"
                )

        if not indexes:
            raise ValueError(
                "put experience into TD without indexes, indexes must be provided"
            )
        if max(indexes) >= self.max_len:
            raise ValueError(
                f"Put experience index {max(indexes)} exceeds the Transfer Dock range {self.max_len}."
            )

        if not self.topic2data.get(topic) or not self.topic2data_status.get(topic):
            self._build_topic_data(topic)

        experience_data = self.topic2data.get(topic)
        experience_data_status = self.topic2data_status.get(topic)

        for column_idx, single_column in enumerate(experience_columns):
            for i, index in enumerate(indexes):
                experience_data[single_column][index] = experience[column_idx][i]
                experience_data_status[single_column][index] = 1

    def _get(self, experience_columns: List[str], indexes: List[int], topic: str = DEFAULT_TOPIC):
        """Get data based on row and column numbers.

        Args:
            experience_columns: Columns from which to get data.
                ['prompts', 'attention_mask']
            indexes: Rows to get data from.
                [0, 1, 2, 4]

        Returns: Data list.
            [
                [
                    tensor([1, 1, 1, 1]),
                    tensor([2, 2, 2, 2]),
                    tensor([3, 3, 3, 3]),
                    tensor([4, 4, 4, 4])
                ],
                [
                    tensor([1]),
                    tensor([2, 2]),
                    tensor([3, 3, 3]),
                    tensor([4, 4, 4, 4])
                ]
            ]

        """
        if len(indexes) == 0:
            return [[] for _ in range(len(experience_columns))]

        if max(indexes) >= self.max_len:
            raise ValueError(
                f"Get experience index {max(indexes)} exceeds the Transfer Dock range {self.max_len}."
            )

        experience = []

        if not self.topic2data.get(topic) or not self.topic2data_status.get(topic):
            return experience

        for single_column in experience_columns:
            self._wait_for_data(single_column, indexes, topic)
            if len(indexes) == 1:
                experience.append([self.topic2data[topic][single_column][indexes[0]]])
            else:
                experience.append(list(itemgetter(*indexes)(self.topic2data[topic][single_column])))

        return experience

    def _wait_for_data(self, single_column: str, indexes: List[int], topic: str = DEFAULT_TOPIC):
        """Wait for data in which column and row to be ready.

        Args:
            single_column: Column that need to wait for data to be ready.
            indexes: Rows that need to wait for data to be ready.

        Returns: None

        """
        experience_data_status = self.topic2data_status[topic]

        if len(indexes) == 1:
            data_ready = experience_data_status[single_column][indexes] == 1
        else:
            data_ready = sum(itemgetter(*indexes)(experience_data_status[single_column])) == len(indexes)

        start_time = time.time()
        while not data_ready:
            elapsed_time = time.time() - start_time
            if (
                    elapsed_time > self.timeout
                    and elapsed_time % self.timeout_interval < 0.1
            ):
                self.logger.warning(f"TIMEOUT: data_ready has slept {elapsed_time} second")
            # Wait until the data in a single column is ready.
            time.sleep(0.1)
            if len(indexes) == 1:
                data_ready = experience_data_status[single_column][indexes] == 1
            else:
                data_ready = sum(itemgetter(*indexes)(experience_data_status[single_column])) == len(indexes)

    def _clear_experience_data_and_status(self, indexes=None, topic: str = DEFAULT_TOPIC):
        """Clear data and data status in TransferDock.

        Returns: None

        """

        if not self.topic2data.get(topic) or not self.topic2data_status.get(topic):
            return

        if indexes is None:
            self.topic2data[topic] = {
                key: [None for _ in range(self.max_len)]
                for key in self.experience_columns
            }
            self.topic2data_status[topic] = {
                key: torch.zeros(self.max_len, dtype=torch.int32)
                for key in self.experience_columns
            }
        else:
            for key in self.experience_columns:
                self.topic2data_status[topic][key][indexes] = 0
            for key in self.experience_columns:
                for idx in indexes:
                    self.topic2data[topic][key][idx] = None

    def get_experience_data(self, topic: str = DEFAULT_TOPIC):
        """Get all data in TransferDock.

        Returns: Data dict.

        """
        return self.topic2data[topic]

    def get_experience_status(self, topic: str = DEFAULT_TOPIC):
        """Get all data status in TransferDock.

        Returns: Data status dict.

        """
        return self.topic2data_status[topic]

    def get_experience_len(self):
        """Get the maximum length of data in TransferDock.

        Returns: The maximum length of data.

        """
        return self.max_len


@ray.remote(max_concurrency=100, num_cpus=10)
class GRPOTransferDock(TransferDock):
    """
    GRPOTransferDock is based on TransferDock and supports managing data transfer between
    GRPO asynchronous tasks in the Ray cluster.
    """

    def __init__(
            self,
            prompts_num: int,
            n_samples_per_prompt: int,
            metrics=None,
            addition_columns: Union[List[str], None] = None,
            addition_consumers: Union[List[str], None] = None,
            timeout: Union[int, None] = None,
            timeout_interval: Union[int, None] = None,
    ) -> None:
        """GRPOTransferDock initialize.

        Args:
            prompts_num: The number of prompts loaded from the dataset.
            n_samples_per_prompt: The number of responses for a prompt.
            metrics: The metrics stored in TransferDock.
            addition_columns: Additional experience columns in TransferDock.
            addition_consumers: Additional consumers in TransferDock.
            timeout: The waiting time for over time printing.
            timeout_interval: Time interval for timeout printing.
        """
        self.experience_columns = [
            "prompts",
            "prompt_length",
            "responses",
            "response_length",
            "attention_mask",
            "labels",
            "input_ids",
            "actor_rollout",
            "rm_scores",
            "token_level_rewards",
            "old_log_probs",
            "ref_log_probs",
            "advantages",
            "returns",
            "meta",
            "reward_tags",
            "max_prompts_len",
            "max_resp_len",
            "update_finished",
            "iteration",
            "position_ids",
            "loss_mask_idx",
            "vllm_log_probs",
        ]

        self.experience_consumers = [
            "trainer",
            "actor_rollout",
            "get_rollout_lengths",
            "actor_log_prob",
            "ref_log_prob",
            "actor_train",
            "compute_advantage",
            "rule_reward",
            "reward_scores",
            "grpo_metrics",
            "sft_metrics",
            "reward_tags",
            "agent",
            "replace_ref_log_prob",
        ]
        self.batch_seqlen_balance_mapper = {
            "ref_log_prob": ["prompt_length", "response_length"],
            "actor_log_prob": ["prompt_length", "response_length"],
            "reward_scores": ["prompt_length", "response_length"],
            "actor_train": ["prompt_length", "response_length"]
        }
        # Initialize to add additional experience columns
        if addition_columns:
            for column in addition_columns:
                if column not in self.experience_columns:
                    self.experience_columns.append(column)

        # Initialize to add additional experience consumers
        if addition_consumers:
            for consumer in addition_consumers:
                if consumer not in self.experience_consumers:
                    self.experience_consumers.append(consumer)

        super().__init__(
            prompts_num,
            n_samples_per_prompt,
            self.experience_columns,
            timeout,
            timeout_interval,
        )

        self.topic2experience_consumer_status = {
            DEFAULT_TOPIC: {
                key: torch.zeros(self.max_len, dtype=torch.int32)
                for key in self.experience_consumers
            }
        }
        # below is not used later
        # self.consumer_sampling_signal = {key: False for key in self.experience_consumers}

        # for race condition when multiple DP getting data

        self.topic2consumer_sampling_lock = {
            DEFAULT_TOPIC: {
                key: threading.Lock()
                for key in self.experience_consumers
            }
        }
        self.metrics = metrics

    def _build_topic_consumer(self, topic: str):
        self.topic2experience_consumer_status[topic] = {
            key: torch.zeros(self.max_len, dtype=torch.int32)
            for key in self.experience_consumers
        }

        self.topic2consumer_sampling_lock[topic] = {
            key: threading.Lock()
            for key in self.experience_consumers
        }

    def get_metrics(self):
        self.logger.info('========== get_metrics is useful ==========')
        return self.metrics

    def get_transfer_dock_max_len(self):
        return self.max_len

    def get_topics(self):
        return self.topic2experience_consumer_status.keys(), self.topic2consumer_sampling_lock.keys()

    def update_metrics(self, key="", value=None, cumulate=False):
        self.logger.info(f'========= in TD start update metrics ==========')
        self.metrics.update(key, value, cumulate=cumulate)
        self.logger.info(f'========= in TD update metrics finish ==========')

    def get_experience(
            self,
            consumer: str,
            experience_columns: List[str],
            experience_count: int = None,
            indexes: List[int] = None,
            get_n_samples: bool = True,
            reward_tags=None,
            topic: str = DEFAULT_TOPIC,
            use_batch_seqlen_balance: bool = False
    ):
        """Get padded experience data from GRPOTransferDock.

        Args:
            consumer: GRPO task stage to get in.
            experience_columns: Columns from which to get data.
            experience_count: Number of data to get.
            indexes: Rows from which to get data.
            pad_id: Pad token.
            multiple: The multiple of TP to pad.
            get_n_samples: Whether to get n samples at the same time.
            target_seq_len: Target sequence length.
            topic: topic for data
            use_batch_seqlen_balance: Whether to enable batch balance with seq_len.

        Returns: Data dict and row numbers.

        """
        self.logger.info(f'========= TD get experience for {consumer} start ==========')
        if consumer not in self.experience_consumers:
            raise ValueError(
                f"get experience ERROR: {consumer} not in TD experience_consumers {self.experience_consumers}"
            )

        for experience_column in experience_columns:
            if experience_column not in self.experience_columns:
                raise ValueError(
                    f"get experience ERROR: {experience_column} not in TD experience_column {self.experience_columns}"
                )

        if not self.topic2experience_consumer_status.get(topic) or not self.topic2consumer_sampling_lock.get(topic):
            raise ValueError(
                f"get experience ERROR: {topic} not in TD topics"
            )

        if reward_tags is not None:
            experience_columns.append("reward_tags")

        if indexes is None:
            if experience_count > self.max_len:
                raise ValueError(
                    f"TD max_len: {self.max_len} need >= experience_count: {experience_count}"
                )

            if self.max_len % experience_count != 0:
                raise ValueError(
                    f"TD max_len:{self.max_len} need be divisible by experience_count: {experience_count}"
                )

            self.logger.info(f"TD use {get_n_samples=}")
            if get_n_samples:
                if experience_count % self.n_samples_per_prompt != 0:
                    raise ValueError(
                        f"get_n_samples need experience_count:{experience_count} must be divisible by "
                        f"n_samples_per_prompt: {self.n_samples_per_prompt}"
                    )
                indexes = self._sample_ready_index_n_samples(
                    consumer, experience_count, experience_columns, reward_tags=reward_tags, topic=topic,
                    use_batch_seqlen_balance=use_batch_seqlen_balance
                )
            else:
                indexes = self._sample_ready_index(
                    consumer, experience_count, experience_columns, reward_tags=reward_tags, topic=topic,
                    use_batch_seqlen_balance=use_batch_seqlen_balance
                )

            if not indexes:
                self.logger.info(f'======= {consumer} get n samples, indexes None =======')
                return None, None
            else:
                self.logger.debug(f'======= {consumer} get n samples, indexes {indexes} =======')
            experience = self._get(experience_columns, indexes, topic=topic)
        else:
            self.topic2experience_consumer_status[topic][consumer][indexes] = 1
            experience = self._get(experience_columns, indexes, topic=topic)

        experience_batch = {}
        for i, experience_column in enumerate(experience_columns):
            experience_batch[experience_column] = experience[i]
        experience_batch_ref = ray.put(experience_batch)
        return experience_batch_ref, indexes

    def put_experience(
            self,
            data_dict: Dict[str, Union[Tensor, List[Tensor]]],
            indexes: List[int] = None,
            topic: str = DEFAULT_TOPIC,
    ):
        """Put data into specified columns and rows.

        Args:
            data_dict: Data dict to put in GRPOTransferDock.
            indexes: Rows to put data in.
            topic: topic for data

        Returns: None

        """
        self.logger.debug('========== TD tries to put data ==========')
        if not indexes:
            raise ValueError(
                "put experience into TD without indexes, indexes must be provided"
            )

        if not self.topic2experience_consumer_status.get(topic) and not self.topic2consumer_sampling_lock.get(topic):
            self._build_topic_consumer(topic)

        experience_columns, experience = trans_input_to_experience(data_dict)
        self._put(experience_columns, experience, indexes, topic)
        self.logger.debug(f"{topic} = {self.topic2data_status[topic]=}")

    def put_prompts_experience(
            self, batch: Dict[str, Tensor], dataset_additional_keys: List[str] = None, pad_id=None,
            topic: str = DEFAULT_TOPIC
    ):
        """Put data into specified columns and rows.

        Args:
            batch: Batch datas from original dataloader.
            dataset_additional_keys: The additional experience types from the dataset.
            if the input prompts contain padding, we need the pad_id too

        Returns: None

        """
        # should we use "input_ids" or "prompts" below?
        prompts = batch["prompts"]
        prompt_length = get_no_pad_length(prompts, self.n_samples_per_prompt, pad_id)

        prompts_data = prompts
        prompts = []
        for prompt in prompts_data:
            for _ in range(self.n_samples_per_prompt):
                prompts.append(copy.deepcopy(prompt))

        add_vals = {}
        for add_keys in dataset_additional_keys:
            if add_keys in batch.keys():
                values = []
                for value in batch[add_keys]:
                    for _ in range(self.n_samples_per_prompt):
                        values.append(value)
                add_vals[add_keys] = values

        indexes = [i for i in range(len(prompt_length))]
        data_dict = dict(
            {"prompt_length": prompt_length, "prompts": prompts}, **add_vals
        )
        experience_columns, experience = trans_input_to_experience(data_dict)

        self._put(experience_columns, experience, indexes, topic)
        self.logger.debug(f"{self.topic2data_status[DEFAULT_TOPIC]=}")

    def _sample_ready_index(
            self,
            consumer: str,
            experience_count: int,
            experience_columns: List[str],
            target_seq_len: int = None,
            reward_tags=None,
            topic: str = DEFAULT_TOPIC,
            use_batch_seqlen_balance: bool = False
    ) -> Optional[List[int]]:
        """Wait for random sampling rows to be ready.

        Args:
            consumer: GRPO task stage to sample in.
            experience_count: Number for rows to sample.
            experience_columns: Columns from which to sample.

        Returns: Sampled row numbers.

        """
        # reward_tags is 'reward' experience['reward_tags'] is 0
        # reward_tags is 'neural_checker' experience['reward_tags'] is 1
        with self.topic2consumer_sampling_lock[topic][consumer]:
            not_consumed_indexes = self.topic2experience_consumer_status[topic][consumer] == 0
            if reward_tags is None:
                # non-reward workers
                data_ready_indexes = torch.all(
                    torch.stack(
                        [self.topic2data_status[topic][single_column] == 1 for single_column in experience_columns]
                    ), dim=0,
                )
            else:
                # reward workers
                other_columns = [col for col in experience_columns if col != 'reward_tags']
                other_ones_mask = torch.all(
                    torch.stack([self.topic2data_status[topic][col] == 1 for col in other_columns]),
                    dim=0,
                )

                reward_tags_mask = self.experience_data_status['reward_tags'] == reward_tags

                data_ready_indexes = reward_tags_mask & other_ones_mask

            usable_indexes = (not_consumed_indexes & data_ready_indexes).nonzero(as_tuple=True)[0]

            if len(usable_indexes) < experience_count:
                if reward_tags is None:
                    return None
                else:
                    experience_count = len(usable_indexes)
            if experience_count > 0:
                if consumer in self.batch_seqlen_balance_mapper and use_batch_seqlen_balance and len(
                        usable_indexes) % experience_count == 0:
                    sampled_indexes = self.batch_seqlen_balance_sampler(
                        consumer, usable_indexes, experience_count, get_n_samples=False
                    )
                    if not sampled_indexes:
                        return None
                else:
                    ### attention the balancing sampler function is NOT ready to use yet ###
                    sampled_indexes = self.batch_balancing_sampler(
                        experience_columns, usable_indexes, experience_count, target_seq_len
                    )
                self.topic2experience_consumer_status[topic][consumer][sampled_indexes] = 1
            else:
                sampled_indexes = None

        return sampled_indexes

    def _sample_ready_index_n_samples(
            self,
            consumer: str,
            experience_count: int,
            experience_columns: List[str],
            target_seq_len: int = None,
            reward_tags=None,
            topic: str = DEFAULT_TOPIC,
            use_batch_seqlen_balance: bool = False
    ) -> Optional[List[int]]:
        """Randomly select a specified number of prepared experiences from TransferDock at multiples of n_sample.

        Args:
            consumer: GRPO task stage to sample in.
            experience_count: Number for rows to sample.
            experience_columns: Columns from which to sample.
            target_seq_len: Sample according with seq_len and target_seq_len.

        Returns: Sampled row numbers.

        """
        self.logger.debug(f'==== for {consumer} enter nsamples ====')
        experience_count_n_samples = experience_count // self.n_samples_per_prompt
        with self.topic2consumer_sampling_lock[topic][consumer]:
            experience_consumer_status_n_samples = (
                    1 - torch.all(
                torch.tensor(
                    torch.reshape(
                        self.topic2experience_consumer_status[topic][consumer],
                        (self.prompts_num, self.n_samples_per_prompt),
                    ) == 0
                ), dim=1,
            ).int()
            )
            self.logger.debug(
                f'==== for {consumer} experience_consumer_status_n_samples {experience_consumer_status_n_samples} ====')

            not_consumed_indexes = experience_consumer_status_n_samples == 0
            self.logger.debug(f'==== for {consumer} experience_data_status {self.topic2data_status[topic]} ====')

            if reward_tags is None:
                experience_data_status_n_samples = {}
                for key, value in self.topic2data_status[topic].items():
                    experience_data_status_n_samples[key] = torch.all(
                        torch.tensor(
                            torch.reshape(value, (self.prompts_num, self.n_samples_per_prompt)) == 1
                        ), dim=1,
                    ).int()
                    # self.logger.debug(f'==== for {consumer} experience_data_status_n_samples key {key}: {experience_data_status_n_samples[key]} ====')
            else:
                experience_data_status_n_samples = {}
                for key, value in self.topic2data_status[topic].items():
                    if (key == 'reward_tags'):
                        experience_data_status_n_samples[key] = torch.all(
                            torch.tensor(
                                torch.reshape(value, (self.prompts_num, self.n_samples_per_prompt)) == reward_tags
                            ), dim=1,
                        ).int()
                    else:
                        experience_data_status_n_samples[key] = torch.all(
                            torch.tensor(
                                torch.reshape(value, (self.prompts_num, self.n_samples_per_prompt)) == 1
                            ), dim=1,
                        ).int()

            data_ready_indexes = torch.all(
                torch.stack(
                    [experience_data_status_n_samples.get(single_column) == 1 for single_column in experience_columns]),
                dim=0,
            )
            self.logger.debug(f'==== for {consumer} data_ready_indexes: {data_ready_indexes} ====')

            usable_indexes = (not_consumed_indexes & data_ready_indexes).nonzero(as_tuple=True)[0]
            self.logger.debug(f'==== for {consumer} usable_indexes: {usable_indexes} ====')

            if len(usable_indexes) < experience_count_n_samples:
                if reward_tags is None:
                    self.logger.debug(f'==== for {consumer} return None ====')
                    return None
                else:
                    ####### important notice, if we have reward models and reward tags, for each tag, the sample number is smaller than experience_count ######
                    experience_count_n_samples = len(usable_indexes)

            if consumer in self.batch_seqlen_balance_mapper and use_batch_seqlen_balance and len(
                    usable_indexes) % experience_count_n_samples == 0:
                sampled_indexes_n_sample = self.batch_seqlen_balance_sampler(
                    consumer, usable_indexes, experience_count_n_samples, get_n_samples=True
                )
                if not sampled_indexes_n_sample:
                    return None
            else:
                sampled_indexes_n_sample = self.batch_balancing_sampler(
                    experience_columns,
                    usable_indexes,
                    experience_count_n_samples,
                    target_seq_len,
                )
            self.logger.debug(f'==== for {consumer} sampled_indexes_n_sample: {sampled_indexes_n_sample} ====')
            sampled_indexes = []
            for n_sample_index in sampled_indexes_n_sample:
                index_list = []
                for index in range(
                        n_sample_index * self.n_samples_per_prompt,
                        (n_sample_index + 1) * self.n_samples_per_prompt
                ):
                    index_list.append(index)

                sampled_indexes += index_list
                self.topic2experience_consumer_status[topic][consumer][sampled_indexes] = 1
                self.logger.debug(f'==== for {consumer} sampled_indexes: {sampled_indexes} ====')
        self.logger.debug(f'=========== finish sample n responses for {consumer} ===========')
        return sampled_indexes

    def all_consumed(self, consumer: str, topic: str = DEFAULT_TOPIC):
        """If consumer has consumed all data in GRPOTransferDock.

        Args:
            consumer: GRPO task stage to consume in.

        Returns: True or False.

        """
        return self.topic2experience_consumer_status[topic][consumer].sum() == self.max_len

    def all_updated(self, single_column: str, topic: str = DEFAULT_TOPIC):
        """If all data in GRPOTransferDock has been through actor update.

        Returns: True or False. 

        """
        return self.topic2data_status[topic][single_column].sum() == self.max_len

    def clear(self, topic: str = DEFAULT_TOPIC):
        """Reset consumer status.Clear data and data status in GRPOTransferDock.

        Returns: None

        """
        self.topic2experience_consumer_status[topic] = {
            key: torch.zeros(self.max_len, dtype=torch.int32)
            for key in self.experience_consumers
        }
        self.topic2consumer_sampling_lock[topic] = {
            key: threading.Lock()
            for key in self.experience_consumers
        }
        self.metrics.reset()
        self._clear_experience_data_and_status(topic=topic)
        # torch.cuda.empty_cache()

    def get_consumer_status(self, topic: str = DEFAULT_TOPIC):
        """Get consumer status.

        Returns: Consumer status dict.

        """
        return self.topic2experience_consumer_status[topic]

    def batch_seqlen_balance_sampler(
            self, consumer, usable_indexes, experience_count, get_n_samples=False
    ):
        from agent_infer.unified_store.utils.seqlen_balancing import get_seqlen_balanced_partitions
        if len(usable_indexes) == experience_count:
            sampled_indexes = [int(usable_indexes[i]) for i in range(experience_count)]
            return sampled_indexes
        seq_len_columns = self.batch_seqlen_balance_mapper.get(consumer)
        if get_n_samples:
            seq_len_list = [
                sum([self.experience_data[key][idx * self.n_samples_per_prompt + addition].item()
                     for addition in range(self.n_samples_per_prompt) for key in seq_len_columns])
                for idx in usable_indexes
            ]
        else:
            seq_len_list = [
                sum([self.experience_data[key][idx].item() for key in seq_len_columns])
                for idx in usable_indexes
            ]
        k_partitions = len(seq_len_list) // experience_count
        sampled_indexes_idx = get_seqlen_balanced_partitions(seq_len_list, k_partitions, equal_size=True)
        if len(sampled_indexes_idx) > 0:
            sampled_indexes = [int(usable_indexes[i]) for i in sampled_indexes_idx[0]]
        else:
            sampled_indexes = None
        return sampled_indexes

    def batch_balancing_sampler(
            self, experience_columns, usable_indexes, experience_count, target_seq_len=None, topic: str = DEFAULT_TOPIC
    ):
        if target_seq_len is None:
            weights = torch.ones(len(usable_indexes))
        else:
            seq_len = torch.tensor(
                [
                    sum([self.topic2data[topic][key][idx].numel() for key in experience_columns])
                    for idx in usable_indexes
                ]
            )
            weights = torch.sigmoid(1 / (torch.abs(seq_len - target_seq_len) + 0.001), dim=0)

        sampled_indexes_idx = torch.multinomial(weights, experience_count, replacement=False).tolist()
        sampled_indexes = [int(usable_indexes[i]) for i in sampled_indexes_idx]

        return sampled_indexes


def pad_experience(
        experience_batch: Dict[str, List[Tensor]],
        pad_id: int,
        multiple: int = 1,
):
    """ Pad dict data.

    Args:
        experience_batch: Dict
            {
                'prompts': [ tensor([1, 1, 1, 1]), 
                             tensor([2, 2, 2, 2]),
                             tensor([3, 3, 3, 3]), 
                             tensor([4, 4, 4, 4])], 
                'attention_mask': [ tensor([1]), 
                                    tensor([2, 2]), 
                                    tensor([3, 3, 3]), 
                                    tensor([4, 4, 4, 4])], 
            }
        pad_id: Pad token.
            0.0
        multiple: The multiple of TP to pad.
            1

    Returns: Merged and padded data dict.
        {
            "prompts": tensor(
                [[1, 1, 1, 1],
                [2, 2, 2, 2],
                [3, 3, 3, 3],
                [4, 4, 4, 4]]),
            "attention_mask": tensor(
                [[1, 0, 0, 0],
                [2, 2, 0, 0],
                [3, 3, 3, 0],
                [4, 4, 4, 4]]),
        }

    """

    def pad_multiples(
            data_list: List[Tensor],
            pad_id: Union[float, int],
            multiple: int = 1
    ) -> Tensor:
        """Pad method for data list.

        Args:
            data_list: Data list.
            pad_id: Pad token.
            multiple: The multiple of TP to pad.

        Returns: Padded tensor.

        """
        padded = pad_sequence(data_list, batch_first=True, padding_value=pad_id)
        max_len = padded.size(1)
        target_len = ((max_len + multiple - 1) // multiple) * multiple
        padded = F.pad(padded, (0, target_len - max_len), value=pad_id)
        return padded

    batch = {}
    if not experience_batch:
        raise ValueError("ERROR: when pad, get an empty experience_batch")
    else:
        for experience_column, experience in experience_batch.items():
            if experience_column in ["prompt_length", "response_length", "rm_scores", "advantages", "max_resp_len",
                                     "max_prompts_len"]:
                padded = torch.cat(experience).reshape(-1, 1)
            elif experience[0].is_floating_point():
                padded = pad_multiples(experience, pad_id=0.0, multiple=multiple)
            else:
                padded = pad_multiples(experience, pad_id=pad_id, multiple=multiple)

            batch[experience_column] = padded

    return batch


def trans_input_to_experience(experience_dict: Dict[str, Union[Tensor, List[Tensor]]]):
    """Split data dict into columns and data list.

    Args:
        experience_dict: Data dict.
            {
                "prompts": tensor(
                    [[1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [3, 3, 3, 3],
                    [4, 4, 4, 4]]),
                "attention_mask": [
                    tensor([1]),
                    tensor([2, 2]),
                    tensor([3, 3, 3]),
                    tensor([4, 4, 4, 4])]
            }

    Returns: Columns and data list.
        ['prompts', 'attention_mask']
        [
            [
                tensor([1, 1, 1, 1]),
                tensor([2, 2, 2, 2]),
                tensor([3, 3, 3, 3]),
                tensor([4, 4, 4, 4])
            ],
            [
                tensor([1)],
                tensor([2, 2]),
                tensor([3, 3, 3]),
                tensor([4, 4, 4, 4])
            ]
        ]

    """
    experience_columns = []
    experience_list = []
    for key, value in experience_dict.items():
        if value is not None:
            experience_columns.append(key)
            if isinstance(value, Tensor):
                if value.dtype == torch.int64:
                    value = value.to(torch.int32)
                value = list(torch.unbind(value, dim=0))
            elif isinstance(value, List):
                value = [val.to(torch.int32) if val.dtype == torch.int64 else val for val in value]
            else:
                raise ValueError(f"value type {type(value)} not supported")
            experience_list.append(value)

    return experience_columns, experience_list


def pack_experience_columns(experience_consumer_stage, experience_dict, experience_count):
    """ 
    Compress experiences by packing tensors into ONE.
    from experience_dict
        {
            'prompts': [ tensor([1, 1, 1]), 
                            tensor([2, 2, 2, 2]),
                            tensor([3, 3, 3]), 
                            tensor([4, 4, 4, 4])], 
            'attention_mask': [ tensor([1]), 
                                tensor([2, 2]), 
                                tensor([3, 3, 3]), 
                                tensor([4, 4, 4, 4])], 
        }
    To batch_data
        {
            'prompts': tensor([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4]),
            'attention_mask': tensor([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
        }
        batch_data_length
        {
            'prompts': tensor([3, 4, 3, 4]), 
            'attention_mask': tensor([1, 2, 3, 4])
        }
    """
    if not experience_dict:
        raise ValueError(f"ERROR: when pack, get an empty experience_dict")

    batch_data = {}
    batch_data_length = {}

    for key, value in experience_dict.items():
        if len(value) != experience_count:
            raise ValueError(f"ERROR: when pack, experience '{key}' number does not match experience_count")
        packed_experience = []
        data_length = []
        for i in range(experience_count):
            packed_experience.extend(value[i].tolist())
            data_length.append(len(value[i]))

        batch_data[key] = torch.tensor(packed_experience, dtype=value[0].dtype)
        batch_data_length[key] = torch.tensor(data_length, dtype=torch.int32)

    return batch_data, batch_data_length


def unpack_experience(batch_data, batch_data_length) -> Dict[str, List[torch.Tensor]]:
    """
    把 pack_experience_columns 的输出还原成“变长序列”，不做 padding。
    返回:
      dict[str, list[1D Tensor]]  —— 每条样本一段，长度为真实长度
    """
    if not batch_data:
        raise ValueError("ERROR: empty batch_data")
    if set(batch_data.keys()) != set(batch_data_length.keys()):
        raise ValueError("ERROR: keys of batch_data and batch_data_length do not match")

    out: Dict[str, List[torch.Tensor]] = {}

    for key, lengths in batch_data_length.items():
        data = batch_data[key]
        lengths = lengths
        out[key] = list(torch.split(data, lengths.tolist()))

    return out


def unpack_pad_experience(batch_data, batch_data_length, pad_id, multiple):
    """
    1. restore the received experience dict
    2. pad the tensor (consider the requirement of multiple)
    from batch_data
        {
            'prompts': tensor([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4]),
            'attention_mask': tensor([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
        }
        batch_data_length
        {
            'prompts': tensor([3, 4, 3, 4]),
            'attention_mask': tensor([1, 2, 3, 4])
        }
    To padded_batch_data (multiple=8) 
    将 max_row_len ​​向上取整到最近的 multiple 的整数倍
        {
            "prompts": tensor(
                [[1, 1, 1, -1, -1, -1, -1, -1],
                [2, 2, 2, 2, -1, -1, -1, -1],
                [3, 3, 3, -1, -1, -1, -1, -1],
                [4, 4, 4, 4, -1, -1, -1, -1]]),
            "attention_mask": tensor(
                [[1, -1, -1, -1, -1, -1, -1, -1],
                [2, 2, -1, -1, -1, -1, -1, -1],
                [3, 3, 3, -1, -1, -1, -1, -1],
                [4, 4, 4, 4, -1, -1, -1, -1]]),
        }
    """
    if not batch_data:
        raise ValueError(f"ERROR: empty batch_data")

    if set(batch_data.keys()) != set(batch_data_length.keys()):
        raise ValueError(f"ERROR: when unpack, keys from batch_data and batch_data_length dictionaries do not match")

    data_device = batch_data[list(batch_data.keys())[0]].device

    padded_batch_data = {}
    for key, length_list in batch_data_length.items():
        if key in ['prompt_length', 'response_length', "rm_scores", "advantages", "max_resp_len", "max_prompts_len"]:
            padded_batch_data[key] = batch_data[key].view(-1, 1)
            continue
        data = batch_data[key]
        data_dtype = batch_data[key].dtype

        lengths = length_list.to(data_device)

        # 计算最大长度
        max_row_len = torch.max(lengths).item()
        if multiple > 1:
            max_row_len = ((max_row_len + multiple - 1) // multiple) * multiple

        # 预分配张量
        if data[0].is_floating_point():
            padded_tensor = torch.full((len(lengths), max_row_len), 0.0,
                                       dtype=data_dtype, device=data_device)
        else:
            padded_tensor = torch.full((len(lengths), max_row_len), pad_id,
                                       dtype=data_dtype, device=data_device)

        # 向量化填充
        cum_length = torch.cat([torch.tensor([0], device=data_device
                                             ), torch.cumsum(lengths, 0)])

        for i, _ in enumerate(lengths):
            seq_len = lengths[i]
            padded_tensor[i, :seq_len] = data[cum_length[i]:cum_length[i + 1]]
        padded_batch_data[key] = padded_tensor

    return padded_batch_data


def pack_and_pad_experience(experience_batch: dict, multiple, pad_id):
    """
    将形如：
        {
            'prompts': [tensor([1,1,1]), tensor([2,2,2,2]), ...],
            ...
        }
      的结构，转换为：
        {
            'prompts': tensor([[1,1,1,-1,...], [...], ...])
        }
      同时自动对齐到 multiple 的整数倍长度。
    """
    if not experience_batch:
        raise ValueError("ERROR: empty experience_batch")

    padded_batch_data = {}

    for key, tensor_list in experience_batch.items():
        # 确保所有元素都是 tensor
        if not isinstance(tensor_list[0], torch.Tensor):
            tensor_list = [torch.tensor(t) for t in tensor_list]

        # 获取 device & dtype
        device = tensor_list[0].device
        dtype = tensor_list[0].dtype

        # 计算 max_row_len，并向上取整到 multiple 的整数倍
        lengths = torch.tensor([len(t) for t in tensor_list], device=device)
        max_row_len = torch.max(lengths).item()
        if multiple > 1:
            max_row_len = ((max_row_len + multiple - 1) // multiple) * multiple

        # 创建填充后的张量
        batch_size = len(tensor_list)
        if tensor_list[0].is_floating_point():
            padded_tensor = torch.full((batch_size, max_row_len), 0.0, dtype=dtype, device=device)
        else:
            padded_tensor = torch.full((batch_size, max_row_len), pad_id, dtype=dtype, device=device)

        # 填充数据
        for i, tensor in enumerate(tensor_list):
            padded_tensor[i, :len(tensor)] = tensor

        padded_batch_data[key] = padded_tensor

    return padded_batch_data


def remove_padding(pad_id, padded_tensor):
    mask = padded_tensor != pad_id
    filtered_sequences = [padded_tensor[i][mask[i]].tolist() for i in range(padded_tensor.size(0))]

    return filtered_sequences


# TODO for prompts and attn_mask we need to extract the real length and tokens
def get_no_pad_length(prompts, n_samples_per_prompt, pad_id=None):
    if not isinstance(prompts, torch.Tensor):
        raise TypeError(f'prompts should be a torch.Tensor, got {type(prompts)} instead.')

    if pad_id is not None:
        prompts_no_pad_length = []
        batch_size = prompts.size(0)
        for i in range(batch_size):
            prompt_token_ids = prompts[i]
            nonzeros = torch.nonzero(prompt_token_ids == pad_id, as_tuple=False)
            if len(nonzeros) != 0:
                first_pad_index = nonzeros[0][0]
            else:
                first_pad_index = len(prompt_token_ids)
            token_ids = prompt_token_ids[:first_pad_index].cpu().numpy().tolist()

            for _ in range(n_samples_per_prompt):
                prompts_no_pad_length.append(torch.tensor([max(1, len(token_ids))]))
        return prompts_no_pad_length
    else:
        prompt_length = []
        for prompt in prompts:
            for _ in range(n_samples_per_prompt):
                prompt_length.append(torch.tensor([len(prompt)]))
        return prompt_length
