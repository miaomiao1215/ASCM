from __future__ import print_function, absolute_import
import os
import numpy as np

import torch
from torch.utils.data import Dataset
import random
import copy
from utils import InputExample, InputFeatures, DictDataset, get_unlabel_ratio, get_unlabel_ratio_fc_center, \
    get_hard_unlabel_ratio, get_unlabel_ratio_train_center, \
    get_unlabel_low_high_category_v8, get_unlabel_low_high_category_v9, \
    get_unlabel_pet, get_unlabel_low_high_category, get_unlabel_pet_add_low_v2, get_unlabel_pet_add_low, \
    get_unlabel_final_avg
from transformers import GPT2Tokenizer
from typing import Tuple, List, Union, Dict

def get_mask_positions(input_ids: List[int], mask_id: int) -> List[int]:
    try:
        # label_idx = input_ids.index(mask_id)     
        # labels = [-1] * len(input_ids)
        # labels[label_idx] = 1
        labels = torch.where(torch.tensor(input_ids)==mask_id, 1, -1)
    except:
        labels = torch.tensor([-1] * len(input_ids))
    return labels

def seq_length(parts: List[Tuple[str, bool]], only_shortenable: bool = False):
    return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

def remove_last(parts: List[Tuple[str, bool]]):
    last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
    parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

def truncate(parts_a: List[Tuple[str, bool]], parts_b: List[Tuple[str, bool]], max_length: int, tokenizer):
    """Truncate two sequences of text to a predefined total maximum length"""
    total_len = seq_length(parts_a) + seq_length(parts_b)
    total_len += tokenizer.num_special_tokens_to_add(bool(parts_b))
    num_tokens_to_remove = total_len - max_length

    if num_tokens_to_remove <= 0:
        return parts_a, parts_b

    for _ in range(num_tokens_to_remove):
        if seq_length(parts_a, only_shortenable=True) > seq_length(parts_b, only_shortenable=True):
            remove_last(parts_a)
        else:
            remove_last(parts_b)
            
    return parts_a, parts_b

def encode(parts_a, parts_b, tokenizer, max_seq_length) \
            -> Tuple[List[int], List[int]]:
        """
        Encode an input example using this pattern-verbalizer pair.

        :param example: the input example to encode
        :param priming: whether to use this example for priming
        :param labeled: if ``priming=True``, whether the label should be appended to this example
        :return: A tuple, consisting of a list of input ids and a list of token type ids
        """

        # kwargs = {}
        kwargs = {'add_prefix_space': True} if isinstance(tokenizer, GPT2Tokenizer) else {}
        parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
        parts_a = [(tokenizer.encode(x, add_special_tokens=False, **kwargs), s) for x, s in parts_a if x]

        if parts_b:
            parts_b = [x if isinstance(x, tuple) else (x, False) for x in parts_b]
            parts_b = [(tokenizer.encode(x, add_special_tokens=False, **kwargs), s) for x, s in parts_b if x]

        parts_a, parts_b = truncate(parts_a, parts_b, max_length=max_seq_length, tokenizer=tokenizer)

        tokens_a = [token_id for part, _ in parts_a for token_id in part]
        tokens_b = [token_id for part, _ in parts_b for token_id in part] if parts_b else None

        input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)

        return input_ids, token_type_ids

def mlm_get_input_features(example: InputExample, parts_a, parts_b, tokenizer, max_seq_length, label_map, mask_id,
                        **kwargs) -> InputFeatures:

    input_ids, token_type_ids = encode(parts_a, parts_b, tokenizer, max_seq_length)

    attention_mask = [1] * len(input_ids)
    padding_length = max_seq_length - len(input_ids)

    if padding_length < 0:
        raise ValueError(f"Maximum sequence length is too small, got {len(input_ids)} input ids")

    input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(attention_mask) == max_seq_length
    assert len(token_type_ids) == max_seq_length


    label = label_map[example.label] if example.label is not None else -100
    logits = example.logits if example.logits else [-1]

    mlm_labels = get_mask_positions(input_ids, mask_id)
    # if self.wrapper.config.model_type == 'gpt2':
    #     # shift labels to the left by one
    #     mlm_labels.append(mlm_labels.pop(0))

    return {'input_ids': torch.tensor(input_ids).long(), 
            'attention_mask': torch.tensor(attention_mask).long(), 
            'token_type_ids': torch.tensor(token_type_ids).long(),
            'label': torch.tensor(label).long(), 
            'mlm_labels': mlm_labels.long(), 
            'logits': torch.tensor(logits).long(), 
            'idx': torch.tensor(example.idx).long()}


class Data_Loader(Dataset):

    def __init__(self, data, tokenizer, max_seq_length):
        self.dataset = data.dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mask = tokenizer.mask_token
        self.mask_id = tokenizer.mask_token_id
        self.label_map = data.label_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        [example_i, part_a, part_b] = self.dataset[index]

        input_features = mlm_get_input_features(example_i, part_a, part_b, self.tokenizer, self.max_seq_length, self.label_map, self.mask_id)
        input_features['label_gt'] = input_features['label']
        input_features['bool_unlabel'] = False

        return input_features


class Data_Loader_unlabel(Dataset):

    def __init__(self, label_data, unlabel_data, train_info_sort_dict, unlabel_info_sort_dict, model, tokenizer, args):
        self.label_dataset = label_data.dataset
        self.num_label_dataset = len(self.label_dataset)

        self.unlabel_data = unlabel_data.dataset

        self.unlabel_idx_select_list = unlabel_info_sort_dict['idx']
        self.unlabel_model_pre_select_list = unlabel_info_sort_dict['model_pre']
        self.unlabel_label_gt_select_list = unlabel_info_sort_dict['labels_gt']
        self.unlabel_probs_pre_select_list = unlabel_info_sort_dict['probs_pre']
        self.unlabel_probs_all_select_list = unlabel_info_sort_dict['probs_all']
        self.num_unlabel_data = len(self.unlabel_idx_select_list)

        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.mask = tokenizer.mask_token
        self.mask_id = tokenizer.mask_token_id
        self.label_map = label_data.label_map
        self.probs_all_template = torch.zeros(len(self.label_map.keys()))

    def __len__(self):
        return self.num_label_dataset + self.num_unlabel_data

    def __getitem__(self, index):
        if index < self.num_label_dataset:
            [example_i, part_a, part_b] = self.label_dataset[index]

            input_features = mlm_get_input_features(example_i, part_a, part_b, self.tokenizer, self.max_seq_length, self.label_map, self.mask_id)
            input_features['probs_all'] = self.probs_all_template.clone()
            input_features['probs_all'][input_features['label']] = 10
            input_features['label_gt'] = input_features['label']
        else:
            unlabel_index = self.unlabel_idx_select_list[index-self.num_label_dataset]
            
            [example_i, part_a, part_b] = self.unlabel_data[unlabel_index]

            input_features = mlm_get_input_features(example_i, part_a, part_b, self.tokenizer, self.max_seq_length, self.label_map, self.mask_id)            
            input_features['label'] = self.unlabel_model_pre_select_list[index-self.num_label_dataset]
            input_features['probs_all'] = self.unlabel_probs_all_select_list[index-self.num_label_dataset]
            input_features['label_gt'] = self.unlabel_label_gt_select_list[index-self.num_label_dataset]

        return input_features


class Data_Loader_unlabel_pet(Dataset):

    def __init__(self, label_data, unlabel_data, unlabel_info_sort_dict, ref_unlabel_info_sort_dict, model, tokenizer, args):
        self.label_dataset = label_data.dataset
        self.num_label_dataset = len(self.label_dataset)

        self.unlabel_data = unlabel_data.dataset
        unlabel_info_select_dict = get_unlabel_pet(ref_unlabel_info_sort_dict, args.unlabel_high_data_num)
        self.unlabel_info_select_dict = unlabel_info_select_dict
        self.unlabel_idx_select_list = unlabel_info_select_dict['idx']
        self.unlabel_model_pre_select_list = unlabel_info_select_dict['model_pre']
        self.unlabel_label_gt_select_list = unlabel_info_select_dict['labels_gt']
        self.unlabel_probs_pre_select_list = unlabel_info_select_dict['probs_pre']
        self.unlabel_probs_all_select_list = unlabel_info_select_dict['probs_all']
        self.num_unlabel_data = len(self.unlabel_idx_select_list)

        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.mask = tokenizer.mask_token
        self.mask_id = tokenizer.mask_token_id
        self.label_map = label_data.label_map
        self.probs_all_template = torch.zeros(len(self.label_map.keys()))

    def __len__(self):
        return self.num_label_dataset + self.num_unlabel_data

    def __getitem__(self, index):
        if index < self.num_label_dataset:
            [example_i, part_a, part_b] = self.label_dataset[index]

            input_features = mlm_get_input_features(example_i, part_a, part_b, self.tokenizer, self.max_seq_length, self.label_map, self.mask_id)
            input_features['label_gt'] = input_features['label']
            input_features['probs_all'] = self.probs_all_template.clone()
            input_features['probs_all'][input_features['label']] = 10
            input_features['bool_unlabel'] = False
        else:
            unlabel_index = self.unlabel_idx_select_list[index-self.num_label_dataset]
            
            [example_i, part_a, part_b] = self.unlabel_data[unlabel_index]

            input_features = mlm_get_input_features(example_i, part_a, part_b, self.tokenizer, self.max_seq_length, self.label_map, self.mask_id)            
            input_features['label'] = self.unlabel_model_pre_select_list[index-self.num_label_dataset]
            input_features['label_gt'] = self.unlabel_label_gt_select_list[index-self.num_label_dataset]
            input_features['probs_all'] = self.unlabel_probs_all_select_list[index-self.num_label_dataset]
            input_features['bool_unlabel'] = True
        return input_features



class Data_Loader_unlabel_final(Dataset):

    def __init__(self, label_data, unlabel_data, work_dir, unlabel_tag, retrain_rounds, tokenizer, args):
        self.label_dataset = label_data.dataset
        self.num_label_dataset = len(self.label_dataset)

        self.unlabel_data = unlabel_data.dataset

        # unlabel_info_select_dict = get_hard_unlabel_ratio(train_info_sort_dict, unlabel_info_sort_dict, model_fc_weight, args.unlabel_ratio, args.category_balance)
        label_select_num = args.batch_size * args.num_train_steps // args.num_class + 1
        unlabel_info_select_dict = get_unlabel_final_avg(work_dir, unlabel_tag, retrain_rounds, label_select_num)
        self.unlabel_info_select_dict = unlabel_info_select_dict
        self.unlabel_idx_select_list = unlabel_info_select_dict['idx']
        self.unlabel_model_pre_select_list = unlabel_info_select_dict['model_pre']
        self.unlabel_label_gt_select_list = unlabel_info_select_dict['labels_gt']
        self.unlabel_probs_pre_select_list = unlabel_info_select_dict['probs_pre']
        self.unlabel_probs_all_select_list = unlabel_info_select_dict['probs_all']
        self.num_unlabel_data = len(self.unlabel_idx_select_list)

        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.mask = tokenizer.mask_token
        self.mask_id = tokenizer.mask_token_id
        self.label_map = label_data.label_map
        self.probs_all_template = torch.zeros(len(self.label_map.keys()))

    def __len__(self):
        return self.num_label_dataset + self.num_unlabel_data

    def __getitem__(self, index):
        if index < self.num_label_dataset:
            [example_i, part_a, part_b] = self.label_dataset[index]

            input_features = mlm_get_input_features(example_i, part_a, part_b, self.tokenizer, self.max_seq_length, self.label_map, self.mask_id)
            input_features['label_gt'] = input_features['label']
            input_features['probs_all'] = self.probs_all_template.clone()
            input_features['probs_all'][input_features['label']] = 10
            input_features['bool_unlabel'] = False
        else:
            unlabel_index = self.unlabel_idx_select_list[index-self.num_label_dataset]
            
            [example_i, part_a, part_b] = self.unlabel_data[unlabel_index]

            input_features = mlm_get_input_features(example_i, part_a, part_b, self.tokenizer, self.max_seq_length, self.label_map, self.mask_id)            
            input_features['label'] = self.unlabel_model_pre_select_list[index-self.num_label_dataset]
            input_features['label_gt'] = self.unlabel_label_gt_select_list[index-self.num_label_dataset]
            input_features['probs_all'] = self.unlabel_probs_all_select_list[index-self.num_label_dataset]
            input_features['bool_unlabel'] = True
        return input_features
