# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import pickle
import random
import string
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, GPT2Tokenizer
import logging
import os
import sys
import errno
from os import path as osp
import shutil
from sklearn import cluster
from glob import glob

names = set()


def set_random_seed(seed, n_gpu):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
        

def __setup_custom_logger(name: str) -> logging.Logger:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    names.add(name)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    if name in names:
        return logging.getLogger(name)
    else:
        return __setup_custom_logger(name)


class LogitsList:
    """A list of logits obtained from a finetuned PET model"""

    def __init__(self, score: float, logits: List[List[float]]):
        """
        Create a new LogitsList.

        :param score: the corresponding PET model's score on the training set
        :param logits: the list of logits, where ``logits[i][j]`` is the score for label ``j`` at example ``i``
        """
        self.score = score
        self.logits = logits

    def __repr__(self):
        return 'LogitsList(score={}, logits[:2]={})'.format(self.score, self.logits[:2])

    def save(self, path: str) -> None:
        """Save this list to a file."""
        with open(path, 'w') as fh:
            fh.write(str(self.score) + '\n')
            for example_logits in self.logits:
                fh.write(' '.join(str(logit) for logit in example_logits) + '\n')

    @staticmethod
    def load(path: str, with_score: bool = True) -> 'LogitsList':
        """Load a list from a file"""
        score = -1
        logits = []
        with open(path, 'r') as fh:
            for line_idx, line in enumerate(fh.readlines()):
                line = line.rstrip('\n')
                if line_idx == 0 and with_score:
                    score = float(line)
                else:
                    logits.append([float(x) for x in line.split()])
        return LogitsList(score=score, logits=logits)


class InputExample(object):
    """A raw input example consisting of one or two segments of text and a label"""

    def __init__(self, guid, text_a, text_b=None, label=None, logits=None, meta: Optional[Dict] = None, idx=-1):
        """
        Create a new InputExample.

        :param guid: a unique textual identifier
        :param text_a: the sequence of text
        :param text_b: an optional, second sequence of text
        :param label: an optional label
        :param logits: an optional list of per-class logits
        :param meta: an optional dictionary to store arbitrary meta information
        :param idx: an optional numeric index
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.logits = logits
        self.idx = idx
        self.meta = meta if meta else {}

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @staticmethod
    def load_examples(path: str) -> List['InputExample']:
        """Load a set of input examples from a file"""
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: List['InputExample'], path: str) -> None:
        """Save a set of input examples to a file"""
        with open(path, 'wb') as fh:
            pickle.dump(examples, fh)


class InputFeatures(object):
    """A set of numeric features obtained from an :class:`InputExample`"""

    def __init__(self, input_ids, attention_mask, token_type_ids, label, mlm_labels=None, logits=None,
                 meta: Optional[Dict] = None, idx=-1):
        """
        Create new InputFeatures.

        :param input_ids: the input ids corresponding to the original text or text sequence
        :param attention_mask: an attention mask, with 0 = no attention, 1 = attention
        :param token_type_ids: segment ids as used by BERT
        :param label: the label
        :param mlm_labels: an optional sequence of labels used for auxiliary language modeling
        :param logits: an optional sequence of per-class logits
        :param meta: an optional dictionary to store arbitrary meta information
        :param idx: an optional numeric index
        """
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.mlm_labels = mlm_labels
        self.logits = logits
        self.idx = idx
        self.meta = meta if meta else {}

    def __repr__(self):
        return str(self.to_json_string())

    def pretty_print(self, tokenizer):
        return f'input_ids         = {tokenizer.convert_ids_to_tokens(self.input_ids)}\n' + \
               f'attention_mask    = {self.attention_mask}\n' + \
               f'token_type_ids    = {self.token_type_ids}\n' + \
               f'mlm_labels        = {self.mlm_labels}\n' + \
               f'logits            = {self.logits}\n' + \
               f'label             = {self.label}'

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class PLMInputFeatures(InputFeatures):
    """A set of numeric input features for a model pretrained with a permuted language modeling objective."""

    def __init__(self, *_, perm_mask, target_mapping, **kwargs):
        super().__init__(**kwargs)
        self.perm_mask = perm_mask
        self.target_mapping = target_mapping

    def pretty_print(self, tokenizer):
        return super().pretty_print(tokenizer) + '\n' + \
               f'perm_mask         = {self.perm_mask}\n' + \
               f'target_mapping    = {self.target_mapping}'


class DictDataset(Dataset):
    """A dataset of tensors that uses a dictionary for key-value mappings"""

    def __init__(self, **tensors):
        tensors.values()

        assert all(next(iter(tensors.values())).size(0) == tensor.size(0) for tensor in tensors.values())
        self.tensors = tensors

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).size(0)


def set_seed(seed: int):
    """ Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def eq_div(N, i):
    """ Equally divide N examples among i buckets. For example, `eq_div(12,3) = [4,4,4]`. """
    return [] if i <= 0 else [N // i + 1] * (N % i) + [N // i] * (i - N % i)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def remove_final_punc(s: str):
    """Remove the last character from a string if it is some form of punctuation"""
    return s.rstrip(string.punctuation)


def lowercase_first(s: str):
    """Lowercase the first letter of a string"""
    return s[0].lower() + s[1:]


def save_logits(path: str, logits: np.ndarray):
    """Save an array of logits to a file"""
    with open(path, 'w') as fh:
        for example_logits in logits:
            fh.write(' '.join(str(logit) for logit in example_logits) + '\n')
    pass


def save_predictions(path: str, wrapper, results: Dict):
    """Save a sequence of predictions to a file"""
    predictions_with_idx = []

    if wrapper.task_helper and wrapper.task_helper.output:
        predictions_with_idx = wrapper.task_helper.output
    else:
        inv_label_map = {idx: label for label, idx in wrapper.preprocessor.label_map.items()}
        for idx, prediction_idx in zip(results['indices'], results['predictions']):
            prediction = inv_label_map[prediction_idx]
            idx = idx.tolist() if isinstance(idx, np.ndarray) else int(idx)
            predictions_with_idx.append({'idx': idx, 'label': prediction})

    with open(path, 'w', encoding='utf8') as fh:
        for line in predictions_with_idx:
            fh.write(json.dumps(line) + '\n')


def softmax(x, temperature=1.0, axis=None):
    """Custom softmax implementation"""
    y = np.atleast_2d(x)

    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    y = y * float(temperature)
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    y = np.exp(y)

    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    p = y / ax_sum

    if len(x.shape) == 1:
        p = p.flatten()
    return p


def get_verbalization_ids(word: str, tokenizer: PreTrainedTokenizer, force_single_token: bool) -> Union[int, List[int]]:
    """
    Get the token ids corresponding to a verbalization

    :param word: the verbalization
    :param tokenizer: the tokenizer to use
    :param force_single_token: whether it should be enforced that the verbalization corresponds to a single token.
           If set to true, this method returns a single int instead of a list and throws an error if the word
           corresponds to multiple tokens.
    :return: either the list of token ids or the single token id corresponding to this word
    """
    kwargs = {'add_prefix_space': True} if isinstance(tokenizer, GPT2Tokenizer) else {}
    ids = tokenizer.encode(word, add_special_tokens=False, **kwargs)
    if not force_single_token:
        return ids
    if len(ids) != 1:
        print('Verbalization "{word}" does not correspond to a single token, got {ids}')
    verbalization_id = ids[0]
    if verbalization_id in tokenizer.all_special_ids:
        print('Verbalization {word} is mapped to a special token {ids}')
    return verbalization_id


def trim_input_ids(input_ids: torch.tensor, pad_token_id, mask_token_id, num_masks: int):
    """
    Trim a sequence of input ids by removing all padding tokens and keeping at most a specific number of mask tokens.

    :param input_ids: the sequence of input token ids
    :param pad_token_id: the id of the pad token
    :param mask_token_id: the id of the mask tokens
    :param num_masks: the number of masks to keeps
    :return: the trimmed sequence of input ids
    """
    assert input_ids.shape[0] == 1
    input_ids_without_pad = [x for x in input_ids[0] if x != pad_token_id]

    trimmed_input_ids = []
    mask_count = 0
    for input_id in input_ids_without_pad:
        if input_id == mask_token_id:
            if mask_count >= num_masks:
                continue
            mask_count += 1
        trimmed_input_ids.append(input_id)

    return torch.tensor([trimmed_input_ids], dtype=torch.long, device=input_ids.device)


def exact_match(predictions: np.ndarray, actuals: np.ndarray, question_ids: np.ndarray):
    """Compute the exact match (EM) for a sequence of predictions and actual labels"""
    unique_questions = set(question_ids)

    q_actuals = list(zip(question_ids, actuals))
    q_predictions = list(zip(question_ids, predictions))

    actuals_per_question = defaultdict(list)
    predictions_per_question = defaultdict(list)

    for qid, val in q_actuals:
        actuals_per_question[qid].append(val)
    for qid, val in q_predictions:
        predictions_per_question[qid].append(val)

    em = 0
    for qid in unique_questions:
        if actuals_per_question[qid] == predictions_per_question[qid]:
            em += 1
    em /= len(unique_questions)

    return em


def distillation_loss(predictions, targets, temperature):
    """Compute the distillation loss (KL divergence between predictions and targets) as described in the PET paper"""
    p = F.log_softmax(predictions / temperature, dim=1)
    q = F.softmax(targets / temperature, dim=1)
    return F.kl_div(p, q, reduction='sum') * (temperature ** 2) / predictions.shape[0]

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def get_pre_info(logits_all, label_all, idxs_all, num_class):
    probs_all = torch.softmax(logits_all, dim=1)
    probs_label, model_pres_all = torch.max(probs_all, dim=1)
    
    probs_sort_indexs = torch.sort(-probs_label)[1]
    # feats_mask_all_sort = feats_mask_all[probs_sort_indexs]
    probs_all_sort = logits_all[probs_sort_indexs]
    probs_label_sort = probs_label[probs_sort_indexs]
    model_pre_sort = model_pres_all[probs_sort_indexs]
    label_all_sort =  label_all[probs_sort_indexs]
    idxs_all_sort =  idxs_all[probs_sort_indexs]

    pre_info_dict = {'probs_all': probs_all_sort, 'probs_pre': probs_label_sort, 
        'model_pre': model_pre_sort, 'labels_gt': label_all_sort, 'idx': idxs_all_sort}

    # pre_info_dict = {'feats_mask': feats_mask_all_sort, 'probs_all': probs_all_sort, 'probs_pre': probs_label_sort, 
    #     'model_pre': model_pre_sort, 'labels_gt': label_all_sort, 'idx': idxs_all_sort}

    return  pre_info_dict

def get_specific_label_info(unlabel_info_sort_dict, label_list):
    model_pre = unlabel_info_sort_dict['model_pre']
    index_list = None
    for label_i in label_list:
        model_pre_i = torch.where(model_pre == label_i)[0]
        if index_list == None:
            index_list = model_pre_i.clone()
        else:
            index_list = torch.cat((index_list, model_pre_i), dim=0)
    
    for key in unlabel_info_sort_dict.keys():
        unlabel_info_sort_dict[key] = unlabel_info_sort_dict[key][index_list]

    return  unlabel_info_sort_dict


def get_unlabel_pet(ref_pre_info_sort_dict, unlabel_data_num):

    # ref_feats_mask = ref_pre_info_sort_dict['feats_mask']
    ref_probs_pre = ref_pre_info_sort_dict['probs_pre']
    ref_probs_all = ref_pre_info_sort_dict['probs_all']
    ref_model_pre = ref_pre_info_sort_dict['model_pre']
    ref_label_gt = ref_pre_info_sort_dict['labels_gt']
    ref_idx = ref_pre_info_sort_dict['idx']
    ref_idx_trans = torch.sort(ref_idx)[1]
    label_list = sorted(list(set(ref_model_pre.cpu().numpy())))
    ref_model_pre_select_dict, ref_label_gt_select_dict, ref_idx_select_dict, ref_probs_pre_select_dict, ref_probs_all_select_dict = {}, {}, {}, {}, {}
    total_num = ref_model_pre.shape[0]
    rng_np = np.random.RandomState()

    for label_i in label_list:

        ref_model_pre_i_index = torch.where(ref_model_pre==label_i)[0]
        sample_prob = ref_probs_pre[ref_model_pre_i_index] / torch.sum(ref_probs_pre[ref_model_pre_i_index])
        ref_model_pre_i_index_np = ref_model_pre_i_index.numpy()
        sample_prob_np = sample_prob.numpy()
        if ref_model_pre_i_index_np.shape[0] > unlabel_data_num:
            ref_model_pre_i_index_select = rng_np.choice(a=ref_model_pre_i_index_np, size=unlabel_data_num, replace=False, p=sample_prob_np)
        else:
            ref_model_pre_i_index_select = ref_model_pre_i_index_np

        ref_model_pre_i_select = ref_model_pre[ref_model_pre_i_index_select]
        ref_label_gt_i_select = ref_label_gt[ref_model_pre_i_index_select]
        ref_idx_i_select = ref_idx[ref_model_pre_i_index_select]
        ref_probs_pre_i_select = ref_probs_pre[ref_model_pre_i_index_select]
        ref_probs_all_i_select = ref_probs_all[ref_model_pre_i_index_select]

        ref_probs_pre_select_dict[label_i] = ref_probs_pre_i_select
        ref_model_pre_select_dict[label_i] = ref_model_pre_i_select
        ref_label_gt_select_dict[label_i] = ref_label_gt_i_select
        ref_idx_select_dict[label_i] = ref_idx_i_select
        ref_probs_all_select_dict[label_i] = ref_probs_all_i_select
        acc_select_i = torch.sum(ref_model_pre_i_select == ref_label_gt_i_select) / ref_label_gt_i_select.shape[0]

        ref_model_pre_i_index_front = ref_model_pre_i_index[0: unlabel_data_num]
        ref_model_pre_i_front = ref_model_pre[ref_model_pre_i_index_front]
        ref_label_gt_i_front = ref_label_gt[ref_model_pre_i_index_front]
        ref_idx_i_front = ref_idx[ref_model_pre_i_index_front]
        ref_probs_pre_i_front = ref_probs_pre[ref_model_pre_i_index_front]
        ref_probs_all_i_front = ref_probs_all[ref_model_pre_i_index_front]
        acc_select_front_i = torch.sum(ref_model_pre_i_front == ref_label_gt_i_front) / ref_label_gt_i_front.shape[0]
        acc_i = torch.sum(ref_model_pre[ref_model_pre_i_index] == ref_label_gt[ref_model_pre_i_index]) / ref_model_pre_i_index.shape[0]
        print('high_prob label: %i, num: %i acc: %.4f acc_select: %.4f acc_front: %.4f'\
            %(label_i, unlabel_data_num, acc_i, acc_select_i, acc_select_front_i))


    pre_num_all, pre_correct_num_all = 0,0
    pre_info_sort_select_dict = defaultdict(list)
    for label_i in label_list:
        model_pre_i = ref_model_pre_select_dict[label_i]
        label_gt_i= ref_label_gt_select_dict[label_i]
        idx_i = ref_idx_select_dict[label_i]
        probs_pre_i = ref_probs_pre_select_dict[label_i]
        probs_all_i = ref_probs_all_select_dict[label_i]
        pre_info_sort_select_dict['model_pre'].extend(model_pre_i)
        pre_info_sort_select_dict['labels_gt'].extend(label_gt_i)
        pre_info_sort_select_dict['idx'].extend(idx_i)
        pre_info_sort_select_dict['probs_pre'].extend(probs_pre_i)
        pre_info_sort_select_dict['probs_all'].extend(probs_all_i)
        pre_num = model_pre_i.shape[0]
        pre_correct_num = torch.sum(model_pre_i == label_gt_i)
        pre_num_all += pre_num
        pre_correct_num_all += pre_correct_num
        acc_prob = pre_correct_num / pre_num
        print('     label: %i, acc: %i/%i, %.4f'%(label_i, pre_correct_num, pre_num, acc_prob))

    print('===>acc: %i/%i, %.4f'%(pre_correct_num_all, pre_num_all, pre_correct_num_all/pre_num_all))
    return pre_info_sort_select_dict

def get_model_train_sort(dir, save_tag, retrain_round):
    if retrain_round == 1:
        model_dir_list = glob(os.path.join(dir, '*_pattern_*'))
    else:
        model_dir_list = glob(os.path.join(dir, '%s_*_pattern_*_round_%i'%(save_tag, retrain_round-1)))
    
    model_acc_list = []
    for model_dir_i in model_dir_list:
        model_best_i = glob(os.path.join(model_dir_i, 'model_best/model_best_*.pth'))[0]
        model_acc_i = float(os.path.basename(model_best_i).replace('.pth', '').split('_')[5])
        model_acc_list.append(model_acc_i)

    sort_index = list(np.argsort(np.array(model_acc_list)))
    model_acc_sort, model_dir_sort = [], []
    for sort_index_i in sort_index: 
        model_acc_sort.append(model_acc_list[sort_index_i])
        model_dir_sort.append(model_dir_list[sort_index_i])

    return model_dir_sort, model_acc_sort

def model_acc_sort(model_dir_list, model_acc_list):

    sort_index = list(np.argsort(np.array(model_acc_list)))
    model_acc_sort, model_dir_sort = [], []
    for sort_index_i in sort_index: 
        model_acc_sort.append(model_acc_list[sort_index_i])
        model_dir_sort.append(model_dir_list[sort_index_i])

    return model_dir_sort, model_acc_sort


def get_unlabel_final_avg(dir, save_tag, retrain_rounds, label_select_num):
    if retrain_rounds == 0:
        model_dir_list = glob(os.path.join(dir, '*_pattern_*'))
    else:
        model_dir_list = glob(os.path.join(dir, '%s_*_pattern_*_round_%i'%(save_tag, retrain_rounds)))
    
    unlabel_info_sort_dict_fuse = {}
    for model_dir_i in sorted(model_dir_list):
        unlabel_info_sort_dict_dir_i = os.path.join(model_dir_i, 'pre_info_sort_dict_model_test.pth')
        unlabel_info_sort_dict_i = torch.load(unlabel_info_sort_dict_dir_i)
        idx_i = unlabel_info_sort_dict_i['idx']
        idx_sort_i, idx_trans_i = torch.sort(idx_i)
        probs_pre_i = unlabel_info_sort_dict_i['probs_pre'][idx_trans_i]
        probs_all_i = unlabel_info_sort_dict_i['probs_all'][idx_trans_i]
        model_pre_i = unlabel_info_sort_dict_i['model_pre'][idx_trans_i]
        label_gt_i = unlabel_info_sort_dict_i['labels_gt'][idx_trans_i]
        acc_i = torch.sum(label_gt_i == model_pre_i) / model_pre_i.shape[0]
        print('======>acc_%s: %.4f'%(unlabel_info_sort_dict_dir_i, acc_i))

        if 'probs_all' in unlabel_info_sort_dict_fuse.keys():
            unlabel_info_sort_dict_fuse['probs_all'] += probs_all_i
        else:
            unlabel_info_sort_dict_fuse['probs_all'] = probs_all_i
            unlabel_info_sort_dict_fuse['labels_gt'] = label_gt_i
            unlabel_info_sort_dict_fuse['idx'] = idx_sort_i
    
    unlabel_info_sort_dict_fuse['probs_all'] = unlabel_info_sort_dict_fuse['probs_all'] / len(model_dir_list)
    probs_all_fuse = torch.softmax(unlabel_info_sort_dict_fuse['probs_all'], dim=1)
    probs_label_fuse, model_pres_all_fuse = torch.max(probs_all_fuse, dim=1)
    unlabel_info_sort_dict_fuse['probs_pre'] = probs_label_fuse
    unlabel_info_sort_dict_fuse['model_pre'] = model_pres_all_fuse

    label_list = sorted(list(set(label_gt_i.cpu().numpy())))
    select_idx = []
    for label_i in label_list:
        model_pre_i_idx = torch.where(unlabel_info_sort_dict_fuse['model_pre']==label_i)[0].tolist()
        random.shuffle(model_pre_i_idx)
        select_idx.extend(model_pre_i_idx[0: label_select_num])
    unlabel_info_sort_dict_fuse['probs_all'] = unlabel_info_sort_dict_fuse['probs_all'][select_idx]
    unlabel_info_sort_dict_fuse['labels_gt'] = unlabel_info_sort_dict_fuse['labels_gt'][select_idx]
    unlabel_info_sort_dict_fuse['idx'] = unlabel_info_sort_dict_fuse['idx'][select_idx]
    unlabel_info_sort_dict_fuse['probs_pre'] = unlabel_info_sort_dict_fuse['probs_pre'][select_idx]
    unlabel_info_sort_dict_fuse['model_pre'] = unlabel_info_sort_dict_fuse['model_pre'][select_idx]
    acc_fuse = torch.sum(unlabel_info_sort_dict_fuse['model_pre'] == unlabel_info_sort_dict_fuse['labels_gt']) / unlabel_info_sort_dict_fuse['model_pre'].shape[0]
    print('===>acc_fuse: %.4f'%(acc_fuse))

    return unlabel_info_sort_dict_fuse        

def get_unlabel_best_model(dir, save_tag, retrain_rounds):
    if retrain_rounds == 0:
        model_dir_list = glob(os.path.join(dir, '*_pattern_*'))
    else:
        model_dir_list = glob(os.path.join(dir, '%s_*_pattern_*_round_%i'%(save_tag, retrain_rounds)))
    
    model_acc_list, model_dict_list = [], []
    for model_dir_i in model_dir_list:
        model_best_i = glob(os.path.join(model_dir_i, 'model_best/model_best_*.pth'))[0]
        model_dict_list.append(model_best_i)
        model_acc_i = float(os.path.basename(model_best_i).replace('.pth', '').split('_')[5])
        model_acc_list.append(model_acc_i)

    best_index = np.argsort(np.array(model_acc_list))[-1]

    return model_dict_list[best_index]

