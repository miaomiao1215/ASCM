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

"""
This file contains the logic for loading training and test data for all tasks.
"""

import csv
import json
import os
import random
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable, Pattern

import utils
from utils import InputExample, InputFeatures, DictDataset
from typing import Tuple, List, Union, Dict
import torch
from transformers import BertConfig, BertTokenizer, RobertaConfig, RobertaTokenizer
import models
import argparse
from torch import nn
from data_loader import mlm_get_input_features
import string
import numpy as np


logger = utils.get_logger('root')



def _shuffle_and_restrict(examples: List[InputExample], num_examples: int, seed: int = 42) -> List[InputExample]:
    """
    Shuffle a list of examples and restrict it to a given maximum size.

    :param examples: the examples to shuffle and restrict
    :param num_examples: the maximum number of examples
    :param seed: the random seed for shuffling
    :return: the first ``num_examples`` elements of the shuffled list
    """
    if 0 < num_examples < len(examples):
        random.Random(seed).shuffle(examples)
        examples = examples[:num_examples]
    return examples


class LimitedExampleList:
    def __init__(self, labels: List[str], max_examples=-1):
        """
        Implementation of a list that stores only a limited amount of examples per label.

        :param labels: the set of all possible labels
        :param max_examples: the maximum number of examples per label. This can either be a fixed number,
               in which case `max_examples` examples are loaded for every label, or a list with the same size as
               `labels`, in which case at most `max_examples[i]` examples are loaded for label `labels[i]`.
        """
        self._labels = labels
        self._examples = []
        self._examples_per_label = defaultdict(int)

        if isinstance(max_examples, list):
            self._max_examples = dict(zip(self._labels, max_examples))
        else:
            self._max_examples = {label: max_examples for label in self._labels}

    def is_full(self):
        """Return `true` iff no more examples can be added to this list"""
        for label in self._labels:
            if self._examples_per_label[label] < self._max_examples[label] or self._max_examples[label] < 0:
                return False
        return True

    def add(self, example: InputExample) -> bool:
        """
        Add a new input example to this list.

        :param example: the example to add
        :returns: `true` iff the example was actually added to the list
        """
        label = example.label
        if self._examples_per_label[label] < self._max_examples[label] or self._max_examples[label] < 0:
            self._examples_per_label[label] += 1
            self._examples.append(example)
            return True
        return False

    def to_list(self):
        return self._examples



class YahooAnswersProcessor(ABC):
    """Processor for the Yahoo Answers data set."""

    def __init__(self, data_dir, mode, pattern_id, tokenizer, config, no_pattern=False):
        super(YahooAnswersProcessor, self).__init__()
        self.pattern_id = pattern_id
        self.no_pattern = no_pattern
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.config = config
        if 'T5' in tokenizer.name_or_path:
            self.mask = '<extra_id_0>'
        else:
            self.mask = tokenizer.mask_token
        self.labels = self.get_labels()
        self.label_map = self.get_label_map()
        self.label_verbalizer_map = {
            "1": ["Society"],
            "2": ["Science"],
            "3": ["Health"],
            "4": ["Education"],
            "5": ["Computer"],
            "6": ["Sports"],
            "7": ["Business"],
            "8": ["Entertainment"],
            "9": ["Relationship"],
            "10": ["Politics"],
        }
        self.get_verbalizer_token_ids()

        if mode == 'train':
            self.dataset = self._create_examples(data_dir, "train", "train")
        elif mode == 'eval':
            self.dataset = self._create_examples(data_dir, "dev", "dev")
        elif mode in 'unlabeled':
            self.dataset = self._create_examples(data_dir, "unlabeled", "train")

    def __len__(self):
        return self.dataset_num

    def get_labels(self):
        return ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

    def get_verbalizer_token_ids(self):
        verbalizer_ids = []
        for label_i in self.labels:
            verbalizer_i = self.label_verbalizer_map[label_i][0]
            verbalizer_ids.append(self.tokenizer.encode(verbalizer_i, add_special_tokens=False)[0])

        self.verbalizer_ids = verbalizer_ids

    def shortenable(self, s):
        return s, True

    def get_label_map(self):
        label_map = {}
        for idx, label_i in enumerate(self.labels):
            label_map[label_i] = idx
        return label_map

    def get_parts(self, example: InputExample):

        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        if self.no_pattern:
            return [text_a, text_b], []

        if self.pattern_id == 0:
            return [self.mask, ':', text_a, text_b], []
        elif self.pattern_id == 1:
            return [self.mask, 'Question:', text_a, text_b], []
        elif self.pattern_id == 2:
            return [text_a, '(', self.mask, ')', text_b], []
        elif self.pattern_id == 3:
            return [text_a, text_b, '(', self.mask, ')'], []
        elif self.pattern_id == 4:
            return ['[ Category:', self.mask, ']', text_a, text_b], []
        elif self.pattern_id == 5:
            return [self.mask, '-', text_a, text_b], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return self.label_verbalizer_map[label]

    def _create_examples(self, path: str, set_type: str, guid_str: str) -> List[InputExample]:
        
        dataset = []

        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, question_title, question_body, answer = row
                guid = "%s-%s" % (guid_str, idx)
                text_a = ' '.join([question_title.replace('\\n', ' ').replace('\\', ' '),
                                   question_body.replace('\\n', ' ').replace('\\', ' ')])
                text_b = answer.replace('\\n', ' ').replace('\\', ' ')

                if set_type == 'unlabeled':
                    label = self.labels[0]

                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx)
                part_a, part_b = self.get_parts(example)

                dataset.append([example, part_a, part_b])

        label_distribution = Counter(example.label for [example, _, _] in dataset)
        logger.info(f"Returning {len(dataset)} {set_type} examples with label dist.: {list(label_distribution.items())}")
        self.dataset_num = len(dataset)
        return dataset


class YelpFullProcessor(ABC):
    """Processor for the Yahoo Answers data set."""

    def __init__(self, data_dir, mode, pattern_id, tokenizer, config, no_pattern=False):
        super(YelpFullProcessor, self).__init__()
        self.pattern_id = pattern_id
        self.no_pattern = no_pattern
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.config = config
        self.mask = tokenizer.mask_token
        self.labels = self.get_labels()
        self.label_map = self.get_label_map()
        self.label_verbalizer_map = {
            "1": ["terrible"],
            "2": ["bad"],
            "3": ["okay"],
            "4": ["good"],
            "5": ["great"],
        }
        self.get_verbalizer_token_ids()

        if mode == 'train':
            self.dataset = self._create_examples(data_dir, "train", "train")
        elif mode == 'eval':
            self.dataset = self._create_examples(data_dir, "dev", "dev")
        elif mode in 'unlabeled':
            self.dataset = self._create_examples(data_dir, "unlabeled", "train")

    def __len__(self):
        return self.dataset_num

    def get_labels(self):
        return ["1", "2", "3", "4", "5"]

    def get_verbalizer_token_ids(self):
        verbalizer_ids = []
        for label_i in self.labels:
            verbalizer_i = self.label_verbalizer_map[label_i][0]
            verbalizer_ids.append(self.tokenizer.encode(verbalizer_i, add_special_tokens=False)[0])

        self.verbalizer_ids = verbalizer_ids

    def shortenable(self, s):
        return s, True

    def get_label_map(self):
        label_map = {}
        for idx, label_i in enumerate(self.labels):
            label_map[label_i] = idx
        return label_map

    def get_parts(self, example: InputExample):

        text = self.shortenable(example.text_a)

        if self.no_pattern:
            return [text], []

        if self.pattern_id == 0:
            return ['It was', self.mask, '.', text], []
        elif self.pattern_id == 1:
            return [text, '. All in all, it was', self.mask, '.'], []
        elif self.pattern_id == 2:
            return ['Just', self.mask, "!"], [text]
        elif self.pattern_id == 3:
            return [text], ['In summary, the restaurant is', self.mask, '.']
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return self.label_verbalizer_map[label]

    def _create_examples(self, path: str, set_type: str, guid_str: str) -> List[InputExample]:
        
        dataset = []

        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, body = row
                guid = "%s-%s" % (guid_str, idx)
                text_a = body.replace('\\n', ' ').replace('\\', ' ')

                if set_type == 'unlabeled':
                    label = self.labels[0]

                example = InputExample(guid=guid, text_a=text_a, label=label, idx=idx)
                part_a, part_b = self.get_parts(example)

                dataset.append([example, part_a, part_b])

        label_distribution = Counter(example.label for [example, _, _] in dataset)
        logger.info(f"Returning {len(dataset)} {set_type} examples with label dist.: {list(label_distribution.items())}")
        self.dataset_num = len(dataset)
        return dataset

class AGNewsProcessor(ABC):
    """Processor for the Yahoo Answers data set."""

    def __init__(self, data_dir, mode, pattern_id, tokenizer, config, no_pattern=False):
        super(AGNewsProcessor, self).__init__()
        self.pattern_id = pattern_id
        self.no_pattern = no_pattern
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.config = config
        self.mask = tokenizer.mask_token
        self.labels = self.get_labels()
        self.label_map = self.get_label_map()
        self.label_verbalizer_map = {
            "1": ["World"],
            "2": ["Sports"],
            "3": ["Business"],
            "4": ["Tech"],
        }
        self.get_verbalizer_token_ids()

        if mode == 'train':
            self.dataset = self._create_examples(data_dir, "train", "train")
        elif mode == 'eval':
            self.dataset = self._create_examples(data_dir, "dev", "dev")
        elif mode in 'unlabeled':
            self.dataset = self._create_examples(data_dir, "unlabeled", "train")

    def __len__(self):
        return self.dataset_num

    def get_labels(self):
        return ["1", "2", "3", "4"]

    def get_verbalizer_token_ids(self):
        verbalizer_ids = []
        for label_i in self.labels:
            verbalizer_i = self.label_verbalizer_map[label_i][0]
            verbalizer_ids.append(self.tokenizer.encode(verbalizer_i, add_special_tokens=False)[0])

        self.verbalizer_ids = verbalizer_ids

    def shortenable(self, s):
        return s, True

    def get_label_map(self):
        label_map = {}
        for idx, label_i in enumerate(self.labels):
            label_map[label_i] = idx
        return label_map

    def get_parts(self, example: InputExample):

        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        if self.no_pattern:
            return [text_a, text_b], []

        if self.pattern_id == 0:
            return [self.mask, 'News:', text_a, text_b], []
        elif self.pattern_id == 1:
            return [self.mask, ':', text_a, text_b], []
        elif self.pattern_id == 2:
            return [text_a, '(', self.mask, ')', text_b], []
        elif self.pattern_id == 3:
            return [text_a, text_b, '(', self.mask, ')'], []
        elif self.pattern_id == 4:
            return ['[ Category:', self.mask, ']', text_a, text_b], []
        elif self.pattern_id == 5:
            return [self.mask, '-', text_a, text_b], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return self.label_verbalizer_map[label]

    def _create_examples(self, path: str, set_type: str, guid_str: str) -> List[InputExample]:
        
        dataset = []

        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline, body = row
                guid = "%s-%s" % (guid_str, idx)
                text_a = headline.replace('\\', ' ')
                text_b = body.replace('\\', ' ')

                if set_type == 'unlabeled':
                    label = self.labels[0]

                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx)

                part_a, part_b = self.get_parts(example)

                dataset.append([example, part_a, part_b])

        label_distribution = Counter(example.label for [example, _, _] in dataset)
        logger.info(f"Returning {len(dataset)} {set_type} examples with label dist.: {list(label_distribution.items())}")
        self.dataset_num = len(dataset)
        return dataset


class MNLI_Processor(ABC):
    """Processor for the Yahoo Answers data set."""

    def __init__(self, data_dir, mode, pattern_id, tokenizer, config, no_pattern=False):
        super(MNLI_Processor, self).__init__()
        self.pattern_id = pattern_id
        self.no_pattern = no_pattern
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.config = config
        self.mask = tokenizer.mask_token
        self.labels = self.get_labels()
        self.label_map = self.get_label_map()
        self.label_verbalizer_map = {
            "contradiction": ["No"],
            "entailment": ["Yes"],
            "neutral": ["Maybe"],
        }
        self.get_verbalizer_token_ids()

        if mode == 'train':
            self.dataset = self._create_examples(data_dir, "train", "train")
        elif mode == 'eval':
            self.dataset = self._create_examples(data_dir, "dev", "dev")
        elif mode in 'unlabeled':
            self.dataset = self._create_examples(data_dir, "unlabeled", "train")

    def __len__(self):
        return self.dataset_num

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    def get_verbalizer_token_ids(self):
        verbalizer_ids = []
        for label_i in self.labels:
            verbalizer_i = self.label_verbalizer_map[label_i][0]
            verbalizer_ids.append(self.tokenizer.encode(verbalizer_i, add_special_tokens=False)[0])

        self.verbalizer_ids = verbalizer_ids

    def shortenable(self, s):
        return s, True

    def get_label_map(self):
        label_map = {}
        for idx, label_i in enumerate(self.labels):
            label_map[label_i] = idx
        return label_map

        
    def get_parts(self, example: InputExample):

        text_a = self.shortenable(example.text_a.rstrip(string.punctuation))
        text_b = self.shortenable(example.text_b)

        if self.no_pattern:
            return [text_a, text_b], []

        if self.pattern_id == 0:
            return ['"', text_a, '" ?'], [self.mask, ', "', text_b, '"']
        elif self.pattern_id == 1:
            return [text_a, '?'], [self.mask, ',', text_b]
        elif self.pattern_id == 2:
            return [text_a, '?', self.mask, ',', text_b], []
        elif self.pattern_id == 3:
            return ['"', text_a, '" ?', self.mask, ', "', text_b, '"'], []

    def verbalize(self, label) -> List[str]:
        return self.label_verbalizer_map[label]

    def _create_examples(self, path: str, set_type: str, guid_str: str) -> List[InputExample]:
        
        dataset = []

        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, text_a, text_b = row
                if label == '-':
                    continue
                guid = "%s-%s" % (guid_str, idx)
                text_a = text_a.strip()
                text_b = text_b.strip()

                if set_type == 'unlabeled':
                    label = self.labels[0]

                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx)
                part_a, part_b = self.get_parts(example)

                dataset.append([example, part_a, part_b])

        label_distribution = Counter(example.label for [example, _, _] in dataset)
        logger.info(f"Returning {len(dataset)} {set_type} examples with label dist.: {list(label_distribution.items())}")
        self.dataset_num = len(dataset)
        return dataset

def generate_default_inputs(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Generate the default inputs required by almost every language model."""
    inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'token_type_ids': batch['token_type_ids']}

    return inputs


"""Create dataset"""

__factory = {
    'YahooAnswers': YahooAnswersProcessor, # yahoo_answers_csv
    'YelpFull': YelpFullProcessor, # yelp_review_full_csv
    'AGNews': AGNewsProcessor,# ag_news_csv
    'MNLI': MNLI_Processor, # multinli
}

def get_names():
    return __factory.keys()

def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](*args, **kwargs)




