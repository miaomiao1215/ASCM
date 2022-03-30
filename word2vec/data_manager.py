from __future__ import print_function, absolute_import
import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import collections
import json
import os.path as osp
from scipy.io import loadmat
import numpy as np
from typing import Optional, List, Union, Dict
from tqdm import tqdm
import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM, BertForMaskedLM, BertTokenizer, GPT2Tokenizer, AlbertForMaskedLM, AlbertTokenizer, XLMRobertaForMaskedLM, XLMRobertaTokenizer
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler

class Sim_Word_ansys(object):

    def __init__(self, file_path):

        self.file_path = file_path
        self.tokenizer = RobertaTokenizer.from_pretrained('xxx/roberta_large')
        self.model_token_embed = RobertaForMaskedLM.from_pretrained('xxx/roberta_large').lm_head.decoder.state_dict()['weight']
        class_dict = {
            'yahoo': ["Society", "Science", "Health", "Education", "Computer", "Sports", "Business", "Entertainment", "Relationship", "Politics"],
            'yelp': ["terrible", "bad", "okay", "good", "great"],
            'agnews': ["World", "Sports", "Business", "Technology"],
            'mnli': ['No', 'Yes', 'Maybe'],
        }
        self.dataset = self.process_data()

    def __len__(self):
        return self.dataset_size

    def process_data(self, ):
        txt_r = open(self.file_path, 'r')
        self.token_dict = defaultdict(list)
        self.token_id_dict = defaultdict(list)
        self.label_list = []
        self.word_list, self.token_list = [], []
        self.data_list = []
        lines = txt_r.readlines()
        idx = 0
        self.dataset_size = 0
        for line_i in lines:
            sim_words_list = line_i.split(' ')
            for sim_words_i in sim_words_list:
                if sim_words_i == '\n':
                    continue
                sim_words_i = sim_words_i.strip()
                sim_words_i_upper = sim_words_i[0].upper() + sim_words_i[1: ]
                sim_words_i_upper = ' ' + sim_words_i_upper
                token_id_i_upper = self.tokenizer.encode(sim_words_i_upper, add_special_tokens=False)[0]
                token_i_upper = self.tokenizer.convert_ids_to_tokens(token_id_i_upper)
                if len(token_i_upper)<2:
                    continue
                if token_id_i_upper in self.token_id_dict[idx]:
                    continue
                self.token_id_dict[idx].append(token_id_i_upper)
                self.token_dict[idx].append(token_i_upper)
                self.label_list.append(idx)
                self.word_list.append(sim_words_i)
                self.token_list.append(token_i_upper)
                self.data_list.append(self.model_token_embed[token_id_i_upper: token_id_i_upper + 1])


                sim_words_i_lower = sim_words_i_upper.lower()
                token_id_i_lower = self.tokenizer.encode(sim_words_i_lower, add_special_tokens=False)[0]
                token_i_lower = self.tokenizer.convert_ids_to_tokens(token_id_i_lower)
                if len(token_i_lower)<2:
                    continue
                if token_id_i_lower in self.token_id_dict[idx]:
                    continue
                self.token_id_dict[idx].append(token_id_i_lower)
                self.token_dict[idx].append(token_i_lower)
                self.label_list.append(idx)
                self.word_list.append(sim_words_i)
                self.token_list.append(token_i_lower)
                self.data_list.append(self.model_token_embed[token_id_i_lower: token_id_i_lower + 1])

            print(self.token_dict[idx])
            print(self.token_id_dict[idx])
            self.dataset_size += len(self.token_dict[idx])
            idx += 1
        
        dataset = torch.cat(self.data_list, dim=0)
        return dataset

class Sim_Word(object):

    def __init__(self, file_path, tokenizer, token_embedding):

        self.file_path = file_path
        self.tokenizer = tokenizer
        self.model_token_embed = token_embedding
        class_dict = {
            'yahoo': ["Society", "Science", "Health", "Education", "Computer", "Sports", "Business", "Entertainment", "Relationship", "Politics"],
            'yelp': ["terrible", "bad", "okay", "good", "great"],
            'agnews': ["World", "Sports", "Business", "Tech"],
            'mnli': ['No', 'Yes', 'Maybe'],
        }
        self.dataset = self.process_data()

    def __len__(self):
        return self.dataset_size

    def process_data(self, ):
        txt_r = open(self.file_path, 'r')
        self.token_dict = defaultdict(list)
        self.token_id_dict = defaultdict(list)
        self.label_list = []
        self.data_list = []
        lines = txt_r.readlines()
        idx = 0
        self.dataset_size = 0
        word_token_num_dict = defaultdict(list)
        for line_i in lines:
            sim_words_list = line_i.split(' ')
            for sim_words_i in sim_words_list:
                sim_words_i = sim_words_i.strip()
                sim_words_i_upper = sim_words_i[0].upper() + sim_words_i[1: ]

                sim_words_i_upper_space = ' ' + sim_words_i_upper
                word_token_num_dict[len(self.tokenizer.encode(sim_words_i_upper_space, add_special_tokens=False))].append(sim_words_i_upper_space)
                token_id_i_upper_space = self.tokenizer.encode(sim_words_i_upper_space, add_special_tokens=False)[0]
                token_i_upper_space = self.tokenizer.convert_ids_to_tokens(token_id_i_upper_space)
                if len(token_i_upper_space)<3:
                    continue
                if token_id_i_upper_space in self.token_id_dict[idx]:
                    continue
                self.token_id_dict[idx].append(token_id_i_upper_space)
                self.token_dict[idx].append(token_i_upper_space)
                self.label_list.append(idx)
                self.data_list.append(self.model_token_embed[token_id_i_upper_space: token_id_i_upper_space + 1])


                sim_words_i_upper_space = ' ' + sim_words_i.lower()
                token_id_i_lower_space = self.tokenizer.encode(sim_words_i_upper_space, add_special_tokens=False)[0]
                token_i_lower_space = self.tokenizer.convert_ids_to_tokens(token_id_i_lower_space)
                word_token_num_dict[len(self.tokenizer.encode(sim_words_i_upper_space, add_special_tokens=False))].append(sim_words_i_upper_space)
                if len(token_i_lower_space)<3:
                    continue
                if token_id_i_lower_space in self.token_id_dict[idx]:
                    continue
                self.token_id_dict[idx].append(token_id_i_lower_space)
                self.token_dict[idx].append(token_i_lower_space)
                self.label_list.append(idx)
                self.data_list.append(self.model_token_embed[token_id_i_lower_space: token_id_i_lower_space + 1])

            print(self.token_dict[idx])
            print(self.token_id_dict[idx])
            self.dataset_size += len(self.token_dict[idx])
            idx += 1

        print(word_token_num_dict)
        dataset = torch.cat(self.data_list, dim=0)
        self.label_list = torch.tensor(self.label_list)
        return dataset



class Sim_Word_Loader(Dataset):

    def __init__(self, data):
        self.dataset = data.dataset
        self.label_list = data.label_list
        self.length = len(data)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        data_i = self.dataset[index]
        label_i = self.label_list[index]

        return data_i, torch.tensor(label_i)




