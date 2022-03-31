# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
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

import argparse
from collections import defaultdict
import os
import random
import time
import math
import json
from functools import partial
import codecs
import zipfile
import re
from tqdm import tqdm
import sys
from glob import glob
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable, backward
import torch.backends.cudnn as cudnn

from torch.utils.data import Dataset
import data_manager
from data_loader import Data_Loader, Data_Loader_unlabel, Data_Loader_unlabel_pet
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, AutoTokenizer
from transformers import AdamW, AutoConfig, get_linear_schedule_with_warmup
from transformers import BertConfig, BertTokenizer
from loss import CrossEntropy_acc, distillation_loss
from torch.optim import Adam
import models
from utils import Logger, set_random_seed, AverageMeter, get_pre_info, get_specific_label_info
from torch.nn.parallel import DistributedDataParallel


# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--eval_only", action='store_true', default=False, help="do predict")
parser.add_argument("--train_unlabel", action='store_true', default=False, help="do predict")

parser.add_argument("--dataset", default="YahooAnswers", type=str, required=False, help="Path to data.")
parser.add_argument("--no_pattern", action='store_true', default=False, help="do predict")
parser.add_argument("--pattern_id", default=0, type=int, required=False, help="PVP pattern.")
parser.add_argument("--num_class", default=10, type=int, required=False, help="classification category number.")
parser.add_argument("--trainset", default="/xxx/dataset/yahoo_answers_csv/train_select_10.csv", type=str, required=False, help="trainset.")
parser.add_argument("--testset", default="/xxx/dataset/yahoo_answers_csv/test.csv", type=str, required=False, help="trainset.")
parser.add_argument("--unlabelset", default="/xxx/dataset/yahoo_answers_csv/unlabeled_select_10.csv", type=str, required=False, help="trainset.")
parser.add_argument("--pretrainset", default="/xxx/dataset/yahoo_answers_csv/train_select_10.csv", type=str, required=False, help="trainset.")
parser.add_argument("--ref_unlabel_pre_info_sort_dict", default="", type=str, required=False, help="trainset.")
parser.add_argument("--unlabel_pre_info_sort_dict", default="", type=str, required=False, help="trainset.")
parser.add_argument("--max_seq_length", default=256, type=int, help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument('--unlable_loader', default='Data_Loader_unlabel_pet', type=str, help="unlabel dataset_loader")
parser.add_argument("--category_balance", action='store_true', default=False, help="")
parser.add_argument("--unlabel_high_data_num", default=1000, type=int, required=False, help="Data_Loader_unlabel_low_high param")
parser.add_argument("--unlabel_low_data_num", default=1000, type=int, required=False, help="Data_Loader_unlabel_low_high param.")

parser.add_argument("--model", default='transformer_pre_fc_tanh_init', type=str, help="model.")
parser.add_argument("--base_model", default='roberta', type=str, help="Pretrain model.")
parser.add_argument("--base_model_type", default='mlm', type=str, help="Pretrain model.")
parser.add_argument("--pretrain_model_path", default='/xxx/pretrain_model/roberta_large', type=str, required=False, help="Path to data.")

parser.add_argument("--bool_attention_mask", action='store_true', default=True, help="do predict")
parser.add_argument('--loss_func', default='CrossEntropy_acc', type=str, help="loss function")
parser.add_argument("--lm_training", action='store_true', default=False, help="do kl loss")
parser.add_argument("--coef_loss_lm", default=0.0001, type=float, help="lm train loss coef")
parser.add_argument("--kl_loss", action='store_true', default=False, help="do kl loss")
parser.add_argument("--temperature", default=2.0, type=float, help="kl loss coef")
parser.add_argument("--ce_kl_loss", action='store_true', default=True, help="ce_loss and kl_loss")
parser.add_argument("--ce_loss_coef", default=0.1, type=float, help="kl loss coef")

parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--test_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
parser.add_argument("--warmup_ratio", default=0.00, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--pretrain_heads", action='store_true', default=False, help="do predict")
parser.add_argument("--num_pretrain_epochs", default=0, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--eval_epochs", default=1, type=float, help="Total number of training epochs to perform.")
parser.add_argument("--seed", default=777, type=int, help="random seed for initialization")
parser.add_argument('--print_steps', type=int, default=400, help="print frequency")

parser.add_argument("--use_cpu", action='store_true', default=False, help="only use cpu")
parser.add_argument("--gpu_id", default='2', type=str, help="gpu devices for using.")
parser.add_argument("--num_workers", default=32, type=int, help="number of gpus to use, 0 for cpu.")

parser.add_argument("--load_trained_model", action='store_true', default=False, help="load trained model.")
parser.add_argument("--model_dict_test", default='', type=str, help="save_model.")
parser.add_argument("--class_state_dict", default='', type=str, help="save_model.")
parser.add_argument("--output_dir", default="", type=str, required=False, help="The output directory.")
parser.add_argument("--tag", default='100', type=str, help="tag for save.")

args = parser.parse_args()
# yapf: enable


def main():
    args.coef_loss_lm = 1.0 if args.model == 'transformer_pre_fc_tanh_init' else 0.0001
    args.n_gpu = len(args.gpu_id.split(',')) + 1
    set_random_seed(args.seed, args.n_gpu)

    args.output_dir = os.path.join(args.output_dir, args.model, args.dataset, args.tag)
    print(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # if args.train_unlabel:
    #     args.kl_loss = True

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if not args.eval_only:
        sys.stdout = Logger(os.path.join(args.output_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(os.path.join(args.output_dir, 'log_evaluate.txt'))

    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_id))
        cudnn.benchmark = True
    else:
        print("Currently using CPU (GPU is highly recommended)")


    class_state_dict = {'YahooAnswers':'xxx.pth',
                        'YelpFull':'',
                        'AGNews':'',
                        'MNLI':'',}


    args.class_state_dict = class_state_dict[args.dataset]
    args.model = args.model.lower()
    model = models.init_model(args.model, args)
    config = model.model_config
    tokenizer = model.tokenizer
    vocab_size = config.vocab_size

    if args.load_trained_model:
        print('======loading trained model======')
        if os.path.exists(args.model_dict_test):
            model_state_dict = torch.load(args.model_dict_test)
            try:
                model.load_state_dict(model_state_dict, strict=True)
            except:
                print('Error occured in state_dict loading!!!')
        else:
            print('No model found in %s!!!'%args.model_dict_test)
            sys.exit(0)


    # Loads dataset.
    print('======Dataset loading======')
    pin_memory = True if use_gpu else False
    if (not args.eval_only):
        train_dataset = data_manager.init_dataset(name=args.dataset, data_dir=args.trainset, mode='train', pattern_id=args.pattern_id, 
            tokenizer=tokenizer, config=config, no_pattern=args.no_pattern)
        print('======Dataset loading Finihsed!!! trainset:{} '.format(len(train_dataset)))
        trainloader = DataLoader(
            Data_Loader(train_dataset, tokenizer, args.max_seq_length),
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            pin_memory=pin_memory, 
            drop_last=True,
            shuffle=True,
        )

        lm_train_dataset = data_manager.init_dataset(name=args.dataset, data_dir=args.unlabelset, mode='train', pattern_id=args.pattern_id, 
            tokenizer=tokenizer, config=config, no_pattern=args.no_pattern)
        print('======Dataset loading Finihsed!!! lmtrainset:{} '.format(len(lm_train_dataset)))
        lmtrainloader = DataLoader(
            Data_Loader(lm_train_dataset, tokenizer, args.max_seq_length),
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            pin_memory=pin_memory, 
            drop_last=True,
            shuffle=True,
        )
        lmtrain_iter = lmtrainloader.__iter__()

    if (not args.eval_only) and args.train_unlabel:
        # acc, test_loss, unlabel_pre_info_sort_dict = evaluate(model, criterion, testloader, use_gpu=use_gpu, args=args, save_name='pre_info_sort_dict_test_epoch_%i.pth'%(epoch))
        unlabel_info_sort_dict = torch.load(args.unlabel_pre_info_sort_dict)
        ref_unlabel_info_sort_dict = torch.load(args.ref_unlabel_pre_info_sort_dict)
        unlabel_dataset = data_manager.init_dataset(name=args.dataset, data_dir=args.unlabelset, mode='unlabeled', pattern_id=args.pattern_id, 
            tokenizer=tokenizer, config=config, no_pattern=args.no_pattern)
        print('======Dataset loading Finihsed!!! unlabelset: {}'.format(len(unlabel_dataset)))
        unlabel_loader = eval(args.unlable_loader)(train_dataset, unlabel_dataset, unlabel_info_sort_dict, ref_unlabel_info_sort_dict, model=model, 
            tokenizer=tokenizer, args=args)
        trainloader = DataLoader(
            unlabel_loader,
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            pin_memory=pin_memory, 
            drop_last=True,
            shuffle=True,
        )



    test_dataset = data_manager.init_dataset(name=args.dataset, data_dir=args.testset, mode='eval', pattern_id=args.pattern_id, 
        tokenizer=tokenizer, config=config, no_pattern=args.no_pattern)
    args.num_class = len(test_dataset.labels)
    verbalizer_ids = test_dataset.verbalizer_ids
    print('======Dataset loading Finihsed!!! testset: {}'.format(len(test_dataset)))

    testloader = DataLoader(
        Data_Loader(test_dataset, tokenizer, args.max_seq_length),
        batch_size=args.test_batch_size, 
        num_workers=args.num_workers,
        pin_memory=pin_memory, 
        drop_last=False,
    )

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    # loss function
    criterion = eval(args.loss_func)(args=args)

    if args.eval_only:
        # Does predictions.
        print("\n=====start predicting=====")
        acc, test_loss, _ = evaluate(model, criterion, testloader, use_gpu, args, save_name='pre_info_sort_dict_model_test.pth')
        print("acc: %.4f\t loss: %.2f\t" %(100 * acc, test_loss))
        print("=====predicting complete=====")
        sys.exit(0)


    # pretrain heads of model
    if args.pretrain_heads:

        pretrain_model(model, lmtrainloader, lmtrain_iter, tokenizer, config, pin_memory, criterion, use_gpu, args)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)

    # Defines learning rate strategy.
    steps_by_epoch = len(trainloader) 
    num_training_steps = steps_by_epoch * args.num_train_epochs
    warmup_steps = args.warmup_ratio * num_training_steps

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)


    torch.save(args, os.path.join(args.output_dir, "training_args.pth"))
    # Starts training.
    best_precision = 0.0
    start_epoch = 0

    for epoch in range(start_epoch, args.num_train_epochs):
        print("\n=====start training of %d epochs=====" % epoch)
        epoch_time = time.time()
            
        train(model, optimizer, lr_scheduler, criterion, trainloader, lmtrainloader, lmtrain_iter, tokenizer, epoch, args, use_gpu)

        if ((epoch + 1) % args.eval_epochs == 0 or (epoch == args.num_train_epochs -1)):
            print("\n=====start evaluating of %d epochs=====" %(epoch + 1))

            acc, test_loss = evaluate(model, criterion, testloader, use_gpu=use_gpu, args=args, save_name=None)

            print("acc: %.4f\t test_loss: %.2f\t" %(100 * acc, test_loss))

            model_state_dict = model.module.state_dict() if args.n_gpu > 0 else model.state_dict()
            os.system('rm -rf %s/model*.pth'%args.output_dir)
            print("Saving model checkpoint to %s", args.output_dir)


            if acc > best_precision:
                # Take care of distributed/parallel training
                best_precision = acc
                model_state_dict = model.module.state_dict() if args.n_gpu > 0 else model.state_dict()
                output_dir = os.path.join(args.output_dir, 'model_best')
                os.makedirs(output_dir, exist_ok=True)
                os.system('rm -rf %s/model*.pth'%output_dir)
                torch.save(model_state_dict, os.path.join(output_dir, 'model_best_epoch_%i_acc_%.2f.pth'%(epoch + 1, 100*best_precision)))
                # tokenizer.save_pretrained(output_dir)
                print("Saving model checkpoint to %s"%output_dir)

        epoch_time = time.time() - epoch_time
        print("epoch time footprint: %d hour %d min %d sec" %
              (epoch_time // 3600, (epoch_time % 3600) // 60, epoch_time % 60))

    
    print("Best precision_all: %.2f\t" %(100*best_precision))
    unlabel_dataset = data_manager.init_dataset(name=args.dataset, data_dir=args.unlabelset, mode='eval', pattern_id=args.pattern_id, 
        tokenizer=tokenizer, config=config, no_pattern=args.no_pattern)
    print('======Dataset loading Finihsed!!! testset: {}'.format(len(unlabel_dataset)))

    unlabelloader = DataLoader(
        Data_Loader(unlabel_dataset, tokenizer, args.max_seq_length),
        batch_size=args.test_batch_size, 
        num_workers=args.num_workers,
        pin_memory=pin_memory, 
        drop_last=False,
    )

    best_model_dir = glob(os.path.join(args.output_dir, 'model_best/*.pth'))[0]
    print(best_model_dir)
    best_model_state_dict = torch.load(best_model_dir)
    model.module.load_state_dict(best_model_state_dict, strict=True)
    print("\n=====start predicting=====")
    acc, test_loss, _ = evaluate(model, criterion, unlabelloader, use_gpu, args, save_name='pre_info_sort_dict_model_test.pth')
    print("acc: %.4f\t loss: %.2f\t" %(100 * acc, test_loss))
    print("=====predicting complete=====")

    
    torch.save(args, os.path.join(args.output_dir, "training_args.pth"))



def mask_tokens(input_ids, tokenizer):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = input_ids.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability 0.15)
    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                            labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()

    # if a version of transformers < 2.4.0 is used, -1 is the expected value for indices to ignore
    # if [int(v) for v in transformers_version.split('.')][:3] >= [2, 4, 0]:
    #     ignore_value = -100
    # else:
    #     ignore_value = -1
    ignore_value = -100

    labels[~masked_indices] = ignore_value  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    # input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    input_ids[indices_replaced] = tokenizer.mask_token_id# '[MASK]'

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return input_ids, labels


def train(model, optimizer, lr_scheduler, criterion, trainloader, lmtrainloader, lmtrain_iter, tokenizer, epoch, args, use_gpu):
    
    model.train()
    step_i = 0
    loss_item = AverageMeter()
    loss_lm_item = AverageMeter()
    loss_epoch_item = AverageMeter()
    acc_all_item = AverageMeter()
    acc_item = AverageMeter()
    acc_lm_item = AverageMeter()
    acc_lm_all_item = AverageMeter()
    step_time = time.time()
    steps_by_epoch = len(trainloader)
    for step, batch in enumerate(trainloader):

        if use_gpu:
            batch = {k: t.cuda() for k, t in batch.items()}

        mlm_labels = batch['mlm_labels']
        labels = batch['label']
        if args.kl_loss:
            probs_all_labels = batch['probs_all']
        labels_gt = batch['label_gt']
        bool_unlabel = batch['bool_unlabel']


        model_input = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
        x_mlm, feats_mask_batch, x_encoder_mask_pre = model(model_input, mlm_labels)

        acc, loss = criterion(x_encoder_mask_pre, labels, labels_gt)
        loss_all = loss

        if args.kl_loss:
            if args.ce_kl_loss:
                if torch.sum(bool_unlabel) > 0 and torch.sum(~bool_unlabel) > 0:
                    kl_loss = distillation_loss(x_encoder_mask_pre[bool_unlabel], probs_all_labels[bool_unlabel], temperature=args.temperature)
                    acc, ce_loss = criterion(x_encoder_mask_pre[~bool_unlabel], labels[~bool_unlabel], labels_gt[~bool_unlabel])
                    loss_all = kl_loss + args.ce_loss_coef * ce_loss
                elif torch.sum(bool_unlabel) > 0:
                    kl_loss = distillation_loss(x_encoder_mask_pre[bool_unlabel], probs_all_labels[bool_unlabel], temperature=args.temperature)
                    loss_all = kl_loss
                else:
                    acc, ce_loss = criterion(x_encoder_mask_pre[~bool_unlabel], labels[~bool_unlabel], labels_gt[~bool_unlabel])
                    loss_all = args.ce_loss_coef * ce_loss
            else:
                kl_loss = distillation_loss(x_encoder_mask_pre, probs_all_labels, temperature=args.temperature)
                loss_all = kl_loss

        if loss_all != None:
            loss_item.update(loss_all.cpu().item(), labels.shape[0])
            loss_epoch_item.update(loss_all.cpu().item(), labels.shape[0])
        acc_item.update(acc.cpu().item(), labels.shape[0])
        acc_all_item.update(acc.cpu().item(), labels.shape[0])

        if loss_all != None:
            loss_all.backward()
        step_i += 1

        # lm_train
        if args.lm_training:
            try:
                lm_batch = lmtrain_iter.__next__()
            except:
                lm_batch = lmtrainloader.__next__()
            if use_gpu:
                lm_batch = {k: t.cuda() for k, t in lm_batch.items()}
            lm_batch['input_ids'], aux_lm_labels= mask_tokens(lm_batch['input_ids'].cpu(), tokenizer)
            if use_gpu:
                lm_batch['input_ids'] = lm_batch['input_ids'].cuda()
                aux_lm_labels = aux_lm_labels.cuda()

            if args.base_model != 'bart':
                lm_model_input = {'input_ids': lm_batch['input_ids'], 'attention_mask': lm_batch['attention_mask'], 'token_type_ids': lm_batch['token_type_ids']}
                lm_x_mlm, _, _ = model(lm_model_input, lm_batch['mlm_labels'])
            else:
                lm_model_input = {'input_ids': lm_batch['input_ids'], 'attention_mask': lm_batch['attention_mask']}
                lm_x_mlm, _, _ = model(model_input, lm_batch['mlm_labels'])

            lm_x_mlm_reshape = lm_x_mlm.view(lm_x_mlm.shape[0] * lm_x_mlm.shape[1], -1)
            aux_lm_labels_reshape = aux_lm_labels.view(-1)
            loss_lm = F.cross_entropy(lm_x_mlm_reshape, aux_lm_labels_reshape, ignore_index=-100)
            model_pre = torch.argmax(lm_x_mlm_reshape, dim=1)
            if aux_lm_labels_reshape != None:
                acc_lm = torch.sum(model_pre == aux_lm_labels_reshape) / torch.where(aux_lm_labels_reshape!=-100)[0].shape[0]
            loss_all += args.coef_loss_lm * loss_lm
            if loss_lm != None:
                acc_lm_item.update(acc_lm.cpu().item(), 1)
                loss_lm_item.update(loss_lm.cpu().item(), 1)
                acc_lm_all_item.update(acc_lm.cpu().item(), 1)
                loss_lm.backward()
                # optimizer.step()
                # optimizer.zero_grad()

        # lm_train

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if step_i % (args.print_steps) == 0:
            if not args.lm_training:
                print("epoch: %d / %d, steps: %d / %d, lr: %f, loss: %.4f, acc: %.4f, speed: %.2f step/s"
                    % (epoch, args.num_train_epochs, step_i, steps_by_epoch, optimizer.param_groups[0]['lr'], 
                    loss_item.avg, acc_item.avg * 100, args.print_steps / (time.time() - step_time)))
            else:
                print("epoch: %d / %d, steps: %d / %d, lr: %f, loss: %.4f, loss_lm: %.4f, acc: %.4f, acc_lm: %.4f, speed: %.2f step/s"
                    % (epoch, args.num_train_epochs, step_i, steps_by_epoch, optimizer.param_groups[0]['lr'], 
                    loss_item.avg, loss_lm_item.avg, acc_item.avg * 100, acc_lm_item.avg * 100, args.print_steps / (time.time() - step_time)))
            step_time = time.time()
            loss_item.reset()
            acc_item.reset()
            loss_lm_item.reset()
            acc_lm_item.reset()

    if not args.lm_training:
        print("====>epoch: %d / %d, lr: %f, loss: %.4f, acc: %.4f"
            % (epoch, args.num_train_epochs, optimizer.param_groups[0]['lr'], loss_epoch_item.avg, acc_all_item.avg * 100))        
    else:
        print("====>epoch: %d / %d, lr: %f, loss: %.4f, acc: %.4f, acc_lm: %.4f"
            % (epoch, args.num_train_epochs, optimizer.param_groups[0]['lr'], loss_epoch_item.avg, acc_all_item.avg * 100, acc_lm_all_item.avg * 100)) 


def evaluate(model, criterion, test_dataloader, use_gpu, args, save_name=None):

    model.eval()
    eval_steps = 0
    feats_mask_all, probs_all, labels_all, idxs_all = None, None, None, None
    loss_item = AverageMeter()
    acc_item = AverageMeter()
    loss_item_all = AverageMeter()
    acc_item_all = AverageMeter()
    step_time = time.time()
    steps_by_epoch = len(test_dataloader)
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        
        if use_gpu:
            batch = {k: t.cuda() for k, t in batch.items()}

        mlm_labels = batch['mlm_labels']
        labels = batch['label']
        idxs = batch['idx']
        
        model_input = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
        x_mlm, feats_mask_batch, x_encoder_mask_pre = model(model_input, mlm_labels)

        acc, loss = criterion(x_encoder_mask_pre, labels)

        loss_item.update(loss.cpu().item(), labels.shape[0])
        acc_item.update(acc.cpu().item(), labels.shape[0])
        loss_item_all.update(loss.cpu().item(), labels.shape[0])
        acc_item_all.update(acc.cpu().item(), labels.shape[0])
        
        if probs_all == None:
            # feats_mask_all = feats_mask_batch.cpu().detach()
            probs_all = x_encoder_mask_pre.cpu().detach()
            labels_all = labels.cpu().detach()
            idxs_all = idxs.cpu().detach()
        else:
            # feats_mask_all = torch.cat([feats_mask_all, feats_mask_batch.cpu().detach()], dim=0)
            probs_all = torch.cat([probs_all, x_encoder_mask_pre.cpu().detach()], dim=0)
            labels_all = torch.cat([labels_all, labels.cpu().detach()], dim=0)
            idxs_all = torch.cat([idxs_all, idxs.cpu().detach()], dim=0)

        eval_steps += 1

        if eval_steps % (args.print_steps) == 0:
            print("steps: %d / %d, loss: %.4f, acc: %.4f, speed: %.2f step/s"
                % (eval_steps, steps_by_epoch, 
                loss_item.avg, acc_item.avg * 100, args.print_steps / (time.time() - step_time)))
            step_time = time.time()
            loss_item.reset()
            acc_item.reset()
    print("===> evaluate loss: %.4f, acc: %.4f"%(loss_item_all.avg, acc_item_all.avg * 100))

    if save_name == None:
        return acc_item_all.avg, loss_item_all.avg

    pre_info_sort_dict = get_pre_info(probs_all, labels_all, idxs_all, args.num_class)
    probs_label = pre_info_sort_dict['probs_pre']
    model_pre = pre_info_sort_dict['model_pre']
    label_gt = pre_info_sort_dict['labels_gt']
    idx_label = pre_info_sort_dict['idx']

    for ratio in np.arange(0.05, 0.4, 0.05):
        print('===>ratio label: %.3f'%ratio)
        pre_num_all = 0
        pre_correct_num_all = 0

        for label_i in range(10):
            pre_label_filter = torch.where(model_pre == label_i)[0]
            pre_label_num = pre_label_filter.shape[0]    
            
            pre_label_index = int(pre_label_num * ratio)
            pre_label_filter_ratio = pre_label_filter[0: pre_label_index+1]

            model_pre_filter = model_pre[pre_label_filter_ratio]
            label_gt_filter = label_gt[pre_label_filter_ratio]
            idx_label_filter = idx_label[pre_label_filter_ratio]
            pre_num = label_gt_filter.shape[0]
            pre_correct_num = torch.sum(label_gt_filter == model_pre_filter)
            pre_num_all += pre_num
            pre_correct_num_all += pre_correct_num
            acc_prob = pre_correct_num / pre_num
            print('     label: %i, acc: %i/%i, %.4f'%(label_i, pre_correct_num, pre_num, acc_prob))

        print('===>ratio: %.2f acc: %i/%i, %.4f'%(ratio, pre_correct_num_all, pre_num_all, pre_correct_num_all/pre_num_all))

    torch.save(pre_info_sort_dict, os.path.join(args.output_dir, save_name))

    return acc_item_all.avg, loss_item_all.avg, pre_info_sort_dict


def pretrain_model(model, lmtrainloader, lmtrain_iter, tokenizer, config, pin_memory, criterion, use_gpu, args):
    bool_lm_training = args.lm_training
    args.lm_training = True
    for name, params in model.named_parameters():
        if not 'classifier' in name:
            params.requires_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    pretrain_dataset = data_manager.init_dataset(name=args.dataset, data_dir=args.pretrainset, mode='train', pattern_id=args.pattern_id, 
        tokenizer=tokenizer, config=config, no_pattern=args.no_pattern)
    print('======Dataset loading Finihsed!!! trainset:{} '.format(len(pretrain_dataset)))
    pretrainloader = DataLoader(
        Data_Loader(pretrain_dataset, tokenizer, args.max_seq_length),
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        pin_memory=pin_memory, 
        drop_last=True,
        shuffle=True,
    )
    optimizer_pretrain = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, eps=1e-8)
    num_training_steps = len(pretrainloader)  * args.num_pretrain_epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer_pretrain, num_warmup_steps=0, num_training_steps=num_training_steps)

    for pretrain_epoch in range(args.num_pretrain_epochs):
        print("\n=====start Pretraining of %d epochs=====" % pretrain_epoch)
        epoch_time = time.time()
        train(model, optimizer_pretrain, lr_scheduler, criterion, pretrainloader, lmtrainloader, lmtrain_iter, tokenizer, pretrain_epoch, args, use_gpu)
        epoch_time = time.time() - epoch_time
        print("epoch time footprint: %d hour %d min %d sec" %
            (epoch_time // 3600, (epoch_time % 3600) // 60, epoch_time % 60))

    args.lm_training = bool_lm_training
    for name, params in model.named_parameters():
        if not 'classifier_norm' in name:
            params.requires_grad = True
    
    del pretrain_dataset, pretrainloader, lr_scheduler, optimizer_pretrain


if __name__ == "__main__":
    main()
    

