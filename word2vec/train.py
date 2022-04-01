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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import Dataset
import argparse

from utils import Logger, AverageMeter
from model import *
from data_manager import Sim_Word, Sim_Word_Loader
from torch.optim import Adam
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaForMaskedLM, BertForMaskedLM, BertTokenizer, BartTokenizer, BartForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument("--eval_only", action='store_true', default=False, help="do predict")
parser.add_argument("--dataset", default='yahoo', type=str, help="do predict")
parser.add_argument("--num_class", default=10, type=int, help="do predict")

parser.add_argument("--base_model", default='roberta', type=str, help="Pretrain model.")
parser.add_argument("--pretrain_model_path", default='/xxx/pretrain_model/roberta_large', type=str, required=False, help="Path to data.")

parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--test_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
parser.add_argument("--warmup_ratio", default=0.00, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

parser.add_argument("--num_train_epochs", default=40, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--eval_epochs", default=4, type=float, help="Total number of training epochs to perform.")
parser.add_argument('--print_steps', type=int, default=5, help="print frequency")

parser.add_argument("--gpu_id", default='0', type=str, help="gpu devices for using.")
parser.add_argument("--num_workers", default=32, type=int, help="number of gpus to use, 0 for cpu.")

parser.add_argument("--load_trained_model", action='store_true', default=False, help="load trained model.")
parser.add_argument("--model_dict_test", default='', type=str, help="save_model.")
parser.add_argument("--output_dir", default="./log/", type=str, required=False, help="The output directory.")

parser.add_argument("--tag", default="test", type=str, required=False, help="The output directory.")
args = parser.parse_args()


def main():
    args.n_gpu = len(args.gpu_id.split(',')) + 1

    args.output_dir = os.path.join(args.output_dir, args.dataset, args.tag)
    print(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_gpu = torch.cuda.is_available()

    sys.stdout = Logger(os.path.join(args.output_dir, 'log_train.txt'))

    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_id))
        cudnn.benchmark = True
    else:
        print("Currently using CPU (GPU is highly recommended)")


    # Loads pretrained model ERNIE
    model = Fc_cls(args.num_class)
    if args.load_trained_model:
        # ['classifier.classifier_fc.weight', 'classifier.classifier_norm_fc.weight', 'classifier.classifier_norm_fc.bias', 'classifier.classifier.weight']
        model_dict_dir = glob(os.path.join(args.model_dict_test, 'model_best/*.pth'))[0]
        print(model_dict_dir)
        model_state_dict = torch.load(model_dict_dir)
        model_state_dict_select = {}
        model_state_dict_select['classifier_fc.weight'] = model_state_dict['classifier.classifier_fc.weight']
        model_state_dict_select['classifier_norm_fc.weight'] = model_state_dict['classifier.classifier_norm_fc.weight']
        model_state_dict_select['classifier_norm_fc.bias'] = model_state_dict['classifier.classifier_norm_fc.bias']
        model_state_dict_select['classifier.weight'] = model_state_dict['classifier.classifier.weight']
        model.load_state_dict(model_state_dict_select, strict=True)

    # Loads dataset.
    print('======Dataset loading======')
    pin_memory = True if use_gpu else False

    # tokenizer = BartTokenizer.from_pretrained('/home/wangzhen/bert-master/pretrain_model/bart_large')
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_model_path)
    token_embedding = RobertaForMaskedLM.from_pretrained(args.pretrain_model_path).lm_head.decoder.state_dict()['weight']
    # token_embedding = BartForConditionalGeneration.from_pretrained('/home/wangzhen/bert-master/pretrain_model/bart_large').lm_head.state_dict()['weight']
    train_dataset = Sim_Word('./word2vec_synonyms/%s.txt'%args.dataset, tokenizer, token_embedding)
    label_list = train_dataset.label_list
    weight_list = torch.zeros(label_list.shape[0])
    for label_i in range(args.num_class):
        weight_list[torch.where(label_list == label_i)] = label_list.shape[0] / torch.where(label_list == label_i)[0].shape[0]
    weight_sampler = WeightedRandomSampler(weights=weight_list, num_samples=label_list.shape[0], replacement=True)
    print('======Dataset loading Finihsed!!! trainset:{} '.format(len(train_dataset)))
    trainloader = DataLoader(
        Sim_Word_Loader(train_dataset),
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        pin_memory=pin_memory, 
        drop_last=True,
        # shuffle=True,
        sampler=weight_sampler,
    )

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    # loss function
    criterion = CrossEntropy_acc()


    if args.eval_only:
        testloader = DataLoader(
            Sim_Word_Loader(train_dataset),
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            pin_memory=pin_memory, 
            drop_last=False,
        )
        acc = eval(model, criterion, testloader, args, use_gpu)
        sys.exit(0)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)

    # Defines learning rate strategy.
    steps_by_epoch = len(trainloader) 
    num_training_steps = steps_by_epoch * args.num_train_epochs
    warmup_steps = args.warmup_ratio * num_training_steps

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)


    best_precision = 0.0
    start_epoch = 1
    for epoch in range(start_epoch, args.num_train_epochs + 1):
        print("\n=====start training of %d epochs=====" % epoch)
        epoch_time = time.time()
            
        acc = train(model, optimizer, lr_scheduler, criterion, trainloader, epoch, args, use_gpu)

        if (epoch % args.eval_epochs) == 0:
            model_state_dict = model.module.state_dict() if args.n_gpu > 0 else model.state_dict()
            torch.save(model_state_dict, os.path.join(args.output_dir, 'model_epoch_%i_acc_%.2f.pth'%(epoch, 100*acc)))


def train(model, optimizer, lr_scheduler, criterion, trainloader, epoch, args, use_gpu):
    
    model.train()
    step_i = 0
    loss_item = AverageMeter()
    loss_epoch_item = AverageMeter()
    acc_all_item = AverageMeter()
    acc_item = AverageMeter()

    step_time = time.time()
    steps_by_epoch = len(trainloader)
    for step, (data, labels) in enumerate(trainloader):

        if use_gpu:
            data = data.cuda()
            labels = labels.cuda()

        model_pre = model(data)

        acc, loss = criterion(model_pre, labels)

        if loss != None:
            loss_item.update(loss.cpu().item(), labels.shape[0])
            loss_epoch_item.update(loss.cpu().item(), labels.shape[0])
        acc_item.update(acc.cpu().item(), labels.shape[0])
        acc_all_item.update(acc.cpu().item(), labels.shape[0])

        if loss != None:
            loss.backward()
        step_i += 1

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if step_i % (args.print_steps) == 0:
            print("epoch: %d / %d, steps: %d / %d, lr: %f, loss: %.4f, acc: %.4f, speed: %.2f step/s"
                % (epoch, args.num_train_epochs, step_i, steps_by_epoch, optimizer.param_groups[0]['lr'], 
                loss_item.avg, acc_item.avg * 100, args.print_steps / (time.time() - step_time)))
            step_time = time.time()
            loss_item.reset()
            acc_item.reset()


    print("====>epoch: %d / %d, lr: %f, loss: %.4f, acc: %.4f"
        % (epoch, args.num_train_epochs, optimizer.param_groups[0]['lr'], loss_epoch_item.avg, acc_all_item.avg * 100))
    
    return acc_all_item.avg

def eval(model, criterion, trainloader, args, use_gpu):
    
    model.eval()
    step_i = 0
    loss_item = AverageMeter()
    loss_epoch_item = AverageMeter()
    acc_all_item = AverageMeter()
    acc_item = AverageMeter()

    step_time = time.time()
    steps_by_epoch = len(trainloader)
    for step, (data, labels) in enumerate(trainloader):

        if use_gpu:
            data = data.cuda()
            labels = labels.cuda()

        model_pre = model(data)

        acc, loss = criterion(model_pre, labels)

        if loss != None:
            loss_item.update(loss.cpu().item(), labels.shape[0])
            loss_epoch_item.update(loss.cpu().item(), labels.shape[0])
        acc_item.update(acc.cpu().item(), labels.shape[0])
        acc_all_item.update(acc.cpu().item(), labels.shape[0])


        if step_i % (args.print_steps) == 0:
            print("steps: %d / %d, loss: %.4f, acc: %.4f, speed: %.2f step/s"
                % (step_i, steps_by_epoch,  
                loss_item.avg, acc_item.avg * 100, args.print_steps / (time.time() - step_time)))
            step_time = time.time()
            loss_item.reset()
            acc_item.reset()


    print("====>loss: %.4f, acc: %.4f"
        % (loss_epoch_item.avg, acc_all_item.avg * 100))
    
    return acc_all_item.avg


class CrossEntropy_acc(nn.Module):

    def __init__(self):
        super(CrossEntropy_acc, self).__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, logits, label):

        loss = self.loss(logits, label)
        model_pre = torch.argmax(logits, dim=1)
        acc = torch.sum(model_pre == label) / label.shape[0]
        return acc, loss


if __name__ == "__main__":
    main()
