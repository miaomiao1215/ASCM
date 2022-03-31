from __future__ import absolute_import

import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from transformers import InputExample, AdamW, PreTrainedTokenizer, BertForMaskedLM, \
    RobertaForMaskedLM, XLMRobertaForMaskedLM, XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer, \
    XLNetLMHeadModel, BertConfig, BertForSequenceClassification, BertTokenizer, RobertaConfig, \
    RobertaForSequenceClassification, RobertaTokenizer, XLMRobertaConfig, XLMRobertaForSequenceClassification, \
    XLMRobertaTokenizer, AlbertForSequenceClassification, AlbertForMaskedLM, AlbertTokenizer, AlbertConfig, \
    BartConfig, BartTokenizer, BartForConditionalGeneration, BartForSequenceClassification, \
    ElectraForMaskedLM, ElectraConfig, ElectraTokenizer, ElectraForSequenceClassification, \
    T5Config, T5Tokenizer, T5ForConditionalGeneration
import time
import os 

__all__ = ['Transformer_PVP', 'Transformer_Pre_fc_tanh_pro', 'Transformer_Pre_fc_tanh_init', 'Transformer_final']


MODEL_CLASSES = {
    'bert': {
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'sequence_classifier': BertForSequenceClassification,
        'mlm': BertForMaskedLM
    },
    'roberta': {
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        'sequence_classifier': RobertaForSequenceClassification,
        'mlm': RobertaForMaskedLM
    },
    'xlm-roberta': {
        'config': XLMRobertaConfig,
        'tokenizer': XLMRobertaTokenizer,
        'sequence_classifier': XLMRobertaForSequenceClassification,
        'mlm': XLMRobertaForMaskedLM
    },
    'xlnet': {
        'config': XLNetConfig,
        'tokenizer': XLNetTokenizer,
        'sequence_classifier': XLNetForSequenceClassification,
        'mlm': XLNetLMHeadModel
    },
    'albert': {
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        'sequence_classifier': AlbertForSequenceClassification,
        'mlm': AlbertForMaskedLM
    },
    'bart': {
        'config': BartConfig,
        'tokenizer': BartTokenizer,
        'sequence_classifier': BartForSequenceClassification,
        'mlm': BartForConditionalGeneration
    },
    'electra': {
        'config': ElectraConfig,
        'tokenizer': ElectraTokenizer,
        'sequence_classifier': ElectraForMaskedLM,
        'mlm': ElectraForMaskedLM
    },
    'T5': {
        'config': T5Config,
        'tokenizer': T5Tokenizer,
        'sequence_classifier': T5ForConditionalGeneration,
        'mlm': T5ForConditionalGeneration
    },
}

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def weights_init_eye(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.eye_(m.weight)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class Transformer_PVP(nn.Module):
    def __init__(self, args, **kwargs):
        super(Transformer_PVP, self).__init__()
        self.args = args
        self.pretrain_model_name = args.base_model.lower()
        config_class = MODEL_CLASSES[self.pretrain_model_name]['config']
        tokenizer_class = MODEL_CLASSES[self.pretrain_model_name]['tokenizer']
        model_class = MODEL_CLASSES[self.pretrain_model_name]['mlm']

        self.model_config = config_class.from_pretrained(args.pretrain_model_path)
        self.tokenizer = tokenizer_class.from_pretrained(args.pretrain_model_path)

        model = model_class.from_pretrained(args.pretrain_model_path)

        if self.pretrain_model_name == 'bert':
            self.model_base = model.bert
            self.model_mlm_cls = model.cls
        elif self.pretrain_model_name == 'roberta':
            self.model_base = model.roberta
            self.model_mlm_cls = model.lm_head
        del model

    def forward(self, x, mlm_labels):
        bool_mlm_labels = mlm_labels > 0
        # x_embed = self.base_model_embeddings(x['input_ids'])
        # x_encoder = self.base_model_encoder(x_embed)
        # x_pool = self.base_model_pooler(x_encoder[0])
        x_encoder = self.model_base(**x)
        x_mlm = self.model_mlm_cls(x_encoder[0])
        x_encoder_mask = x_encoder[0][bool_mlm_labels]
        x_mlm_mask = x_mlm[bool_mlm_labels]
        # x = self.classifier(x_encoder.last_hidden_state)

        return x_mlm, x_encoder_mask, x_mlm_mask


class Transformer_final(nn.Module):
    def __init__(self, args, **kwargs):
        super(Transformer_final, self).__init__()
        self.args = args
        self.pretrain_model_name = args.base_model.lower()
        config_class = MODEL_CLASSES[self.pretrain_model_name]['config']
        tokenizer_class = MODEL_CLASSES[self.pretrain_model_name]['tokenizer']
        model_class = MODEL_CLASSES[self.pretrain_model_name]['sequence_classifier']

        self.model_config = config_class.from_pretrained(args.pretrain_model_path)
        self.tokenizer = tokenizer_class.from_pretrained(args.pretrain_model_path)

        model = model_class.from_pretrained(args.pretrain_model_path)

        if self.pretrain_model_name == 'bert':
            self.model_base = model.bert
            self.model_mlm_cls = model.cls
        elif self.pretrain_model_name == 'roberta':
            self.model_base = model.roberta
            self.model_mlm_cls = model.classifier.dense
        del model

        self.classifier = nn.Linear(self.model_config.hidden_size, args.num_class, bias=True)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, mlm_labels):
        bool_mlm_labels = mlm_labels > 0
        # x_embed = self.base_model_embeddings(x['input_ids'])
        # x_encoder = self.base_model_encoder(x_embed)
        # x_pool = self.base_model_pooler(x_encoder[0])
        x_encoder = self.model_base(**x)
        x_encoder_seq = self.model_mlm_cls(x_encoder[0][:, 0])
        x_pre = self.classifier(x_encoder_seq)

        # x = self.classifier(x_encoder.last_hidden_state)

        return x_encoder_seq, x_encoder_seq, x_pre

class Transformer_Pre_fc_tanh_pro(nn.Module):
    def __init__(self, args, **kwargs):
        super(Transformer_Pre_fc_tanh_pro, self).__init__()
        self.args = args
        self.pretrain_model_name = args.base_model.lower()
        config_class = MODEL_CLASSES[self.pretrain_model_name]['config']
        tokenizer_class = MODEL_CLASSES[self.pretrain_model_name]['tokenizer']
        model_class = MODEL_CLASSES[self.pretrain_model_name][args.base_model_type]

        self.model_config = config_class.from_pretrained(args.pretrain_model_path)
        self.tokenizer = tokenizer_class.from_pretrained(args.pretrain_model_path)

        model = model_class.from_pretrained(args.pretrain_model_path)
        if self.pretrain_model_name == 'bert':
            self.model_base = model.bert
            self.model_mlm_cls = model.cls
        elif self.pretrain_model_name == 'roberta':
            self.model_base = model.roberta
            self.model_dense_mlm = nn.Sequential(model.lm_head.dense, nn.GELU(), model.lm_head.layer_norm)
            self.model_mlm_cls = model.lm_head.decoder
        elif self.pretrain_model_name == 'albert':
            self.model_base = model.albert
            self.model_mlm_cls = model.predictions
        elif self.pretrain_model_name == 'xlnet':
            self.model_base = model.transformer
            self.model_mlm_cls = model.lm_loss
        elif self.pretrain_model_name == 'bart':
            self.model_base = model.model
            self.model_mlm_cls = model.lm_head
        elif self.pretrain_model_name == 'electra':
            self.model_base = model.electra
            self.model_mlm_cls = model.generator_lm_head
        del model

        self.classifier_fc = nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size, bias=False)
        if self.pretrain_model_name == 'bart':
            self.classifier_norm_fc = nn.LayerNorm(self.model_config.hidden_size, eps=1e-12)
        else:
            self.classifier_norm_fc = nn.LayerNorm(self.model_config.hidden_size, eps=self.model_config.layer_norm_eps)
        self.classifier_norm_fc.bias.requires_grad_(False)
        self.classifier_act_fc = nn.Tanh()
        self.classifier = nn.Linear(self.model_config.hidden_size, args.num_class, bias=False)
        # self.transform_act_fn = nn.GELU()
        # self.classifier = nn.Linear(self.configuration.to_dict()['hidden_size'], args.num_classes)
        self.classifier.apply(weights_init_classifier)
        self.classifier_fc.apply(weights_init_classifier)
        self.classifier_norm_fc.apply(weights_init_kaiming)

    def forward(self, x, mlm_labels):
        bool_mlm_labels = mlm_labels > 0
        # x_embed = self.base_model_embeddings(x['input_ids'])
        # x_encoder = self.base_model_encoder(x_embed)
        # x_pool = self.base_model_pooler(x_encoder[0])
        x_encoder = self.model_base(**x)
        x_encoder = self.model_dense_mlm(x_encoder[0])
        x_mlm = self.model_mlm_cls(x_encoder)
        x_encoder_mask = x_encoder[bool_mlm_labels]
        x_mlm_mask = x_mlm[bool_mlm_labels]

        x_encoder_mask_fc = self.classifier_act_fc(self.classifier_norm_fc(self.classifier_fc(x_encoder_mask)))
        x_encoder_mask_pre = self.classifier(x_encoder_mask_fc)

        return x_mlm, x_encoder_mask_fc, x_encoder_mask_pre

class classifier(nn.Module):
    def __init__(self, feat_num, num_class):
        super(classifier, self).__init__()
        self.num_class = num_class
        self.classifier_fc = nn.Linear(feat_num, feat_num, bias=False)
        self.classifier_norm_fc = nn.LayerNorm(feat_num, eps=1e-12)
        self.classifier_norm_fc.bias.requires_grad_(False)
        self.classifier_act_fc = nn.Tanh()
        self.classifier = nn.Linear(feat_num, num_class, bias=False)
        # self.transform_act_fn = nn.GELU()
        # self.classifier = nn.Linear(self.configuration.to_dict()['hidden_size'], args.num_classes)
        self.classifier.apply(weights_init_classifier)
        self.classifier_fc.apply(weights_init_classifier)
    
    def forward(self, x):

        x = self.classifier_act_fc(self.classifier_norm_fc(self.classifier_fc(x)))
        x = self.classifier(x)
        return x


class Transformer_Pre_fc_tanh_init(nn.Module):
    def __init__(self, args, **kwargs):
        super(Transformer_Pre_fc_tanh_init, self).__init__()
        self.args = args
        self.pretrain_model_name = args.base_model.lower()
        config_class = MODEL_CLASSES[self.pretrain_model_name]['config']
        tokenizer_class = MODEL_CLASSES[self.pretrain_model_name]['tokenizer']
        model_class = MODEL_CLASSES[self.pretrain_model_name][args.base_model_type]

        self.model_config = config_class.from_pretrained(args.pretrain_model_path)
        self.tokenizer = tokenizer_class.from_pretrained(args.pretrain_model_path)

        model = model_class.from_pretrained(args.pretrain_model_path)
        if self.pretrain_model_name == 'bert':
            self.model_base = model.bert
            self.model_mlm_cls = model.cls
        elif self.pretrain_model_name == 'roberta':
            self.model_base = model.roberta
            self.model_dense_mlm = nn.Sequential(model.lm_head.dense, nn.GELU(), model.lm_head.layer_norm)
            self.model_mlm_cls = model.lm_head.decoder
        elif self.pretrain_model_name == 'albert':
            self.model_base = model.albert
            self.model_mlm_cls = model.predictions
        elif self.pretrain_model_name == 'xlnet':
            self.model_base = model.transformer
            self.model_mlm_cls = model.lm_loss
        elif self.pretrain_model_name == 'bart':
            self.model_base = model.model
            self.model_mlm_cls = model.lm_head
        elif self.pretrain_model_name == 'electra':
            self.model_base = model.electra
            self.model_mlm_cls = model.generator_lm_head
        del model

        self.classifier = classifier(self.model_config.hidden_size, args.num_class)
        classifier_state_dict = torch.load(args.class_state_dict)
        try:
            self.classifier.load_state_dict(classifier_state_dict, strict=True)
        except:
            print('Error occured in state_dict loading!!!')

    def forward(self, x, mlm_labels):
        bool_mlm_labels = mlm_labels > 0
        # x_embed = self.base_model_embeddings(x['input_ids'])
        # x_encoder = self.base_model_encoder(x_embed)
        # x_pool = self.base_model_pooler(x_encoder[0])
        x_encoder = self.model_base(**x)
        x_encoder = self.model_dense_mlm(x_encoder[0])
        x_mlm = self.model_mlm_cls(x_encoder)
        x_encoder_mask = x_encoder[bool_mlm_labels]
        x_mlm_mask = x_mlm[bool_mlm_labels]

        x_encoder_mask_pre = self.classifier(x_encoder_mask)

        return x_mlm, x_encoder_mask, x_encoder_mask_pre




