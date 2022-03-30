import os
import random
import argparse
from typing import Pattern
import copy
import numpy as np
from glob import glob
# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="YahooAnswers", type=str, required=False, help="Path to data.")
parser.add_argument("--work_dir", default="10_lm", type=str, required=False, help="trainset.")

parser.add_argument("--train_num", default=10, type=int, required=False, help="samples number per category.")

parser.add_argument("--base_model", default='roberta', type=str, help="Pretrain model.")
parser.add_argument("--pretrain_model_path", default='/xxx/bert-master/pretrain_model/roberta_large', type=str, required=False, help="Path to data.")

parser.add_argument('--unlable_loader', default='Data_Loader_unlabel_pet', type=str, help="unlabel dataset_loader")#Data_Loader_unlabel_pet_add_low_v2

parser.add_argument("--kl_loss", action='store_true', default=False, help="do kl loss")
parser.add_argument("--coef_loss_lm", default=0.0001, type=float, help="lm train loss coef")
parser.add_argument("--gpu_id", default='2', type=str, help="gpu devices for using.")

parser.add_argument("--output_dir", default="./log/1015/", type=str, required=False, help="The output directory.")
parser.add_argument("--tag", default='ipet', type=str, help="tag for save.")
args = parser.parse_args()

dataset_dict = {
    'YahooAnswers': {
        'trainset': '/xxx/dataset/yahoo_answers_csv/train_select_%i.csv'%args.train_num,
        'testset': '/xxx/dataset/yahoo_answers_csv/test.csv',
        'pretrainset': '/xxx/dataset/yahoo_answers_csv/train_select_%i.csv'%args.train_num,
        'unlabelset': '/xxx/dataset/yahoo_answers_csv/unlabeled_select_%i.csv'%args.train_num,
        'pattern_num': 6,
        'num_class': 10,
    },
    'YelpFull': {
        'trainset': '/xxx/dataset/yelp_review_full_csv/train_select_%i.csv'%args.train_num,
        'testset': '/xxx/dataset/yelp_review_full_csv/test.csv',
        'pretrainset': '/xxx/dataset/yelp_review_full_csv/train_select_%i.csv'%args.train_num,
        'unlabelset': '/xxx/dataset/yelp_review_full_csv/unlabeled_select_%i.csv'%args.train_num,
        'pattern_num': 4,
        'num_class': 5,
    },
    'AGNews': {
        'trainset': '/xxx/dataset/ag_news_csv/train_select_%i.csv'%args.train_num,
        'testset': '/xxx/dataset/ag_news_csv/test.csv',
        'pretrainset': '/xxx/dataset/ag_news_csv/train_select_%i.csv'%args.train_num,
        'unlabelset': '/xxx/dataset/ag_news_csv/unlabeled_select_%i.csv'%args.train_num,
        'pattern_num': 6,
        'num_class': 4,
    },
    'MNLI': {
        'trainset': '/xxx/dataset/multinli/train_select_%i.csv'%args.train_num,
        'testset': '/xxx/dataset/multinli/test_matched.csv',
        'pretrainset': '/xxx/dataset/multinli/train_select_%i.csv'%args.train_num,
        'unlabelset': '/xxx/dataset/multinli/unlabeled_select_%i.csv'%args.train_num,
        'pattern_num': 2,
        'num_class': 3,
    },
}

train_epoch_dict = {
    10: {
        'train_epoch': 500, 'eval_epochs': 200, 'pretrain_epoch': 0, 'ipet_round': 3, 'retrain_step': 300,       
    },
    50: {
        'train_epoch': 100, 'eval_epochs': 40, 'pretrain_epoch': 0, 'ipet_round': 2, 'retrain_step': 300,      
    },
    100: {
        'train_epoch': 50, 'eval_epochs': 20, 'pretrain_epoch': 0, 'ipet_round': 2, 'retrain_step': 300,       
    },
    1000: {
        'train_epoch': 5, 'eval_epochs': 2, 'pretrain_epoch': 0, 'ipet_round': 0, 'retrain_step': 300,       
    },
}

batch_size = 8 if args.train_num == 10 else 16
pattern_list = list(range(dataset_dict[args.dataset]['pattern_num']))
pretrain_command_list = []
for pattern_i in range(dataset_dict[args.dataset]['pattern_num']):
    eval_epochs = train_epoch_dict[args.train_num]['eval_epochs']
    if args.dataset == "YahooAnswers":
        eval_epochs = int(np.ceil(eval_epochs * 1.25))
    print('===>start pretrain model with labled data')
    command_base = 'python train_lm.py --lm_training --pretrain_heads'
    command_base += ' --dataset %s'%args.dataset
    command_base += ' --batch_size %i'%batch_size
    command_base += ' --pattern_id %i'%pattern_i
    command_base += ' --num_class %i'%dataset_dict[args.dataset]['num_class']
    command_base += ' --trainset %s'%dataset_dict[args.dataset]['trainset']
    command_base += ' --testset %s'%dataset_dict[args.dataset]['testset']
    command_base += ' --pretrainset %s'%dataset_dict[args.dataset]['pretrainset']
    command_base += ' --unlabelset %s'%dataset_dict[args.dataset]['unlabelset']
    command_base += ' --base_model %s'%args.base_model
    command_base += ' --pretrain_model_path %s'%args.pretrain_model_path
    command_base += ' --coef_loss_lm %f'%args.coef_loss_lm
    command_base += ' --num_train_epochs %i'%train_epoch_dict[args.train_num]['train_epoch']
    command_base += ' --num_pretrain_epochs %i'%train_epoch_dict[args.train_num]['pretrain_epoch']
    command_base += ' --eval_epochs %i'%eval_epochs
    command_base += ' --tag %s/%s_pattern_%s'%(args.work_dir, os.path.basename(args.pretrain_model_path), pattern_i)
    command_base += ' --gpu_id %s'%args.gpu_id
    print('        pattern: %i, command: %s'%(pattern_i, command_base))
    pretrain_command_list.append(command_base)


for pretrain_command_i in pretrain_command_list:
    print(pretrain_command_i)
    try:
        os.system(pretrain_command_i)
    except:
        print('=========something wrong!!!===========')


print('\n')
print('\n')
print('===>start pretrain model with unlabled data')
for retrain_round_i in range(1, train_epoch_dict[args.train_num]['ipet_round'] + 1):
# for retrain_round_i in range(1, 2):
    retrain_command_list = []
    print('retrain round: %i'%retrain_round_i)
    for pattern_i in range(dataset_dict[args.dataset]['pattern_num']):

        ref_pattern_list = copy.deepcopy(pattern_list)
        ref_pattern_list.remove(pattern_i)
        ref_pattern_select = random.choice(ref_pattern_list)

        unlabel_low_data_num = 0
        unlabel_high_data_num = int(np.ceil(args.train_num * (5 ** retrain_round_i) / dataset_dict[args.dataset]['num_class']))
        unlabel_data_num = unlabel_high_data_num * dataset_dict[args.dataset]['num_class'] + args.train_num
        steps_epoch = unlabel_data_num // batch_size
        num_train_epoch_retrain = int(np.ceil(train_epoch_dict[args.train_num]['retrain_step'] / steps_epoch))
        eval_epoch_retrain = num_train_epoch_retrain // 3 + 1
        # eval_epoch_retrain = 2 if args.dataset == 'YahooAnswers' else 1

        if retrain_round_i == 1:
            unlabel_pre_info_sort_dict = './xxx/transformer_pre_fc_tanh_init/%s/%s/%s_pattern_%s/pre_info_sort_dict_model_test.pth'\
                %(args.dataset, args.work_dir, os.path.basename(args.pretrain_model_path), pattern_i)
            ref_unlabel_pre_info_sort_dict = './xxx/transformer_pre_fc_tanh_init/%s/%s/%s_pattern_%s/pre_info_sort_dict_model_test.pth'\
                %(args.dataset, args.work_dir, os.path.basename(args.pretrain_model_path), ref_pattern_select)

            model_dict = glob('./xxx/transformer_pre_fc_tanh_init/%s/%s/%s_pattern_%s/model_best/*.pth'\
                %(args.dataset, args.work_dir, os.path.basename(args.pretrain_model_path), pattern_i))[0]
        else:
            unlabel_pre_info_sort_dict = glob('./xxx/transformer_pre_fc_tanh_init/%s/%s/%s_%s_pattern_%s_ref_*_round_%i/pre_info_sort_dict_model_test.pth'\
                %(args.dataset, args.work_dir, args.tag, os.path.basename(args.pretrain_model_path), pattern_i, retrain_round_i-1))[0]
            ref_unlabel_pre_info_sort_dict = glob('./xxx/transformer_pre_fc_tanh_init/%s/%s/%s_%s_pattern_%s_ref_*_round_%i/pre_info_sort_dict_model_test.pth'\
                %(args.dataset, args.work_dir, args.tag, os.path.basename(args.pretrain_model_path), ref_pattern_select, retrain_round_i-1))[0]

            model_dict = glob('./xxx/transformer_pre_fc_tanh_init/%s/%s/%s_%s_pattern_%s_ref_*_round_%i/model_best/*.pth'\
                %(args.dataset, args.work_dir, args.tag, os.path.basename(args.pretrain_model_path), pattern_i, retrain_round_i-1))[0]


        command_base = 'python train_lm.py --train_unlabel --category_balance --print_steps 100 --load_trained_model --lm_training '
        command_base += ' --dataset %s'%args.dataset
        command_base += ' --batch_size %i'%batch_size
        command_base += ' --pattern_id %i'%pattern_i
        command_base += ' --num_class %i'%dataset_dict[args.dataset]['num_class']
        command_base += ' --trainset %s'%dataset_dict[args.dataset]['trainset']
        command_base += ' --testset %s'%dataset_dict[args.dataset]['testset']
        command_base += ' --unlabelset %s'%dataset_dict[args.dataset]['unlabelset']
        command_base += ' --unlabel_pre_info_sort_dict %s'%unlabel_pre_info_sort_dict
        command_base += ' --ref_unlabel_pre_info_sort_dict %s'%ref_unlabel_pre_info_sort_dict
        command_base += ' --unlable_loader %s'%args.unlable_loader
        command_base += ' --unlabel_high_data_num %s'%unlabel_high_data_num
        # command_base += ' --unlabel_low_data_num %s'%unlabel_low_data_num
        command_base += ' --base_model %s'%args.base_model
        command_base += ' --pretrain_model_path %s'%args.pretrain_model_path
        command_base += ' --model_dict_test %s'%model_dict
        command_base += ' --coef_loss_lm %f'%args.coef_loss_lm
        command_base += ' --num_train_epochs %i'%num_train_epoch_retrain
        command_base += ' --eval_epochs %i'%eval_epoch_retrain
        command_base += ' --tag %s/%s_%s_pattern_%s_ref_%i_round_%i'\
            %(args.work_dir, args.tag, os.path.basename(args.pretrain_model_path), pattern_i, ref_pattern_select, retrain_round_i)

        if args.kl_loss:
            command_base += ' --kl_loss '
        command_base += ' --gpu_id %s'%args.gpu_id
        print('        pattern: %i, command: %s'%(pattern_i, command_base))
        retrain_command_list.append(command_base)

    for retrain_command_i in retrain_command_list:
        print(retrain_command_i)
        try:
            os.system(retrain_command_i)
        except:
            print('=========something wrong!!!===========')



command_base = 'python train_final.py  '
command_base += ' --dataset %s'%args.dataset
command_base += ' --batch_size %i'%batch_size
command_base += ' --num_class %i'%dataset_dict[args.dataset]['num_class']
command_base += ' --trainset %s'%dataset_dict[args.dataset]['trainset']
command_base += ' --testset %s'%dataset_dict[args.dataset]['testset']
command_base += ' --unlabelset %s'%dataset_dict[args.dataset]['unlabelset']
command_base += ' --pretrain_model_path %s'%args.pretrain_model_path
command_base += ' --no_pattern --kl_loss '
command_base += ' --gpu_id %s'%args.gpu_id
command_base += ' --retrain_rounds %i'%train_epoch_dict[args.train_num]['ipet_round']
command_base += ' --output_dir %s'%args.output_dir
command_base += ' --unlabel_tag %s'%args.tag
command_base += ' --tag %s/%s_final '%(args.work_dir, args.tag)
print(command_base)
os.system(command_base)

