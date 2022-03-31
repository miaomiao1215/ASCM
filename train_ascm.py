import os
import random
import argparse
from typing import Pattern
import copy
import numpy as np
from glob import glob
from utils import get_model_train_sort, model_acc_sort
# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="YahooAnswers", type=str, required=False, help="Path to data.")
parser.add_argument("--work_dir", default="10_lm", type=str, required=False, help="trainset.")

parser.add_argument("--train_num", default=10, type=int, required=False, help="samples number per category.")

parser.add_argument("--model", default='transformer_pre_fc_tanh_init', type=str, help="Pretrain model.")
parser.add_argument("--base_model", default='roberta', type=str, help="Pretrain model.")
parser.add_argument("--pretrain_model_path", default='/xxx/bert-master/pretrain_model/roberta_large', type=str, required=False, help="Path to data.")

parser.add_argument('--unlable_loader', default='Data_Loader_unlabel_pet', type=str, help="unlabel dataset_loader")#Data_Loader_unlabel_pet_add_low_v2

parser.add_argument("--kl_loss", action='store_true', default=False, help="do kl loss")
parser.add_argument("--coef_loss_lm", default=0.0001, type=float, help="lm train loss coef")
parser.add_argument("--gpu_id", default='2', type=str, help="gpu devices for using.")

parser.add_argument("--output_dir", default="./log/", type=str, required=False, help="The output directory.")
parser.add_argument("--rounds", default=1, type=int, help="")
parser.add_argument("--tag", default='ipet_stair', type=str, help="tag for save.")
args = parser.parse_args()

if args.rounds!=1:
    args.output_dir = "./log/%i/"%args.rounds
    dataset_dict = {
        'YahooAnswers': {
            'trainset': '/xxx/dataset/yahoo_answers_csv/train_select_%i_%i.csv'%(args.train_num, args.rounds),
            'testset': '/xxx/dataset/yahoo_answers_csv/test.csv',
            'pretrainset': '/xxx/dataset/yahoo_answers_csv/train_select_%i_%i.csv'%(args.train_num, args.rounds),
            'unlabelset': '/xxx/dataset/yahoo_answers_csv/unlabeled_select_%i_%i.csv'%(args.train_num, args.rounds),
            'pattern_num': 6,
            'num_class': 10,
        },
        'YelpFull': {
            'trainset': '/xxx/dataset/yelp_review_full_csv/train_select_%i_%i.csv'%(args.train_num, args.rounds),
            'testset': '/xxx/dataset/yelp_review_full_csv/test.csv',
            'pretrainset': '/xxx/dataset/yelp_review_full_csv/train_select_%i_%i.csv'%(args.train_num, args.rounds),
            'unlabelset': '/xxx/dataset/yelp_review_full_csv/unlabeled_select_%i_%i.csv'%(args.train_num, args.rounds),
            'pattern_num': 4,
            'num_class': 5,
        },
        'AGNews': {
            'trainset': '/xxx/dataset/ag_news_csv/train_select_%i_%i.csv'%(args.train_num, args.rounds),
            'testset': '/xxx/dataset/ag_news_csv/test.csv',
            'pretrainset': '/xxx/dataset/ag_news_csv/train_select_%i_%i.csv'%(args.train_num, args.rounds),
            'unlabelset': '/xxx/dataset/ag_news_csv/unlabeled_select_%i_%i.csv'%(args.train_num, args.rounds),
            'pattern_num': 6,
            'num_class': 4,
        },
        'MNLI': {
            'trainset': '/xxx/dataset/multinli/train_select_%i_%i.csv'%(args.train_num, args.rounds),
            'testset': '/xxx/dataset/multinli/test_matched.csv',
            'pretrainset': '/xxx/dataset/multinli/train_select_%i_%i.csv'%(args.train_num, args.rounds),
            'unlabelset': '/xxx/dataset/multinli/unlabeled_select_%i_%i.csv'%(args.train_num, args.rounds),
            'pattern_num': 2,
            'num_class': 3,
        },
    }

else:
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
        'train_epoch': 300, 'eval_epochs': 200, 'pretrain_epoch': 0, 'ipet_round': 3, 'retrain_step': 300,       
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
test_batch_size = 4
pattern_list = list(range(dataset_dict[args.dataset]['pattern_num']))
pretrain_command_list = []
for pattern_i in range(dataset_dict[args.dataset]['pattern_num']):
    eval_epochs = train_epoch_dict[args.train_num]['eval_epochs']
    if args.dataset == "YahooAnswers":
        eval_epochs = int(np.ceil(eval_epochs * 1.25))
    print('===>start pretrain model with labled data')
    command_base = 'python train_lm.py --lm_training '
    command_base += ' --dataset %s'%args.dataset
    command_base += ' --batch_size %i'%batch_size
    command_base += ' --test_batch_size %i'%test_batch_size
    command_base += ' --pattern_id %i'%pattern_i
    command_base += ' --num_class %i'%dataset_dict[args.dataset]['num_class']
    command_base += ' --trainset %s'%dataset_dict[args.dataset]['trainset']
    command_base += ' --testset %s'%dataset_dict[args.dataset]['testset']
    command_base += ' --pretrainset %s'%dataset_dict[args.dataset]['pretrainset']
    command_base += ' --unlabelset %s'%dataset_dict[args.dataset]['unlabelset']
    command_base += ' --model %s'%args.model
    command_base += ' --base_model %s'%args.base_model
    command_base += ' --pretrain_model_path %s'%args.pretrain_model_path
    command_base += ' --coef_loss_lm %f'%args.coef_loss_lm
    command_base += ' --num_train_epochs %i'%train_epoch_dict[args.train_num]['train_epoch']
    command_base += ' --num_pretrain_epochs %i'%train_epoch_dict[args.train_num]['pretrain_epoch']
    command_base += ' --eval_epochs %i'%eval_epochs
    command_base += ' --output_dir %s'%args.output_dir
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


## comments following content if you only want to test ASCM without SL(semi-supervised).

print('\n')
print('\n')
print('===>start pretrain model with unlabled data')
for retrain_round_i in range(1, train_epoch_dict[args.train_num]['ipet_round'] + 1):
# for retrain_round_i in range(1, 2):
    retrain_command_list = []
    print('retrain round: %i'%retrain_round_i)
    retrain_model_dir_sort, retrain_model_acc_sort = get_model_train_sort('%stransformer_pre_fc_tanh_init/%s/%s'%(args.output_dir, args.dataset, args.work_dir), args.tag, retrain_round_i)
    ref_model_dir_sort = copy.deepcopy(retrain_model_dir_sort)
    ref_model_acc_sort = copy.deepcopy(retrain_model_acc_sort)
    for retrain_model_dir_i in retrain_model_dir_sort:
        pattern_i = retrain_model_dir_i[retrain_model_dir_i.index('pattern'): ].split('_')[1]

        ref_model_dir_select = ref_model_dir_sort[-1]
        ref_pattern_select = ref_model_dir_select[ref_model_dir_select.index('pattern'): ].split('_')[1]
        if ref_pattern_select == pattern_i:
            ref_model_dir_select = ref_model_dir_sort[-2]
            ref_pattern_select = ref_model_dir_select[ref_model_dir_select.index('pattern'): ].split('_')[1]

        unlabel_low_data_num = 0
        unlabel_high_data_num = int(np.ceil(args.train_num * (5 ** retrain_round_i) / dataset_dict[args.dataset]['num_class']))
        unlabel_data_num = unlabel_high_data_num * dataset_dict[args.dataset]['num_class'] + args.train_num
        steps_epoch = unlabel_data_num // batch_size
        num_train_epoch_retrain = int(np.ceil(train_epoch_dict[args.train_num]['retrain_step'] / steps_epoch))
        eval_epoch_retrain = num_train_epoch_retrain // 3 + 1
        # eval_epoch_retrain = 2 if args.dataset == 'YahooAnswers' else 1

        unlabel_pre_info_sort_dict = os.path.join(retrain_model_dir_i, 'pre_info_sort_dict_model_test.pth')
        ref_unlabel_pre_info_sort_dict = os.path.join(ref_model_dir_select, 'pre_info_sort_dict_model_test.pth')
        model_dict = glob(os.path.join(retrain_model_dir_i, 'model_best/*.pth'))[0]


        command_base = 'python train_lm.py --train_unlabel --category_balance --print_steps 100 --load_trained_model --lm_training '
        command_base += ' --dataset %s'%args.dataset
        command_base += ' --batch_size %i'%batch_size
        command_base += ' --test_batch_size %i'%test_batch_size
        command_base += ' --pattern_id %s'%pattern_i
        command_base += ' --num_class %i'%dataset_dict[args.dataset]['num_class']
        command_base += ' --trainset %s'%dataset_dict[args.dataset]['trainset']
        command_base += ' --testset %s'%dataset_dict[args.dataset]['testset']
        command_base += ' --unlabelset %s'%dataset_dict[args.dataset]['unlabelset']
        command_base += ' --unlabel_pre_info_sort_dict %s'%unlabel_pre_info_sort_dict
        command_base += ' --ref_unlabel_pre_info_sort_dict %s'%ref_unlabel_pre_info_sort_dict
        if retrain_round_i == 1:
            command_base += ' --unlable_loader Data_Loader_unlabel_pet'
        else:
            command_base += ' --unlable_loader %s'%args.unlable_loader
        command_base += ' --unlabel_high_data_num %s'%unlabel_high_data_num
        # command_base += ' --unlabel_low_data_num %s'%unlabel_low_data_num
        command_base += ' --base_model %s'%args.base_model
        command_base += ' --pretrain_model_path %s'%args.pretrain_model_path
        command_base += ' --model_dict_test %s'%model_dict
        command_base += ' --coef_loss_lm %f'%args.coef_loss_lm
        command_base += ' --num_train_epochs %i'%num_train_epoch_retrain
        command_base += ' --eval_epochs %i'%eval_epoch_retrain
        command_base += ' --output_dir %s'%args.output_dir
        command_base += ' --tag %s/%s_%s_pattern_%s_ref_%s_round_%i'\
            %(args.work_dir, args.tag, os.path.basename(args.pretrain_model_path), pattern_i, ref_pattern_select, retrain_round_i)

        if args.kl_loss:
            command_base += ' --kl_loss '
        command_base += ' --gpu_id %s'%args.gpu_id
        print('        pattern: %s, command: %s'%(pattern_i, command_base))

        try:
            os.system(command_base)
        except:
            print('=========something wrong!!!===========')

        retrain_save_dir_i = '%stransformer_pre_fc_tanh_init/%s/%s/%s_%s_pattern_%s_ref_%s_round_%i'\
            %(args.output_dir,args.dataset, args.work_dir, args.tag, os.path.basename(args.pretrain_model_path), pattern_i, ref_pattern_select, retrain_round_i)
        ref_model_dir_sort.append(retrain_save_dir_i)
        retrain_best_model_i = glob(os.path.join(retrain_save_dir_i, 'model_best/*.pth'))[0]
        model_acc_i = float(os.path.basename(retrain_best_model_i).replace('.pth', '').split('_')[5])
        ref_model_acc_sort.append(model_acc_i)

        ref_model_dir_sort, ref_model_acc_sort = model_acc_sort(ref_model_dir_sort, ref_model_acc_sort)


command_base = 'python train_final.py  '
command_base += ' --dataset %s'%args.dataset
command_base += ' --batch_size %i'%batch_size
command_base += ' --test_batch_size %i'%test_batch_size
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
