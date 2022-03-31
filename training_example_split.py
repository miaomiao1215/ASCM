import os
import csv
from collections import defaultdict
import argparse
import random

def eq_div(N, i):
    """ Equally divide N examples among i buckets. For example, `eq_div(12,3) = [4,4,4]`. """
    return [] if i <= 0 else [N // i + 1] * (N % i) + [N // i] * (i - N % i)

def create_example_csv(path, label_index):
    examples_dict = defaultdict(list)

    with open(path, encoding='utf8') as f_r:
        reader = csv.reader(f_r, delimiter=',')
        for idx, row in enumerate(reader):
            label = row[label_index]
            examples_dict[label].append(row)

    return examples_dict

def example_split(examples_dict, train_num_category_list, unlabel_num_per_category):
    train_examples_dict, unlable_example_dict = {}, {}
    idx = 0
    for key, value in examples_dict.items():
        random.shuffle(value)
        train_num_per_category_i = train_num_category_list[idx]
        train_examples_dict[key] = value[0: train_num_per_category_i]
        unlable_example_dict[key] = value[train_num_per_category_i: train_num_per_category_i + unlabel_num_per_category]
        idx += 1
    return train_examples_dict, unlable_example_dict

def csv_write(examples_dict, save_dir):
    total_num = 0
    with open(save_dir, 'w', encoding='utf8') as f_w:
        writer = csv.writer(f_w)
        for key, value in examples_dict.items():
            writer.writerows(value)
            total_num += len(value)
    f_w.close()
    print('%s writing finished!!! total rows: %i'%(save_dir, total_num))



parser = argparse.ArgumentParser()
parser.add_argument("--ori_train_csv_path", default='/xxx/ASCM_main/data/ag_news_csv/train.csv', type=str, help="")
parser.add_argument("--train_csv_save_path", default='/xxx/ASCM_main/data/ag_news_csv/train_select_10_2.csv', type=str, help="")
parser.add_argument("--unlabel_csv_save_path", default='/xxx/ASCM_main/data/ag_news_csv/unlabeled_select_10_2.csv', type=str, help="")
parser.add_argument("--num_class", default=4, type=int, help="")## YahooAnswers: 10; YelpFull: 5; AGNew: 4; MNLI: 3;
parser.add_argument("--train_num", default=10, type=int, help="")
parser.add_argument("--unlabel_num_per_category", default=10000, type=int, help="")
parser.add_argument("--lable_index_csv", default=0, type=int, help="")
args = parser.parse_args()


if __name__ == "__main__":
    assert os.path.exists(args.ori_train_csv_path), 'Wrong occured!!! %s not exist!!!'%args.ori_train_csv_path

    examples_dict = create_example_csv(args.ori_train_csv_path, args.lable_index_csv)
    print('labels: {}'.format(examples_dict.keys()))

    train_num_category_list = eq_div(args.train_num, args.num_class)
    train_examples_dict, unlable_example_dict = example_split(examples_dict, train_num_category_list, args.unlabel_num_per_category)

    csv_write(train_examples_dict, args.train_csv_save_path)
    csv_write(unlable_example_dict, args.unlabel_csv_save_path)
    