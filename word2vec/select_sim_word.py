import gensim.downloader
from gensim.models import KeyedVectors
# print(list(gensim.downloader.info()['models'].keys()))
from transformers import RobertaTokenizer
from gensim.models import Word2Vec
from collections import defaultdict
import os

class_dict = {
    'yahoo': ["Society", "Science", "Health", "Education", "Computer", "Sports", "Business", "Entertainment", "Relationship", "Politics"],
    'yelp': ["terrible", "bad", "okay", "good", "great"],
    'agnews': ["World", "Sports", "Business", "Technology"],
    'mnli': ['No', 'Yes', 'Maybe'],
}

class_lower_dict = {
    'yahoo': ["society", "science", "health", "education", "computer", "sports", "business", "entertainment", "relationship", "politics"],
    'yelp': ["terrible", "bad", "okay", "good", "great"],
    'agnews': ["world", "sports", "business", "technology"],
    'mnli': ['no', 'yes', 'maybe'],
}

def find_best_label(label_list):
    best_score = 0.0
    best_label = label_list[0][0]
    for label_i in label_list:
        if label_i[1] > best_score:
            best_score = label_i[1]
            best_label = label_i[0]
    return best_label



model = KeyedVectors.load_word2vec_format('/xxx/word2vec-google-news-300/word2vec-google-news-300.gz', binary=True)
for dataset in ['yahoo', 'yelp', 'agnews', 'mnli']:
    score_thre = {'yahoo': 0.4, 'yelp': 0.4, 'agnews': 0.4, 'mnli': 0.6}
    class_label_words = class_dict[dataset]
    word_label_dict = defaultdict(list)
    for class_label_word_i in class_label_words:
        sim_word_list = model.most_similar(class_label_word_i.lower(), topn=100)
        for (sim_word_i, score_i) in sim_word_list:
            if score_i < score_thre[dataset]:
                break
            if sim_word_i in class_lower_dict[dataset]:
                continue
            if '-' in sim_word_i or '_' in sim_word_i or '"' in sim_word_i or '/' in sim_word_i or '(' in sim_word_i or ')' in sim_word_i or '!' in sim_word_i or '\'' in sim_word_i or 'just' in sim_word_i or 'but' in sim_word_i:
                continue
            word_label_dict[sim_word_i].append([class_label_word_i, score_i])

    label_sim_word_list = defaultdict(list)
    for word_i, label_list_i in word_label_dict.items():
        if len(label_list_i) == 1:
            label_sim_word_list[label_list_i[0][0]].append(word_i)
        else:
            best_label_i = find_best_label(label_list_i)
            label_sim_word_list[best_label_i].append(word_i)

    print(label_sim_word_list)

    os.makedirs('./word2vec_synonyms', exist_ok=True)

    txt_w = open('./word2vec_synonyms/%s.txt'%dataset, 'w')
    for class_label_word_i in class_label_words:
        if class_label_word_i in label_sim_word_list.keys():
            txt_w.writelines('%s '%class_label_word_i + ' '.join(label_sim_word_list[class_label_word_i]) + '\n')
        else:
            txt_w.writelines(('%s '%class_label_word_i + '\n'))
    txt_w.close()