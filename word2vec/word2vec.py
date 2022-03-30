import os
import csv
import gensim
from gensim.models import word2vec

def Yahoo_text():
    path = '/xxx/dataset/yahoo_answers_csv/train.csv'
    text_all = []
    with open(path, encoding='utf8') as f:
        reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(reader):
            label, question_title, question_body, answer = row
            text_a = ' '.join([question_title.replace('\\n', ' ').replace('\\', ' '),
                                question_body.replace('\\n', ' ').replace('\\', ' ')])
            text_b = answer.replace('\\n', ' ').replace('\\', ' ')
            text_i = (text_a + text_b).replace(',', '').replace('.', '').replace(';', '').replace(':', '').replace('?', '')
            text_all.append(text_i.lower().split(' '))
    return text_all

def Yelp_text():
    path = '/xxx/dataset/yelp_review_full_csv/train.csv'
    text_all = []
    with open(path, encoding='utf8') as f:
        reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(reader):
            label, body = row
            text_a = body.replace('\\n', ' ').replace('\\', ' ')
            text_i = text_a.replace(',', '').replace('.', '').replace(';', '').replace(':', '').replace('?', '')
            text_all.append(text_i.lower().split(' '))
    return text_all

def AGNews_text():
    path = '/xxx/dataset/ag_news_csv/train.csv'
    text_all = []
    with open(path, encoding='utf8') as f:
        reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(reader):
            label, headline, body = row
            text_a = headline.replace('\\', ' ')
            text_b = body.replace('\\', ' ')
            text_i = (text_a + text_b).replace(',', '').replace('.', '').replace(';', '').replace(':', '').replace('?', '')
            text_all.append(text_i.split(' '))
    return text_all

def MNLI_text():
    path = '/xxx/dataset/multinli/train.csv'
    text_all = []
    with open(path, encoding='utf8') as f:
        reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(reader):
            label, text_a, text_b = row
            text_a = text_a.strip()
            text_b = text_b.strip()
            text_i = (text_a + text_b).replace(',', '').replace('.', '').replace(';', '').replace(':', '').replace('?', '')
            text_all.append(text_i.lower().split(' '))
    return text_all


if __name__ == '__main__':
    text_list = AGNews_text()
    model = word2vec.Word2Vec(text_list, vector_size=100, sg=0, workers=16)
    model.save('./word2vec_model/agnews.model')
    # model.load('./word2vec_model/yelp.model')
    # sim_word_list = model.wv.similar_by_word('sports', topn=100)
