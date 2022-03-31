# ASCM
Our work " ASCM: An Answer Space Clustered Prompting Method without Answer Engineering" is presented on ACL 2022 (Findings).

ASCM is a prompting learning method, focusing on text classification and NLI tasks, which needs no answer engineering (different from manual answer design, discret answer searching, or soft verbalizer) by SCM and SI. SCM (FC + BN + Tanh) is used to transform PLM-encoded token embeddings to another embedding space where words cluster based on semantics or some relationships. And SI is used to initiate additional neural networks (SCM and SC) and attain significant improvements on few-shot tasks. Together ASCM outperform PET (manual answer design) by a large margin especially on fewer labeled samples. SL (stair learning) is an iterative semi-supervised learning method for exploiting unlabeled datasets.

## Resources:
1. PLM: Roberta_base: [https://huggingface.co/roberta-large]
2. Word2vec model: [https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/] [https://developer.syn.co.in/tutorial/bot/oscova/pretrained-vectors.html]
3. datasets: (AG's News): [http://goo.gl/JyCnZq]
             (Yahoo): [http://goo.gl/JyCnZq]
             (Yelp): [http://goo.gl/JyCnZq]
             (MNLI): [https://cims.nyu.edu/˜sbowman/multinli/]

## Training step:
1. Prepare data:  training_example_split.py (a random-splited AG's News dataset can be found in data folder with 10 labeled samples and 4x10000 unlabeled samples)
2. Prepare word2vec model for initialization of SCM and SC (SI: Synonym Initialization)  
2.1 train task-specific word2vec model.: word2vec/word2vec.py or download pre-trained word2vec embeddings such as word2vec-google-news-300.gz (better).  
2.2 generate synonym dataset: word2vec/select_sim_word.py (For MNLI task, you can use the manual designed synonym dataset (folder word2vec/word2vec_synonyms/mnli_manual_designed.txt)introduced in appendix, and modify line 105 in word2vec/train.py to "train_dataset = Sim_Word('./word2vec_synonyms/%s_manual_designed.txt'%args.dataset, tokenizer, token_embedding)")  
2.3 skip this step or filter repeating words such as "Yes" and "Yes!!!" and wrong words such as Realtionship  
2.4 pre-train SCM and SC layer: 
```
python word2vec/train.py --dataset yahoo \ ##[yahoo, yelp, agnews, mnli];
     --batch_size 8 \
     --num_class 10 \ ##YahooAnswers: 10; YelpFull: 5; AGNew: 4; MNLI: 3;
     --gpu_id 0 \
```

3. train ASCM+SL/iPET model: train_ascm.py/train_ipet.py or train ASCM model (comment the iterative parts);  
3.1 modify the code about pretrained roberta model path in train_ascm.py (line 18)/train_ipet.py(line 16);   
3.2 modify the code about pretrained SCM and SC (step 2.4, select the best model) in train_lm.py(line 145: class_state_dict)  
3.3 modify dataset path in train_ascm.py (line 31-102)/train_ipet.py (line 28-61);   

```
python train_ascm.py --dataset YahooAnswers \ ##[YahooAnswers, YelpFull, AGNews, MNLI]
     --work_dir 10_lm \
     --train_num 10 \
     --gpu_id 0 \
     --output_dir ./log \
```
To accelerate the training process, you can reduce the frequency of evaluation or the number of unlabeled dataset in SL/iPET (unlabel_num_per_category in training_example_split.py).

## Training ASCM without SL.
(yahoo, number of labled samples = 10)use command 

```
python train_lm.py --lm_training \
     --dataset YahooAnswers \ ##[YahooAnswers, YelpFull, AGNews, MNLI]
     --batch_size 8 \
     --test_batch_size 8 \
     --pattern_id 0 \ ##YahooAnswers: 0-5; YelpFull: 0-3; AGNew: 0-5; MNLI: 0-1;
     --num_class 10 \ ##YahooAnswers: 10; YelpFull: 5; AGNew: 4; MNLI: 3;
     --trainset /xxx/dataset/yahoo_answers_csv/train_select_10.csv \
     --testset /xxx/dataset/yahoo_answers_csv/test_select_10.csv \
     --pretrainset /xxx/dataset/yahoo_answers_csv/unlabeled_select_10.csv \
     --unlabelset /xxx/dataset/yahoo_answers_csv/unlabeled_select_10.csv \
     --model transformer_pre_fc_tanh_init \
     --base_model roberta \
     --pretrain_model_path /xxx/pretrain_model/roberta_large \
     --coef_loss_lm 1.0 \
     --num_train_epochs 300 \
     --eval_epochs 100 \
     --output_dir ./log \
     --tag pattern_0 \
     --gpu_id 0
```
## ASCM Results without SL.
|                   | Yelp | AG's | Yahoo | MNLI |
|  ---------------  | -----------  | ------------- | ------------ | ------------- |
| T=10              | 53.0 | 82.5 | 62.0 | 48.5 |
| T=50              | 61.2 | 88.3 | 68.4 | 68.9 |
| T=100              | 62.7 | 89.2 | 68.6 | 74.1 |
| T=1000              | 64.8 | 91.1 | 73.3 | 80.5 |
