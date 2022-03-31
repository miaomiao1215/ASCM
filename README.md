# ASCM
Our work " ASCM: An Answer Space Clustered Prompting Method without Answer Engineering" is presented on ACL 2022 (Findings).

ASCM is a prompting learning method, focusing on text classification and NLI tasks, which needs no answer engineering (different from manual answer design, discret answer searching, or soft verbalizer) by SCM and SI. SCM (FC + BN + Tanh) is used to transform PLM-encoded token embeddings to another embedding space where words cluster based on semantics or some relationships. And SI is used to initiate additional neural networks (SCM and SC) and attain significant improvements on few-shot tasks. Together ASCM outperform PET (manual answer design) by a large margin especially on fewer labeled samples. SL (stair learning) is an iterative semi-supervised learning method for exploiting unlabeled datasets.

1. PLM: Roberta_base: [https://huggingface.co/roberta-base]
2. Word2vec model: [https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/] [https://developer.syn.co.in/tutorial/bot/oscova/pretrained-vectors.html]
3. datasets: (AG's News): [http://goo.gl/JyCnZq]
             (Yahoo): [http://goo.gl/JyCnZq]
             (Yelp): [http://goo.gl/JyCnZq]
             (MNLI): [https://cims.nyu.edu/˜sbowman/multinli/]

Training step:
1. Prepare data:  training_example_split.py
2. Prepare word2vec model for initialization of SCM and SC
(1) train task-specific word2vec model.: word2vec/word2vec.py or download pre-trained word2vec embeddings such as word2vec-google-news-300.gz (better).
(2) generate synonym dataset: word2vec/select_sim_word.py
(3) skip this step or filter repeating words such as "Yes" and "Yes!!!" and wrong words such as Realtionship
(4) pre-train SCM and SC layer: word2vec/train.py
3. train ASCM+SL/iPET model: train_ascm.py/train_ipet.py or train ASCM model (comment the iterative parts); 
(1) modify the code about pretrained roberta in train_ascm.py/train_ipet.py; 
(2) modify the code about pretrained word2vec SCM and SC in train_lm.py(line 145: class_state_dict)
(3) modify dataset path in train_ascm.py/train_ipet.py; 

To accelerate the training process, you can reduce the frequency of evaluation or the number of unlabeled dataset in SL/iPET (unlabel_num_per_category in training_example_split.py).
