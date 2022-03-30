# ASCM
Training step:
1. Prepare data:  training_example_split.py
2. Prepare word2vec model for initialization of SCM and SC
(1) train task-specific word2vec model.: word2vec/word2vec.py Or download pre-trained word2vec embeddings such as word2vec-google-news-300.gz.
(2) generate synonym dataset: word2vec/select_sim_word.py
(3) filter repeating words such as "Yes" and "Yes!!!" and wrong words such as Realtionship
(4) pre-train SCM and SC layer: word2vec/train.py
3. train ASCM+SL/iPET model: train_ascm.py/train_ipet.py; 
(1) modify the code about pretrained roberta in train_ascm.py/train_ipet.py; 
(2) modify the code about pretrained word2vec SCM and SC in train_lm.py(line 145: class_state_dict)
(3) modify dataset path in train_ascm.py/train_ipet.py; 

To accelerate the training process, you can reduce the frequency of evaluation or the number of unlabeled dataset in SL/iPET (unlabel_num_per_category in training_example_split.py).
