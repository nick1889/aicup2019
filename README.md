# aicup2019
AI CUP 2019 Abstract Classification

use [bert](https://arxiv.org/abs/1810.04805) from [google-research-bert](https://github.com/google-research/bert).    

### Requirements
* tensorflow    
* pandas  
* numpy    
* sklearn    
* spacy    
* tqdm

### Download pre-trained model
Pre-trained model ``BERT-Base, Uncased`` from [bert](https://github.com/google-research/bert).    

### Generate pre-train data
Raw data: [Extra Data - Citation Network Graph](https://github.com/itsmystyle/AI-CUP-2019-Abstract-Labeling-and-Classification-Tutorial/tree/master/Citation%20Network%20Data#extra-data---citation-network-graph)    
Preprocess script:     
``python data_preprocess.py``    
Generate script:
    
    python create_pretraining_data.py \
    --input_file=./data/corpus.txt \
    --output_file=./pretrain_data \
    --vocab_file=uncased_L-12_H-768_A-12/vocab.txt \
    --do_lower_case=True \
    --max_seq_length=384 \
    --max_predictions_per_seq=38 \
    --masked_lm_prob=0.1 \ 
    --random_seed=12345 \
    --dupe_factor=5 

### Pre-train BERT

    python run_pretraining.py
    --input_file=./data \
    --output_dir=./pretraining_output \
    --do_train=True \
    --do_eval=True \
    --bert_config_file=uncased_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=uncased_L-12_H-768_A-12/bert_model.ckpt \
    --train_batch_size=32 \
    --max_seq_length=384 \
    --max_predictions_per_seq=38 \
    --num_train_steps=100000 \
    --num_warmup_steps=50 \
    --learning_rate=5e-5

### Finetune
    
    python run_doc_classifier.py \
    --train_data_index=0 \
    --data_dir=./data \
    --output_dir=./model_0 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --bert_config_file=uncased_L-12_H-768_A-12/bert_config.json \
    --task_name=doc \
    --init_checkpoint=./pretraining_output/model.ckpt-100000 \
    --do_lower_case=True \
    --max_seq_length=384 \
    --do_train=True \
    --do_eval=False \
    --do_predict=True \ 
    --predict_eval_data=True \
    --train_batch_size=32 \
    --eval_batch_size=16 \
    --predict_batch_size=32 \
    --vocab_file=uncased_L-12_H-768_A-12/vocab.txt \
    --bert_config_file=uncased_L-12_H-768_A-12/bert_config.json \
    
### Predict

    python run_doc_classifier.py \
    --train_data_index=0
    --init_checkpoint=./result_model_0/model.ckpt-1050
    --output_dir=./model_0
    --data_dir=./data
    --bert_config_file=uncased_L-12_H-768_A-12/bert_config.json
    --task_name=doc
    --do_lower_case=True
    --max_seq_length=384
    --do_train=False
    --do_eval=False
    --do_predict=True
    --predict_eval_data=False
    --train_batch_size=32
    --eval_batch_size=16
    --predict_batch_size=32
    --learning_rate=2e-5
    --num_train_epochs=3.0
    --vocab_file=uncased_L-12_H-768_A-12/vocab.txt

### Ensemble
view [ensemble.ipynb](https://github.com/nick1889/aicup2019/blob/master/ensemble.ipynb)