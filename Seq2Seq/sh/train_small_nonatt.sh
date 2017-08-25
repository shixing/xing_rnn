PY=../py/run.py
MODEL_DIR=../model/model_small_nonatt
TRAIN_PATH_FROM=../data/small/train.src
DEV_PATH_FROM=../data/small/valid.src
TEST_PATH_FROM=../data/small/test.src
TRAIN_PATH_TO=../data/small/train.tgt
DEV_PATH_TO=../data/small/valid.tgt
TEST_PATH_TO=../data/small/test.tgt


export CUDA_VISIBLE_DEVICES=1


python $PY --mode TRAIN --model_dir $MODEL_DIR \
    --train_path_from $TRAIN_PATH_FROM --dev_path_from $DEV_PATH_FROM \
    --train_path_to $TRAIN_PATH_TO --dev_path_to $DEV_PATH_TO \
    --batch_size 10 --from_vocab_size 100 --to_vocab_size 100 --size 50 --num_layers 2 \
    --n_epoch 200 --saveCheckpoint True --attention False --learning_rate 0.1 --optimizer adagrad --keep_prob 0.9 --L 22 --n_bucket 2
    
