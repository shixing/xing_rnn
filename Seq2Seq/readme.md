Seq2Seq model based on Tensorflow
----------

[TOC]

# Introduction
A tensorflow based **Seq2Seq model** implementation with following highlights:

 1. Multiple attention choices : **additive** and **multiplicative**, both with **scale** option.
 2. Multi-GPU support : **model parallel** (put different layer on different GPU) and **data  parallel** (put multiple replicas on different GPU).
 3. Beam search decoding.
 4. Easy hyperparameter tuning: a single script to generate both **training** and **decoding** scripts on different hyperparameter settings (**grid search**).
 5. A good documentation to illustrate **padding**, **source reverse**, **buckets**, and **multi-GPU support**
 6. Detail analysis the effect of different hyperparameter settings on **speed**, **GPU memory usage** and **performance**.

# Required Packages

Current version is compatible with `tensorflow 1.12` and `cuda 8.0`

#Train & Decode

# Hyperparameter Tuning

## Descriptions
## Grid Search 

# Advanced Guide

## File Structure
```
data/{data_set_id}/
	train.src
	train.tgt
	dev.src
	dev.tgt
	test.src
	test.tgt
model/{model_id}/
	data_cache/
		train.src.ids
		train.tgt.ids
		dev.src.ids
		dev.tgt.ids
		vocab.from
		vocab.to
	saved_model/
		train.summary/
	decode_output/
		{decode_id}.output
		{decode_id}.bleu
		{decode_id}/
			test.src.id
	log.TRAIN.txt
	log.BEAM_DECODE.{decode_id}.txt
```

## Attention


## Bucket sizes

`max_source_length` = max source length

`max_target_length` = max target length + 1 (due to _GO and _EOS)

## Multi-GPU support 
### Model parallel
### Data parallel 
	
## Tensorboard:

```bash
# on the server where you do training
$ source init_tensorflow.sh
$ tensorboard --logdir=model/{model_id} --port=8080
# anther terminal (port foward)
$ ssh -L 6006:127.0.0.1:8080 xingshi@hpc-login2.usc.edu
```
Then open browser: http://localhost:6006

# Experiments

## Speed and GPU memory usage

### Small Scale
In this section, a baseline model (2 layer with 200 hidden states) is trained on 2 K20 GPUs. Later, we choose a single hyperparameter and change its value to see the effects on both speed and GPU memory consumption.  Some highlights can be extracted from the following table:

1. Double the **batch size** will  double the memory usage,  and almost double the speed.
2. Double the **hidden state size** will double the memory usage.
3. Increase the **number of buckets** will only speed up training with the same memory consumption.
4. Double the **maximum length** of both source side and target side will double the memory consumptions.
5. **Target vocab size** affects both speed and memory consumption heavily.
6. Putting softmax layer on another GPU will speed up the whole training. 

NOTE:

1. In "N00001", each digit represents the device placement of the following layers: [Input layer@GPU0, LSTM layer 1@GPU0, LSTM layer 2@GPU0, Attention layer@GPU0, Softmax layer@**GPU1**]
2. Due to the legacy reason, the speed here is **target words per second**. However, our current code will  report **source + target words per second** 

| Hyperparameter | Changed value | Baseline value | Speed (word/s) | GPU0 (MB) | GPU1 (MB) | 
|---|---|---|---|---|---|
 |  |  |  | 900 | 612 | 2150 | 
 | Batch size | 64 | 32 | 1500 | 612 | 2150 | 
 | Batch size | 128 | 32 | 2500 | 1124 | 4198 | 
 | Hidden size | 400 | 32 | 730 | 2148 | 4198 | 
 | Buckets | 2 | 1 | 1350 | 612 | 2150 | 
 | Buckets | 4 | 1 | 1750 | 612 | 2150 | 
 | Buckets | 8 | 1 | 1900 | 612 | 2150 | 
 | Max length | 100 | 50 | 520 | 1124 | 4198 | 
 | Source vocab | 10000 | 40000 | 900 | 612 | 2150 | 
 | Target vocab | 50000 | 40000 | 800 | 612 | 4198 | 
 | Optimizor | SGD | Adagrad | 900 | 614 | 2150 | 
 | Softmax | Sampled 500 | Full-softmax | 1450 | 612 | 131 | 
 | Device placement | N00000 | N00001 | 730 | 2150 | 62 | 
 | Attention | Non-attention | Additive | 1000 | 356 | 2150 | 
 | Attention | Non-attention + N00000 | Additive | 930 | 2150 | 62 | 

### Large Settings
Several large scale models (2 layers, 40k target vocab , max length 50, additive attention) are trained on a 4 K80 GPUs machine. The speed here is still **target words per second**.

One important observation is that we should put LSTM layer 2 and attention layer on the same GPU to get better speed(compare row 3 vs 4).

| Source Vocab | Hidden Size | Buckets | Device Placement | Speed | GPU0 | GPU1 | GPU2 | GPU3 | 
|---|---|---|---|---|---|---|---|---|
 | 40k | 200 | 1 | N00000 | 2700 | 4198 | 62 | 62 | 62 | 
 | 40k | 200 | 1 | N00001 | 3300 | 1124 | 2150 | 62 | 62 | 
 | 40k | 200 | 1 | N00012 | 2900 | 612 | 1123 | 2150 | 62 | 
 | 40k | 200 | 1 | N00112 | 3150 | 356 | 1123 | 2150 | 62 | 
 | 200k | 1000 | 1 | N01223 | 1100 | 1124 | 4198 | 8294 | 10907 | 
 | 200k | 1000 | 5 | N01223 | 2280 | 1124 | 4198 | 8294 | 10907 | 
 | 200k | 1000 | 5 | N00001 | 2100 | 8294 | 10907 | 62 | 62 | 

# Reference

# TODO
1. Layer batch normalization
2. BPE encoding

# Contact 
Please contact Xing Shi (shixing19910105@gmail.com) for any questions. 