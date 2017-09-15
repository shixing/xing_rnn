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
7. [TODO]BPE;
8. [TODO]Ensemble;

# Installation
Current version is compatible with `tensorflow 1.12` and `cuda 8.0`

```bash
git clone https://github.com/shixing/xing_rnn.git
cd xing_rnn/Seq2Seq/
bash init.sh  # create folder: model, jobs
```

# Train & Decode

## Single Model on a Single Machine
Folder `data/small` contains the train, dev and test data set for a toy task `string copy`. To train a seq2seq model on your `GPU` or `CPU`, run the following script: 
```bash 
cd xingshi_rnn/Seq2Seq/sh
bash smallm4h100d07l01n2attadagradAddNS.train.sh
```
As the script name indicates, it's a 2 layers (n2), 100 hidden states (h100) seq2seq model with additive attention (att AddNS), and trained by adagrad (adagrad) with learining rate 0.1 (l01), dropout rate 0.3 (d07) and batch size 4 (m4). The saved model and log file will appears in `xing_rnn/Seq2Seq/model/smallm4h100d07l01n2attadagradAddNS/`.
To decode and get the BLEU score
```bash
bash smallm4h100d07l01n2attadagradAddNS.b10.decode.sh
bash smallm4h100d07l01n2attadagradAddNS.b10.bleu.sh
```
As the script name indicates, the beam size of decoding is 10. The decode output will be `model/smallm4h100d07l01n2attadagradAddNS/decode_output/b10.output`, and the BLEU score will be `model/smallm4h100d07l01n2attadagradAddNS/decode_output/b10.bleu`.

## Data Parallel on a Multi-GPU Machine
We also support data parallelism for training, i.e. you can make several replica of your model and place them on different GPUs, all these replica will train in a `synchronize` way. This essentially means a larger batch size, it's more stable so that you can choose a larger learning rate to start with. More detials about `synchronize-SGD`, please refer to [Revisiting Distributed Synchronous SGD](https://openreview.net/pdf?id=D1VDZ5kMAu5jEJ1zfEWL).
To train our string copy seq2seq model with 4 replicas on a `4 GPU` machine, run the following script: 
```bash 
bash smallm4h100d07l01n2attadagradDIST4AddNS.train.sh
```
The model and log will be saved at `model/smallm4h100d07l01n2attadagradDIST4AddNS`.
Similarly, to decode and evaluate the BLEU score, run the following script:
```bash
bash smallm4h100d07l01n2attadagradDIST4AddNS.b10.train.sh
bash smallm4h100d07l01n2attadagradDIST4AddNS.b10.train.sh
```
The decode output will be `model/smallm4h100d07l01n2attadagradDIST4AddNS/decode_output/b10.output`, and the BLEU score will be `model/smallm4h100d07l01n2attadagradDIST4AddNS/decode_output/b10.bleu`.

# Hyperparameter Tuning

## Descriptions
Details of each hyper parameters can be view by:
```
cd py/
python run.py -h 
python runDistributed.py -h # for the distributed version
```
## Grid Search 
We also provide a toolkit for easy grid search of desired hyper parameters. Run the following script: 
```
cd xing_rnn/Seq2Seq/py/util/
python generate_jobs_small.py
```
This generates a series of scripts at folder `xing_rnn/Seq2Seq/jobs/`. These scripts contains 5 sets of  scripts where each set have one training script (`{model_id}.train.sh`), several decoding and BLEU calculation scripts with different beam size (`{model_id}.b{beam_size}.decode.sh` and `{model_id}.b{beam_size}.bleu.sh`). The toolkit will replect the hyper parameters in the name of generated scripts. The hyper parameters of the 5 sets of scripts are: 
```
smallm4h100d05l01n2attadagradAddNS
smallm4h100d05l01n2attadagradMulNS
smallm4h100d07l01n2attadagradAddNS
smallm4h100d07l01n2attadagradMulNS
smallm4h100d07l01n2attadagradDIST4AddNS
```
You can customize the searching space by create a new python file like `generate_jobs_xxx.py`, and just copy and paste the code from `generate_jobs_small.py`. Then you can change the values in the variable `grids` to reflect the the searching range. For example, we want to search the learning rate in [0.05,0.1,0.5,1.0], dropout rate in [0.5,0.7,0.9] and two different attention styles ['additive'，'multiply'], we can make the following `grids`: 
```python 
grids = {"name":["xxx"],
        "batch_size":[4],
        "size": [100],
        "dropout":[0.5, 0.7, 0.9],
        "learning_rate":[0.05, 0.1, 0.5, 1.0],
        "n_epoch":[100],
        "num_layers":[2],
        "attention":[True],
        "from_vocab_size":[100],
        "to_vocab_size":[100],
        "min_source_length":[0],
        "max_source_length":[22],
        "min_target_length":[0],
        "max_target_length":[22],
        "n_bucket":[2],
        "optimizer":["adagrad"],
        "N":["00000"],
        "attention_style":["additive",'multiply'],
        "attention_scale":[False]
    }
```
It will generate us 3\*4\*2 = 24 sets of scripts in `xing_rnn/Seq2Seq/jobs`.

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

## Dropout

For encoder, dropout happens at the input of each LSTM layers. 
For decoder, dropout happens at the input of each LSTM layers. Additionally, we will also dropout the output of the top LSTM layer. 
For attention layer, we do not do any dropouts at either the input or output. 

## Dynamic RNN
By default, the code will use `tf.nn.dynamic_rnn`. It's about 15% faster and can reduce the memory usage on GPU by 50%, thanks to the `swap_memory` option.

You can choose to use `tf.nn.static_rnn` by setting the flag `dynamic_rnn = False`. 

## Bucket sizes
`max_source_length` = max source length
`max_target_length` = max target length + 1 (due to _GO and _EOS)

If we use  `tf.nn.dynamic_rnn`, we do not need to create a seperate seq2seq model for each bucket. However, we still need the bucket mechanism to find similar length pairs to form in a batch. 

Here is how we find similar length sentences into a batch: 
1. Hash the pair into buckets according to each bucket's source length and target length. 
2. To form a batch, we randomly choose a bucket, and then randomly select `batch_size` pairs as a batch. 
3. Calculate the actual max source length and max target length in that batch. Then we pad each pair in the batch to make sure every pair in a batch have the same length. 




## Multi-GPU support 
### Model parallel
### Data parallel 
	
## Decoding Length Control

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
7. **Alwahys use dynamic_rnn, faster and less memory usage.**

NOTE:

1. In "N00001", each digit represents the device placement of the following layers: [Input layer@GPU0, LSTM layer 1@GPU0, LSTM layer 2@GPU0, Attention layer@GPU0, Softmax layer@**GPU1**]
2. The speed here is **source + target words per second**.

 | Hyperparameter | Changed value | Baseline value | Speed (word) | GPU0 (MB) | GPU1 (MB) | 
|---|---|---|---|---|---|
 | Baseline |  |  | 1890 | 612 | 2150 | 
 | Batch size | 64 | 32 | 3150 | 612 | 2150 | 
 | Batch size | 128 | 32 | 5250 | 1124 | 4198 | 
 | Hidden size | 400 | 32 | 1533 | 2148 | 4198 | 
 | Buckets | 2 | 1 | 2835 | 612 | 2150 | 
 | Buckets | 4 | 1 | 3675 | 612 | 2150 | 
 | Buckets | 8 | 1 | 3990 | 612 | 2150 | 
 | Max length | 100 | 50 | 1092 | 1124 | 4198 | 
 | Source vocab | 10000 | 40000 | 1890 | 612 | 2150 | 
 | Target vocab | 50000 | 40000 | 1680 | 612 | 4198 | 
 | Optimizor | SGD | Adagrad | 1890 | 614 | 2150 | 
 | Softmax | Sampled 500 | Full-softmax | 3045 | 612 | 131 | 
 | Device placement | N00000 | N00001 | 1533 | 2150 | 62 | 
 | Attention | Non-attention | Additive | 2100 | 356 | 2150 | 
 | Attention | Non-attention + N00000 | Additive | 1953 | 2150 | 62 | 
 | Dynamic RNN | dynamic_rnn | static_rnn | 2400 | 356 | 742 | 




### Large Settings
Several large scale models (2 layers, 40k target vocab , max length 50, additive attention, batch size 128) are trained on a 4 K80 GPUs machine. The speed here is still **source + target words per second**.

One important observation is that we should put LSTM layer 2 and attention layer on the same GPU to get better speed(compare row 3 vs 4).

| RNN | Source Vocab | Hidden size | Buckets | Device Placement | Speed | GPU0 | GPU1 | GPU2 | GPU3 | 
|---|---|---|---|---|---|---|---|---|---|
 | static_rnn | 40k | 200 | 1 | N00000 | 5400 | 4198 | 62 | 62 | 62 | 
 | static_rnn | 40k | 200 | 1 | N00001 | 6800 | 1124 | 2150 | 62 | 62 | 
 | static_rnn | 40k | 200 | 1 | N00012 | 5800 | 612 | 1123 | 2150 | 62 | 
 | static_rnn | 40k | 200 | 1 | N00112 | 6300 | 356 | 1123 | 2150 | 62 | 
 | dynamic_rnn | 40k | 200 | 1 | N00001 | 7100 | 1124 | 2278 | 62 | 62 | 
 | static_rnn | 200k | 1000 | 1 | N01223 | 2100 | 1124 | 4198 | 8294 | 10907 | 
 | static_rnn | 200k | 1000 | 5 | N01223 | 4300 | 1124 | 4198 | 8294 | 10907 | 
 | static_rnn | 200k | 1000 | 5 | N00001 | 4000 | 8294 | 10907 | 62 | 62 | 
 | dynamic_rnn | 200k | 1000 | 5 | N00001 | 4230 | 4198 | 2150 | 62 | 62 | 
 | dynamic_rnn | 200k | 1000 | 5 | N01223 | 4280 | 356 | 1126 | 4198 | 2150 | 



# Reference
[Tensorflow NMT](https://github.com/tensorflow/nmt)
[Distributed Tensorflow](https://github.com/tensorflow/benchmarks/blob/master/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py)


# TODO
1. Layer batch normalization
2. BPE encoding

# Contact 
Please contact Xing Shi (shixing19910105@gmail.com) for any questions. 