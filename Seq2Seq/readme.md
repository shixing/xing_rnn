Data Format:

# File Structure

```
raw_data/
	train
	dev
	test

model_{version}/
	data_cache/
		train.ids
		dev.ids
		test.ids
		vocab
	saved_model/
		summary
	log.train.txt
	log.beam_decode.txt
	log.force_decode.txt
```

# Bucket sizes

`max_source_length` = max source length

`max_target_length` = max target length + 1 (due to _GO and _EOS)

	

# Using Tensorboard:

```bash
# on HPC
$ source init_tensorflow.sh
$ tensorboard --logdir=model/model_ptb --port=8080
# anther terminal (port foward)
$ ssh -L 6006:127.0.0.1:8080 xingshi@hpc-login2.usc.edu
```
Then open browser: localhost:6006






# TODO:
1. batch normalization

# Speed Test

## Small Settings

Att V 40k 40k H 200  L 50  bucket 1 m 64  N 001 S : 1500/s GRAM: 612 2150

Att V 40k 40k H 200  L 50  bucket 1 m 128  N 001 S : 2500/s GRAM: 1124 4198

Att V 40k 40k H 400  L 50  bucket 1 m 32  N 001 S : 730/s GRAM: 2148 4198

Att V 40k 40k H 200  L 50  bucket 2 m 32  N 001 S : 1350/s GRAM: 612 2150

Att V 40k 40k H 200  L 50  bucket 4 m 32  N 001 S : 1750/s GRAM: 612 2150

Att V 40k 40k H 200  L 50  bucket 8 m 32  N 001 S : 1900/s GRAM: 612 2150

Att V 40k 40k H 200  L 100  bucket 1 m 32  N 001 S : 520/s GRAM: 1124 4198

Att V 10k 40k H 200  L 50  bucket 1 m 32  N 001 S : 900/s GRAM: 612 2150

Att V 40k 50k H 200  L 50  bucket 1 m 32  N 001 S : 800/s GRAM: 612 4198

SGD Att V 40k 40k H 200  L 50  bucket 1 m 32  N 001 S : 900/s GRAM: 614 2150

Att V 40k 40k H 200  L 50  bucket 1 m 32  N 001 S : 900/s GRAM: 612 2150

Att V 40k 40k H 200  L 50  bucket 1 m 32  N 000 S : 730/s GRAM: 2150 62

Non-Att V 40k 40k H 200  L 50  bucket 1 m 32  N 001 S : 1000/s GRAM: 356 2150

Non-Att V 40k 40k H 200  L 50  bucket 1 m 32  N 000 S : 930/s GRAM: 2150 62

## Large Settings

Att V 200k 40k H 1000  L 50  bucket 5 m 128  N 01223 S : 2280/s  GRAM: 1124 4198 8294 10907

Att V 200k 40k H 1000  L 50  bucket 5 m 128  N 00001 S : 2100/s  GRAM: 8294 10907 62 62