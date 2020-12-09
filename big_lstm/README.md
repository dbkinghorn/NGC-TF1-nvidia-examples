# F-LM

This code was forked from https://github.com/rafaljozefowicz/lm 
Which implemented LSTM language model baseline from https://arxiv.org/abs/1602.02410
The code supports running on the machine with multiple GPUs using synchronized gradient updates (which is the main difference with the paper).

# Main modification from the forked code

* Support for TensorFlow 1.0 (original supported the 0.10 API) 
* More fine grained summary tracking
* full eval mode
* dataset download script 
 
# Performance

## Hyperparameters
* number of time steps: 20
* number of shards: 8
* number of LSTM layers in a stack: 2
* learning rate: 0.2 (does not affect performance)
* max grad norm (grad clipping): 1 (does not affect performance)
* retention probability (for dropout): 0.9 (does not affect performance)
* embedding size: 1024
* projected size: 1024
* internal state size: 8192
* batch size per GPU: 512

## Performance on DGX-1 (words per second)
* 1 GPU  : 6,936
* 2 GPUs : 12,728
* 4 GPUs : 23,481
* 8 GPUs : 33,900 

## Dependencies
* TensorFlow 1.0
* Python 2.7
* 1B Word Benchmark Dataset (see `download_1b_words_data.sh` in this repository to download data)

## To run

NOTE: the `--logdir` option isn't for traditional logs, it's to store TensorBoard logs and checkpoints for future model retraining, or to run inference once the model has been trained.

Assuming the data directory is in: `/data/1-billion-word-language-modeling-benchmark-r13output/`i and the training logs are in `/logs`, execute:

```

#train

MAX_TRAIN_TIME_SEC=180

python single_lm_train.py --mode=train --logdir=/logs --num_gpus=8 --datadir=/data/1-billion-word-language-modeling-benchmark-r13output/ --hpconfig run_profiler=False,max_time=${MAX_TRAIN_TIME_SEC},num_steps=20,num_shards=8,num_layers=2,learning_rate=0.2,max_grad_norm=1,keep_prob=0.9,emb_size=1024,projected_size=1024,state_size=8192,num_sampled=8192,batch_size=512

#eval
python single_lm_train.py --logdir=/logs --num_gpus=8 --datadir=/data/1-billion-word-language-modeling-benchmark-r13output/ --mode=eval_full --hpconfig run_profiler=False,num_steps=20,num_shards=8,num_layers=2,learning_rate=0.2,max_grad_norm=1,keep_prob=0.9,emb_size=1024,projected_size=1024,state_size=8192,num_sampled=8192,batch_size=512 
```

## To change hyper-parameters

The command accepts and additional argument `--hpconfig` which allows to override various hyper-parameters, including (default values listed):
* batch_size=128 - batch size
* num_steps=20 - number of unrolled LSTM steps
* num_shards=8 -  embedding and softmax matrices are split into this many shards
* num_layers=1 - number of LSTM layers
* learning_rate=0.2 - learning rate for adagrad
* max_grad_norm=10.0 - maximum acceptable gradient norm 
* keep_prob=0.9 - for dropout between layers (here: 10% dropout before and after each LSTM layer)
* emb_size=512 - size of the embedding
* state_size=2048 - LSTM state size
* projected_size=512 - LSTM projection size 
* num_sampled=8192 - number of word target samples for IS objective during training

## Feedback
Original code/paper: rafjoz@gmail.com
Adaptation for TensorFlow 1.0, download script, refactoring: mkolodziej@nvidia.com.
