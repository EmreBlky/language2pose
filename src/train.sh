#!/bin/bash

python train_wordConditioned.py -batch_size 5 -cpk jl2p -curriculum 1 -dataset KITMocap -early_stopping 1 -exp 1 -f_new 8 -feats_kind rifke \
       -losses "['SmoothL1Loss']" -lr 0.001 -mask "[0]" -model Seq2SeqConditioned9 -modelKwargs "{'hidden_size':1024, 'use_tp':False, 's2v':'lstm'}" -num_epochs 1000 \
       -path2data /viscam/u/ying1123/dataset/kit-mocap/kit-mocap-lang2pose -render_list subsets/render_list -s2v 1 -save_dir save/model/ -tb 1 -time 16 -transforms "['zNorm']" -cuda 0 \
       -data_subset subsets/data_subset_cur_full

#python train_wordConditioned.py -batch_size 100 -cpk jl2p -curriculum 1 -dataset KITMocap -early_stopping 1 -exp 1 -f_new 8 -feats_kind rifke \
#       -losses "['SmoothL1Loss']" -lr 0.001 -mask "[0]" -model Seq2SeqConditioned9 -modelKwargs "{'hidden_size':1024, 'use_tp':False, 's2v':'lstm'}" -num_epochs 1000 \
#       -path2data /viscam/u/ying1123/dataset/kit-mocap/kit-mocap-lang2pose -render_list subsets/render_list -s2v 1 -save_dir save/model/ -tb 1 -time 16 -transforms "['zNorm']" -cuda -1

