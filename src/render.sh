#!/bin/bash                                                                                                                                                                              

#python -W ignore render.py -path2data /viscam/u/ying1123/dataset/kit-mocap/kit-mocap-lang2pose \
#-render_list utils/render_list

python render.py -path2data /viscam/u/ying1123/language2pose/src/save/model/exp_42_cpk_jl2p_model_Seq2SeqConditioned9_time_16_chunks_1\
                 -render_list subsets/render_list

#python render.py -path2data /viscam/u/ying1123/dataset/kit-mocap/kit-mocap-primitive/00001 \
#                 -render_list subsets/render_list

