#!/bin/bash
#
#SBATCH --job-name=render_lang2pose
#SBATCH --partition=viscam
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=54

eval "$(conda shell.bash hook)"
conda activate torchnew

python render.py -path2data /viscam/u/ying1123/language2pose/src/save/model/exp_46_cpk_jl2p_model_Seq2SeqConditioned9_time_16_chunks_1\
                 -render_list subsets/render_list

