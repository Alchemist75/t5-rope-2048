#!/bin/bash
#SBATCH --job-name=eval_dr_openmatch
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00

CW22B_PATH="/data/datasets/clueweb22/ClueWeb22_B/txt/en/en00/"
#eval "$(conda shell.bash hook)"
#conda activate openmatch

model_to_eval=./models/t5rope-cw22b-2048-noupdate-5ep/
model_name=$(basename "$model_to_eval")
embeddings_out="./data/embeddings/dev/$model_name"
mkdir -p $embeddings_out

spl=$1

text_length=2048
n_gpus=4
#corpus_dir=$CW22B_PATH/en00$spl/
corpus_path=./data/corpus_cw22btrain.small.tsv


accelerate launch --num_processes $n_gpus --multi_gpu --main_process_port 29777 OpenMatch/src/openmatch/driver/build_index.py  \
    --output_dir $embeddings_out \
    --model_name_or_path $model_to_eval \
    --per_device_eval_batch_size 40  \
    --corpus_path $corpus_path \
    --doc_template "Title: <title> Text: <text>"  \
    --doc_column_names id,title,text  \
    --p_max_len $text_length  \
    --fp16  \
    --dataloader_num_workers 1 \
    --rope True
