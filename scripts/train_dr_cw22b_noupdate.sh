#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:A6000:2
#SBATCH --time=1-00:00:00
module load cuda-12.1
eval "$(conda shell.bash hook)"
conda activate openmatch

DATA_PATH="."

text_length=2048
n_gpus=2

export OPENMATCH_DIR="/home/zhenfan/t5-rope-2048/AnchorDR/OpenMatch/src/openmatch/"

train_qrels=$DATA_PATH/data/training_data/train.qrels.tsv
train_queries=$DATA_PATH/data/training_data/train.query.txt
corpus=$DATA_PATH/data/training_data/corpus_cw22btrain.tsv
negatives=$DATA_PATH/data/training_data/train.negatives.tsv

initial_model=jmvcoelho/t5-base-marco-crop-pretrain-2048

trained_model_name=t5rope-cw22b-$text_length-noupdate-5ep

train_data_folder=$DATA_PATH/data/training_data/$trained_model_name
mkdir -p $train_data_folder

echo "########################################"
echo "Building initial data"
echo "########################################"
'''
python OpenMatch/scripts/msmarco/build_train.py \
   --tokenizer_name $initial_model \
   --negative_file $negatives  \
   --qrels $train_qrels  \
   --queries $train_queries  \
   --collection $corpus \
   --truncate $text_length \
   --save_to $train_data_folder  \
   --doc_template "Title: <title> Text: <text>" \
   --n_sample 9


cat $train_data_folder/split*.jsonl > $train_data_folder/full.jsonl
rm $train_data_folder/split*.jsonl

line_count=$(wc -l $train_data_folder/full.jsonl | awk '{print $1}')
n_val=500
n_train=$((line_count - n_val))

echo $n_train

tail -n $n_val $train_data_folder/full.jsonl > $train_data_folder/val.jsonl
head -n $n_train $train_data_folder/full.jsonl > $train_data_folder/train.jsonl

#rm $train_data_folder/full.jsonl
'''
echo "########################################"
echo "Train episodes"
echo "########################################"

train_data=$train_data_folder/train.jsonl
valid_data=$train_data_folder/val.jsonl
output_path=$DATA_PATH/models/$trained_model_name


accelerate launch --num_processes $n_gpus --multi_gpu --main_process_port 29989 OpenMatch/src/openmatch/driver/train_dr.py  \
    --output_dir $output_path \
    --model_name_or_path $initial_model \
    --do_train \
    --save_steps 125  \
    --eval_steps 125  \
    --fp16 \
    --train_path $train_data  \
    --eval_path $valid_data  \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 4 \
    --train_n_passages 10  \
    --learning_rate 5e-6  \
    --q_max_len 32  \
    --p_max_len $text_length  \
    --num_train_epochs 2  \
    --report_to wandb \
    --logging_steps 10 \
    --run_name $trained_model_name \
    --evaluation_strategy steps \
    --dataloader_num_workers 8 \
    --rope True \
    --grad_cache True \
    --use_mapping_dataset True \
    --gc_p_chunk_size 24 \
    --gc_q_chunk_size 24 \
    --negatives_x_device True 

# Hard negative sampling - ANCE negative refresh
# set variables for next training episode

initial_model=$output_path

embeddings_out=$DATA_PATH/data/embeddings/train/$trained_model_name
run_save=$DATA_PATH/data/negatives/$trained_model_name

trained_model_name=$trained_model_name-self-hn-$i
train_data_folder=$DATA_PATH/data/training_data/$trained_model_name

mkdir -p $run_save
mkdir -p $embeddings_out

# Train again with the hard negatives - do not set the variable for the last one
if [ $i -ne $num_episodes ]; then
    train_data=$train_data_folder/train.jsonl
    valid_data=$train_data_folder/val.jsonl
    output_path=$DATA_PATH/models/marco/$trained_model_name
fi
