export EXPERIMENT_NAME=instruct_BLIP2_deepSpeed_vicuna13b
export DATASET_NAME=flickr
export MODEL_DIR=models/
export NCCL_P2P_LEVEL=NVL
export MODEL_NAME=instructblip-vicuna-13b
model_name_or_path=instructblip-vicuna-13b
processor_path=instructblip-vicuna-13b
# Calculate the number of GPUs
IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
num_gpus="${#GPU_IDS[@]}"

echo "Number of GPUs available: $num_gpus"



bs=1
eval_bs=2
lr=5e-5
dropout=0.1
epoch=2
seed=2048
do_train=True
do_test=True
do_valid=True
master_port=29507
backbone_model=vicuna
model_type=instructblip
data_dir=MIC_tool

eval_steps=1000
save_steps=1000

deepspeed_config=config/deepspeed_config.json
generation_max_length=128
label_max_length=128
max_seq_length=1152

load_datatype=json
DONE_PREPROCESS=False
TRAIN_PREPROCESS=True
# Set TRAIN_PREPROCESS to be False if you want to use the streaming preprocess of huggingface dataset; 
# OR it will preprocess the data in the data colloctor fuction of the dataset
# 926000
train_data_size=1889000
gradient_accumulation_steps=8
image_place_holder='<visual_embedding>'

max_steps=$((($epoch * $train_data_size) / ($bs * $gradient_accumulation_steps * $num_gpus)))

# if set train_preprocess to be False and done_preprocess to be False, to enable the streaminig preprocess of huggingface dataset
# then you need to set the number of max_steps and uncomment the last line
echo "Max Step per GPU: $max_steps"

# python -m debugpy --wait-for-client --listen 5679 run.py \
deepspeed --master_port $master_port --num_gpus $num_gpus run.py \
--experiment_name ${EXPERIMENT_NAME} \
--dataset_name ${DATASET_NAME} \
--dataset_config_name None \
--max_seq_length ${max_seq_length} \
--overwrite_cache True \
--pad_to_max_length True \
--done_preprocess ${DONE_PREPROCESS} \
--training_preprocess ${TRAIN_PREPROCESS} \
--load_datatype ${load_datatype} \
--train_file ${data_dir}/MMICL_vicuna_json/train/train.jsonl \
--validation_file ${data_dir}/MMICL_vicuna_json/val/val.jsonl \
--test_file ${data_dir}/MMICL_vicuna_json/test/test.jsonl \
--do_train $do_train \
--do_eval $do_valid \
--do_predict $do_test \
--per_device_train_batch_size ${bs} \
--per_device_eval_batch_size ${eval_bs} \
--bf16 \
--bf16_full_eval \
--model_type $model_type \
--save_total_limit 2 \
--gradient_accumulation_steps ${gradient_accumulation_steps} \
--num_train_epochs ${epoch} \
--output_dir checkpoints/${EXPERIMENT_NAME} \
--learning_rate ${lr} \
--weight_decay 0.0005 \
--seed ${seed} \
--warmup_ratio 0.2 \
--evaluation_strategy steps \
--eval_steps ${eval_steps} \
--remove_unused_columns False \
--model_name_or_path $model_name_or_path \
--processor_path $processor_path \
--use_fast_tokenizer True \
--model_revision main \
--eval_type val \
--generation_max_length ${generation_max_length} \
--label_max_length ${label_max_length} \
--max_eval_samples 800 \
--max_predict_samples 800 \
--using_instruct_qformer False \
--run_name instructblip-vicuna-13b \
--load_best_model_at_end True \
--metric_for_best_model accuracy \
--greater_is_better True \
--backbone_model $backbone_model \
--save_strategy steps \
--save_steps ${save_steps} \
--full_bf16_training True \
--dataloader_num_workers 64 \
--image_place_holder ${image_place_holder} \
--logging_steps 100 \
--data_dir ${data_dir} \
--overwrite_output_dir \
--deepspeed ${deepspeed_config} \
# --max_steps ${max_steps} \

