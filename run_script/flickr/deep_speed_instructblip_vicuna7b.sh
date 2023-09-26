#!/bin/bash --login
export EXPERIMENT_NAME=instruct_BLIP2_deepSpeed_vicuna_7b_unfreeze_Qformer_Projection_LLM_QV_weight_no_instructqformer_8_14_futher_trained
export DATASET_NAME=flickr
export CUDA_VISIBLE_DEVICES=1,2,3,5,6,7
export MODEL_DIR=models/
export MODEL_NAME=instructblip-vicuna-7b
model_name_or_path=Salesforce/instructblip-vicuna-7b
processor_path=Salesforce/instructblip-vicuna-7b
# model_name_or_path=${MODEL_DIR}${MODEL_NAME}
# remember to change the consponding tokenizer model
bs=3
eval_bs=3
lr=3e-4
dropout=0.1
epoch=5
seed=1234
do_train=True
do_test=True
do_valid=True
master_port=29504
backbone_model=vicuna
model_type=instructblip
data_dir=prompt_data_8_11_max_figure5_vicuna_json
# python -m debugpy --wait-for-client --listen 5679 run.py \
deepspeed --master_port $master_port run.py \
--experiment_name ${EXPERIMENT_NAME} \
--dataset_name ${DATASET_NAME} \
--dataset_config_name None \
--max_seq_length 650 \
--overwrite_cache True \
--pad_to_max_length True \
--train_file Vision-PromptSource/${data_dir}/bilp2-prompt-allshot-multiinst_final_ver \
--validation_file Vision-PromptSource/${data_dir}/bilp2-prompt-allshot-multiinst_final_ver \
--test_file Vision-PromptSource/${data_dir}/bilp2-prompt-allshot-multiinst_final_ver \
--do_train $do_train \
--do_eval $do_valid \
--do_predict $do_test \
--per_device_train_batch_size ${bs} \
--per_device_eval_batch_size ${eval_bs} \
--bf16 \
--bf16_full_eval \
--model_type $model_type \
--save_total_limit 4 \
--gradient_accumulation_steps 6 \
--num_train_epochs ${epoch} \
--output_dir checkpoints/${EXPERIMENT_NAME} \
--learning_rate ${lr} \
--weight_decay 0.0005 \
--seed ${seed} \
--warmup_ratio 0.2 \
--evaluation_strategy steps \
--eval_steps 1000 \
--remove_unused_columns False \
--model_name_or_path $model_name_or_path \
--processor_path $processor_path \
--use_fast_tokenizer True \
--model_revision main \
--eval_type val \
--generation_max_length 256 \
--done_preprocess True \
--max_eval_samples 1500 \
--max_predict_samples 1500 \
--using_instruct_qformer False \
--run_name instructblip-vicuna-7b-Projection_QV_weight_no_instructqformer_futher_trained \
--load_best_model_at_end True \
--metric_for_best_model accuracy \
--greater_is_better True \
--dataloader_num_workers 64 \
--backbone_model $backbone_model \
--save_strategy steps \
--save_steps 1000 \
--deepspeed config/deepspeed_config.json \
--overwrite_output_dir \
--full_bf16_training True \
# --deepspeed config/deepspeed_config_zero3.json \
# --load_best_model_at_end \
# --multiple_choice True

