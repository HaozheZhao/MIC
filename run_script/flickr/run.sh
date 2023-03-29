###
 # @Author: JustBluce 972281745@qq.com
 # @Date: 2022-11-24 13:29:31
 # @LastEditors: JustBluce 972281745@qq.com
 # @LastEditTime: 2023-02-18 21:04:18
 # @FilePath: /SNIPS/run_script/SNIPS/run_boolq.sh
 # @Description: 测试SNIPS用到的脚本
###

export EXPERIMENT_NAME=BLIP2_FLICKR_deepSpeed_3layer_unforzen
export DATASET_NAME=flickr
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export MODEL_DIR=/home/haozhezhao/models/
export MODEL_NAME=blip2-flan-t5-xl
# 没有这行会卡住
# export NCCL_P2P_DISABLE=1

bs=5
eval_bs=8
lr=5e-4
dropout=0.1
epoch=5
seed=1234

accelerate launch --config_file config/accelerate_zero3_config_6gpu.yaml run.py \
--experiment_name ${EXPERIMENT_NAME} \
--dataset_name ${DATASET_NAME} \
--dataset_config_name None \
--max_seq_length 512 \
--overwrite_cache True \
--pad_to_max_length True \
--train_file /home/haozhezhao/Vision-PromptSource/arrow_data_train \
--validation_file /home/haozhezhao/Vision-PromptSource/arrow_data_val \
--test_file /home/haozhezhao/Vision-PromptSource/arrow_data_test \
--do_train \
--do_eval \
--do_predict \
--bf16 \
--per_device_train_batch_size ${bs} \
--per_device_eval_batch_size ${eval_bs} \
--gradient_accumulation_steps 1 \
--num_train_epochs ${epoch} \
--output_dir checkpoints/${EXPERIMENT_NAME} \
--overwrite_output_dir \
--learning_rate ${lr} \
--weight_decay 0.0005 \
--seed ${seed} \
--warmup_ratio 0.2 \
--evaluation_strategy steps \
--eval_steps 250 \
--remove_unused_columns False \
--model_name_or_path ${MODEL_DIR}${MODEL_NAME} \
--use_fast_tokenizer True \
--model_revision main \
--eval_type val \
--generation_max_length 32 \
--do_full_training True \
--max_eval_samples 1000 \
--max_predict_samples 1000 \
--load_best_model_at_end \
# --multiple_choice True

