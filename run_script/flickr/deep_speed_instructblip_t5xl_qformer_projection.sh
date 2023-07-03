###
 # @Author: JustBluce 972281745@qq.com
 # @Date: 2022-11-24 13:29:31
 # @LastEditors: JustBluce 972281745@qq.com
 # @LastEditTime: 2023-02-18 21:04:18
 # @FilePath: /SNIPS/run_script/SNIPS/run_boolq.sh
 # @Description: 测试SNIPS用到的脚本
###

export EXPERIMENT_NAME=instruct_BLIP2_deepSpeed_t5xl_unfreeze_Qformer_Projection_final
export DATASET_NAME=flickr
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_DIR=/home/haozhezhao/models/
export MODEL_NAME=blip2-flan-t5-xxl
model_name_or_path=Salesforce/instructblip-flan-t5-xl
# model_name_or_path=${MODEL_DIR}${MODEL_NAME}
# remember to change the consponding tokenizer model

bs=6
eval_bs=8
lr=5e-5
dropout=0.1
epoch=3
seed=1234
do_train=True
do_test=True
do_valid=True
master_port=29503
model_type=instructblip
deepspeed --master_port $master_port run.py \
--experiment_name ${EXPERIMENT_NAME} \
--dataset_name ${DATASET_NAME} \
--dataset_config_name None \
--max_seq_length 512 \
--overwrite_cache True \
--pad_to_max_length True \
--train_file /home/haozhezhao/Vision-PromptSource/prompt_data_6_5_sampled_json/bilp2-prompt-allshot-multiinst_final_ver \
--validation_file /home/haozhezhao/Vision-PromptSource/prompt_data_6_5_sampled_json/bilp2-prompt-allshot-multiinst_final_ver \
--test_file /home/haozhezhao/Vision-PromptSource/prompt_data_6_5_sampled_json/bilp2-prompt-allshot-multiinst_final_ver \
--do_train $do_train \
--do_eval $do_valid \
--do_predict $do_test \
--per_device_train_batch_size ${bs} \
--bf16 \
--model_type $model_type \
--save_total_limit 3 \
--per_device_eval_batch_size ${eval_bs} \
--gradient_accumulation_steps 1 \
--num_train_epochs ${epoch} \
--output_dir checkpoints/${EXPERIMENT_NAME} \
--overwrite_output_dir \
--learning_rate ${lr} \
--weight_decay 0.0005 \
--seed ${seed} \
--warmup_ratio 0.2 \
--save_strategy steps \
--save_steps 1000 \
--evaluation_strategy steps \
--eval_steps 1000 \
--remove_unused_columns False \
--model_name_or_path $model_name_or_path \
--use_fast_tokenizer True \
--model_revision main \
--eval_type val \
--generation_max_length 50 \
--do_full_training True \
--max_eval_samples 3000 \
--max_predict_samples 3000 \
--deepspeed config/deepspeed_config.json
# --load_best_model_at_end \
# --multiple_choice True

