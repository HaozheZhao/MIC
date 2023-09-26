export EXPERIMENT_NAME=BLIP2_deepSpeed_t5xl_unfreeze_Qformer_Projection_LLM_QV_weight
export DATASET_NAME=flickr
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export MODEL_DIR=models/
export MODEL_NAME=blip2-flan-t5-xl
model_name_or_path=model/blip2-flan-t5-xl


bs=12
eval_bs=16
lr=5e-5
dropout=0.1
epoch=5
seed=1234
do_train=True
do_test=True
do_valid=True
master_port=29504
model_type=blip2
deepspeed --master_port $master_port run.py \
--experiment_name ${EXPERIMENT_NAME} \
--dataset_name ${DATASET_NAME} \
--dataset_config_name None \
--max_seq_length 512 \
--overwrite_cache True \
--pad_to_max_length True \
--train_file Vision-PromptSource/prompt_data_6_5_sampled_json/bilp2-prompt-allshot-multiinst_final_ver \
--validation_file Vision-PromptSource/prompt_data_6_5_sampled_json/bilp2-prompt-allshot-multiinst_final_ver \
--test_file Vision-PromptSource/prompt_data_6_5_sampled_json/bilp2-prompt-allshot-multiinst_final_ver \
--do_train $do_train \
--do_eval $do_valid \
--do_predict $do_test \
--per_device_train_batch_size ${bs} \
--bf16 \
--model_type $model_type \
--save_total_limit 5 \
--per_device_eval_batch_size ${eval_bs} \
--gradient_accumulation_steps 8 \
--num_train_epochs ${epoch} \
--output_dir checkpoints/${EXPERIMENT_NAME} \
--overwrite_output_dir \
--learning_rate ${lr} \
--weight_decay 0.0005 \
--seed ${seed} \
--warmup_ratio 0.2 \
--evaluation_strategy steps \
--eval_steps 100 \
--remove_unused_columns False \
--model_name_or_path $model_name_or_path \
--use_fast_tokenizer True \
--model_type 'blip2' \
--model_revision main \
--eval_type val \
--generation_max_length 256 \
--done_preprocess True \
--max_eval_samples 8000 \
--max_predict_samples 8000 \
--run_name ${EXPERIMENT_NAME} \
--deepspeed config/deepspeed_config.json \
# --save_strategy steps \
# --save_steps 1000 \
# --load_best_model_at_end \
# --multiple_choice True

