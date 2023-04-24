
export EXPERIMENT_NAME=BLIP2_FLICKR_deepSpeed_mix_shot_multiinst_6B
export DATASET_NAME=flickr
export CUDA_VISIBLE_DEVICES=3,4
export MODEL_DIR=/home/haozhezhao/models/
export MODEL_NAME=blip2-flan-t5-xxl

bs=2
eval_bs=4
lr=1e-4
dropout=0.1
epoch=2
seed=1234

accelerate launch --config_file config/accelerate_zero3_config_2gpu.yaml run.py \
--experiment_name ${EXPERIMENT_NAME} \
--dataset_name ${DATASET_NAME} \
--dataset_config_name None \
--max_seq_length 512 \
--overwrite_cache True \
--pad_to_max_length True \
--train_file /home/haozhezhao/Vision-PromptSource/arrow_data_bilp2-prompt-allshot-multiinst_train \
--validation_file /home/haozhezhao/Vision-PromptSource/arrow_data_bilp2-prompt-allshot-multiinst_val \
--test_file /home/haozhezhao/Vision-PromptSource/arrow_data_bilp2-prompt-allshot-multiinst_test \
--do_train \
--do_eval \
--do_predict \
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
--save_strategy steps \
--save_steps 5000 \
--evaluation_strategy steps \
--eval_steps 250 \
--remove_unused_columns False \
--model_name_or_path ${MODEL_DIR}${MODEL_NAME} \
--use_fast_tokenizer True \
--model_revision main \
--eval_type val \
--generation_max_length 32 \
--do_full_training True \
--max_eval_samples 3000 \
--max_predict_samples 3000 \
# --load_best_model_at_end \
# --multiple_choice True

