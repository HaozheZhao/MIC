###
 # @Author: JustBluce 972281745@qq.com
 # @Date: 2022-11-24 13:29:31
 # @LastEditors: JustBluce 972281745@qq.com
 # @LastEditTime: 2023-02-18 21:04:18
 # @FilePath: /SNIPS/run_script/SNIPS/run_boolq.sh
 # @Description: 测试SNIPS用到的脚本
###

export DATASET_NAME=flickr
export CUDA_VISIBLE_DEVICES=0
export MODEL_DIR=/home/haozhezhao/models/
export MODEL_NAME=/home/haozhezhao/models/blip2-flan-t5-xl
model_name_or_path=/home/haozhezhao/VisionLanguagePromptSource/transformers-deepspeed/checkpoints/BLIP2_deepSpeed_t5xl_unfreeze_Qformer_Projection_LLM_QV_weight/checkpoint-1500
# model_name_or_path=${MODEL_DIR}${MODEL_NAME}
# remember to change the consponding tokenizer model

bs=6
eval_bs=15
lr=1e-4
dropout=0.1
epoch=4
seed=1234
do_train=False
do_test=True
do_valid=False
master_port=29505
evaluation_strategy=no
model_type=blip2
datasets=("flickr" "nocaps" "gqa" "miniimage" "nlvr2" "okvqa" "vqa" "wikiart")  # 添加其他数据集名称到这个数组

for dataset in "${datasets[@]}"; do
  echo "Running evaluation for dataset: $dataset" 
    EXPERIMENT_NAME=BLIP2_t5xl_fewshot_trained/eval_${dataset} 
    python run.py \
    --experiment_name ${EXPERIMENT_NAME} \
    --dataset_name ${DATASET_NAME} \
    --dataset_config_name None \
    --max_seq_length 512 \
    --overwrite_cache True \
    --pad_to_max_length True \
    --test_file /home/haozhezhao/Vision-PromptSource/prompt_data_6_11_few-shot_json/bilp2-prompt-allshot-multiinst_final_ver/${dataset}/\*/ \
    --do_train $do_train \
    --do_eval $do_valid \
    --do_predict $do_test \
    --per_device_train_batch_size ${bs} \
    --bf16 \
    --bf16_full_eval \
    --model_type $model_type \
    --save_total_limit 5 \
    --per_device_eval_batch_size ${eval_bs} \
    --processor_path ${MODEL_NAME} \
    --gradient_accumulation_steps 8 \
    --num_train_epochs ${epoch} \
    --output_dir checkpoints/${EXPERIMENT_NAME} \
    --overwrite_output_dir \
    --learning_rate ${lr} \
    --weight_decay 0.0005 \
    --seed ${seed} \
    --warmup_ratio 0.2 \
    --evaluation_strategy ${evaluation_strategy} \
    --remove_unused_columns False \
    --model_name_or_path ${model_name_or_path} \
    --use_fast_tokenizer True \
    --model_revision main \
    --eval_type val \
    --generation_max_length 256 \
    --do_full_training True \
    --using_instruct_qformer False \
    --run_name blip2-flan-t5-xl-fewhot_trained_eval_${dataset}  \
    --only_evaluate True \
    # --deepspeed config/deepspeed_config_zero3.json \
    # --load_best_model_at_end True \
    # --metric_for_best_model accuracy \
    # --greater_is_better True \
    # --eval_steps 50 \
    # --save_strategy steps \
    # --save_steps 200 \
    # --load_best_model_at_end \
    # --multiple_choice True
done
