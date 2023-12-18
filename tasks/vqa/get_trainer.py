import logging
import random
from transformers import AdamW

from .dataset import FlickrDataset
from model.utils import get_model
from training.trainer_blip2 import BLIP2Trainer
from training.trainer_instructblip2 import InstructBLIP2Trainer
from model.instructblip.processing_instructblip import InstructBlipProcessor
from model.instructblip.configuration_instructblip import InstructBlipConfig

from accelerate import Accelerator, DistributedType
from model.blip2.processing_blip_2 import Blip2Processor
from model.blip2.configuration_blip_2 import Blip2Config
import torch
logger = logging.getLogger(__name__)


def get_trainer(args):
    model_args, data_args, training_args = args

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    model_type = model_args.model_type
    backbone_model = model_args.backbone_model
    if model_type == 'blip2':
        config = Blip2Config.from_pretrained(
                    model_args.model_name_or_path
            )
    elif model_type == 'instructblip':
        config = InstructBlipConfig.from_pretrained(
            model_args.model_name_or_path
        )
    if backbone_model == 'vicuna':
        config.text_config.pad_token_id = 32000
    config.text_config.max_sequence_length = data_args.max_seq_length
    model = get_model(model_args, config)
    if training_args.full_bf16_training:
        model = model.to(dtype=torch.bfloat16)
        
    if model_args.image_place_holder is not None:
        image_place_holder = model_args.image_place_holder 
    else:
        image_place_holder = "图" if model_args.backbone_model == 'flan-t5' else "<visual_embedding>"
    print(f"image_place_holder: {image_place_holder}")
    processor_path =  model_args.processor_path if model_args.processor_path is not None else model_args.model_name_or_path
    if model_type == 'blip2':
        processor = Blip2Processor.from_pretrained(
            processor_path,
        )
        if backbone_model == 'flan-t5':
            sp = [image_place_holder]+[f"<image{i}>" for i in range(20)]
            sp = sp+processor.tokenizer.additional_special_tokens[len(sp):]
            processor.tokenizer.add_special_tokens({'additional_special_tokens':sp})
        else: # opt
            sp = [image_place_holder]+[f"<image{i}>" for i in range(20)]
            processor.tokenizer.add_special_tokens({'additional_special_tokens':sp})
            model.language_model.resize_token_embeddings(len(processor.tokenizer))

    elif model_type == 'instructblip':
        processor = InstructBlipProcessor.from_pretrained(
            processor_path
        )
     
        if backbone_model == 'flan-t5':
            sp = [image_place_holder]+[f"<image{i}>" for i in range(20)]
            sp = sp+processor.tokenizer.additional_special_tokens[len(sp):]
            processor.tokenizer.add_special_tokens({'additional_special_tokens':sp})
        else: #vicuna
            sp = [image_place_holder]+[f"<image{i}>" for i in range(20)]
            processor.tokenizer.add_special_tokens({'additional_special_tokens':sp})
            processor.qformer_tokenizer.add_special_tokens({'additional_special_tokens':sp})
            model.language_model.resize_token_embeddings(len(processor.tokenizer))
            model.language_model.model.embed_tokens.weight.requires_grad=True
        # bert tokenizer for q-former
        processor.qformer_tokenizer.add_special_tokens({'additional_special_tokens':sp})
        if model.qformer.embeddings.word_embeddings.weight.shape[0] != len(processor.qformer_tokenizer):
            model.qformer.resize_token_embeddings(len(processor.qformer_tokenizer))
    config.text_config._from_model_config =False
    
    dataset = FlickrDataset(processor, model_args, data_args, training_args, config)
    special_visual_token_id = dataset.special_visual_token_id
    model_args.special_visual_token_id = special_visual_token_id
    # if training_args.do_train:
        # for index in random.sample(range(len(dataset.train_dataset)), 1):
        #     logger.info(f"Sample keys {index} of the training set: {dataset.train_dataset[index].keys()}.")
        #     if not data_args.done_preprocess:
        #         input_text = dataset.train_dataset[index]["input_text"]
        #         logger.info(f"Sample input_text {index} of the training set: {input_text}.")
        #         output_text = dataset.train_dataset[index]["output_text"]
        #         logger.info(f"Sample output_text {index} of the training set: {output_text}.")

        #     input_ids = dataset.train_dataset[index]["input_ids"]
        #     logger.info(f"Sample input_ids {index} of the training set: {input_ids}.")
        #     attention_mask = dataset.train_dataset[index]["attention_mask"]
        #     logger.info(f"Sample attention_mask {index} of the training set: {attention_mask}.")
        #     label = dataset.train_dataset[index]["label"]
        #     logger.info(f"Sample label {index} of the training set: {label}.")

    if model_type == 'blip2':
        trainer = BLIP2Trainer(
            processor=processor,
            model=model,
            config=config,
            args=training_args,
            model_args=model_args,
            train_dataset=dataset.train_dataset if training_args.do_train else None,
            eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
            predict_dataset=dataset.predict_dataset, 
            compute_metrics=dataset.compute_metrics,
            data_collator=dataset.data_collator,
        )
    elif model_type == 'instructblip':
            trainer = InstructBLIP2Trainer(
            processor=processor,
            model=model,
            config=config,
            args=training_args,
            model_args=model_args,
            train_dataset=dataset.train_dataset if training_args.do_train else None,
            eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
            predict_dataset=dataset.predict_dataset, 
            compute_metrics=dataset.compute_metrics,
            data_collator=dataset.data_collator,
        )

    return trainer, dataset.predict_dataset
