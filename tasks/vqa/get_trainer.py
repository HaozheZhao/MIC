'''
Author: JustBluce 972281745@qq.com
Date: 2022-11-24 09:02:57
LastEditors: JustBluce 972281745@qq.com
LastEditTime: 2023-02-18 10:57:49
FilePath: /ATIS/tasks/atis/get_trainer.py
Description: 设置trainer
'''
import logging
import random
from transformers import AdamW

from .dataset import FlickrDataset
from model.utils import get_model
from training.trainer_blip2 import BLIP2Trainer
from accelerate import Accelerator, DistributedType
from model.blip2.processing_blip_2 import Blip2Processor
from model.blip2.configuration_blip_2 import Blip2Config
import torch
logger = logging.getLogger(__name__)


def get_trainer(args):
    model_args, data_args, training_args = args

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    processor = Blip2Processor.from_pretrained(
        '/home/haozhezhao/models/blip2-flan-t5-xxl',
    )
    sp = ["图"]+[f"<image{i}>" for i in range(20)]
    sp = sp+processor.tokenizer.additional_special_tokens[len(sp):]
    processor.tokenizer.add_special_tokens({'additional_special_tokens':sp})

    config = Blip2Config.from_pretrained(
        # '/home/haozhezhao/models/blip2-flan-t5-xxl'
        model_args.model_name_or_path
    )
    
    dataset = FlickrDataset(processor, model_args, data_args, training_args, config)

    if training_args.do_train:
        for index in random.sample(range(len(dataset.train_dataset)), 1):
            logger.info(f"Sample keys {index} of the training set: {dataset.train_dataset[index].keys()}.")
            if not data_args.do_full_training:
                input_text = dataset.train_dataset[index]["input_text"]
                logger.info(f"Sample input_text {index} of the training set: {input_text}.")
                output_text = dataset.train_dataset[index]["output_text"]
                logger.info(f"Sample output_text {index} of the training set: {output_text}.")

            input_ids = dataset.train_dataset[index]["input_ids"]
            logger.info(f"Sample input_ids {index} of the training set: {input_ids}.")
            attention_mask = dataset.train_dataset[index]["attention_mask"]
            logger.info(f"Sample attention_mask {index} of the training set: {attention_mask}.")
            label = dataset.train_dataset[index]["label"]
            logger.info(f"Sample label {index} of the training set: {label}.")

            # pixel_values = dataset.train_dataset[index]["pixel_values"]
            # logger.info(f"Sample inputs shape {index} of the training set: {pixel_values.shape}.")
    # accelerator = Accelerator()
    # model = get_model(model_args, config).to(dtype=torch.bfloat16)
    config.text_config._from_model_config =False
    model = get_model(model_args, config)

    trainer = BLIP2Trainer(
        processor=processor,
        model=model,
        config=config,
        args=training_args,
        model_args=model_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        # test_dataset=dataset.eval_dataset if training_args.do_test else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        predict_dataset=dataset.predict_dataset, # 这个是用来测试的
        compute_metrics=dataset.compute_metrics,
        data_collator=dataset.data_collator,
    )

    return trainer, dataset.predict_dataset
