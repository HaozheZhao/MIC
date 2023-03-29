'''
Author: JustBluce 972281745@qq.com
Date: 2022-11-24 09:02:57
LastEditors: JustBluce 972281745@qq.com
LastEditTime: 2023-01-11 09:48:51
FilePath: /ATIS/tasks/atis/get_trainer.py
Description: 设置trainer
'''
import logging
import random

from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from .dataset import IntentDataset
from model.utils import get_model
from training.trainer_intent import IntentTrainer

logger = logging.getLogger(__name__)


def get_trainer(args):
    model_args, data_args, training_args = args

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
    )
    dataset = IntentDataset(tokenizer, model_args, data_args, training_args, config)

    if training_args.do_train:
        for index in random.sample(range(len(dataset.train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {dataset.train_dataset[index]}.")

    model = get_model(model_args, config)

    trainer = IntentTrainer(
        model=model,
        config=config,
        args=training_args,
        model_args=model_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        predict_dataset=dataset.predict_dataset, # 这个是用来测试的
        compute_metrics=dataset.compute_metrics,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        intent_label_set=dataset.intent_label_set
    )

    return trainer, dataset.predict_dataset
