from ast import Raise
from datasets import load_dataset, load_metric
import random
import json
import ast
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
import torch.nn as nn
import numpy as np
import logging
from datasets import load_metric
from collections import defaultdict

from sklearn.metrics import accuracy_score, zero_one_loss, precision_score, recall_score, f1_score, hamming_loss, roc_auc_score

logger = logging.getLogger(__name__)


class IntentDataset():
    def __init__(self, tokenizer, model_args, data_args, training_args, config):

        self.rng = random.Random(training_args.seed)

        self.tokenizer = tokenizer
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.config = config

        self.mask = self.tokenizer.mask_token
        self.pad = self.tokenizer.pad_token

        if self.data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            self.padding = "longest"

        if self.data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        if self.data_args.dataset_name == 'atis':
            # 载入文件，读取必要的label信息
            self.raw_dataset = load_dataset('json', data_files=self.data_args.train_file)
            self.predict_dataset = load_dataset('json', data_files=self.data_args.test_file)
            self.raw_dataset = self.raw_dataset.shuffle(training_args.seed)

            with open(self.data_args.label_file) as f:
                lines = f.read() ##Assume the sample file has 3 lines
                first_line = lines.split('\n', 1)[0]
                self.intent_label_set = ast.literal_eval(first_line)
            self.config.num_labels = len(self.intent_label_set)

            # 预处理
            self.raw_dataset = self.raw_dataset.map(
                self.preprocess_function_train,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Runing one tokenization"
            )
            self.predict_dataset = self.predict_dataset.map(
                self.preprocess_function_test,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Runing one tokenization"
            )
            self.raw_dataset = self.raw_dataset.map(
                self.preprocess_function_batched,
                batched=True,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Runing one tokenization"
            )
            self.predict_dataset = self.predict_dataset.map(
                self.preprocess_function_batched,
                batched=True,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Runing one tokenization"
            )

            # shuffle
            bond = int(self.raw_dataset["train"].num_rows // (1 / self.data_args.dev_rate))
            print(f"debug: bond {str(bond)}")

            # Split train and dev datasets
            self.eval_dataset = self.raw_dataset["train"].select(range(bond))
            self.train_dataset = self.raw_dataset["train"].select(range(bond, self.raw_dataset["train"].num_rows, 1))
            self.predict_dataset = self.predict_dataset["train"]
            self.predict_dataset = self.predict_dataset.remove_columns(self.intent_label_set)

            # 读取样本个数
            print(f"length raw dataset {self.raw_dataset.num_rows}")
            print(f"length train dataset {self.train_dataset.num_rows}")
            print(f"length eval dataset {self.eval_dataset.num_rows}")
            print(f"length test dataset {self.predict_dataset.num_rows}")
        elif self.data_args.dataset_name in ['snips', 'crosswoz', 'multiwoz']:
            self.train_dataset = load_dataset('json', data_files=self.data_args.train_file)
            self.eval_dataset = load_dataset('json', data_files=self.data_args.validation_file)
            self.predict_dataset = load_dataset('json', data_files=self.data_args.test_file)
            self.train_dataset = self.train_dataset.shuffle(training_args.seed)

            with open(self.data_args.label_file) as f:
                lines = f.read() ##Assume the sample file has 3 lines
                first_line = lines.split('\n', 1)[0]
                self.intent_label_set = ast.literal_eval(first_line)

            print(f"debug: intent_label_set {self.intent_label_set}")

            self.config.num_labels = len(self.intent_label_set)

            self.train_dataset = self.train_dataset.map(
                self.preprocess_function_train,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Runing one tokenization"
            )
            self.eval_dataset = self.eval_dataset.map(
                self.preprocess_function_train,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Runing one tokenization"
            )
            self.predict_dataset = self.predict_dataset.map(
                self.preprocess_function_test,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Runing one tokenization"
            )
            self.train_dataset = self.train_dataset.map(
                self.preprocess_function_batched,
                batched=True,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Runing one tokenization"
            )
            self.eval_dataset = self.eval_dataset.map(
                self.preprocess_function_batched,
                batched=True,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Runing one tokenization"
            )
            self.predict_dataset = self.predict_dataset.map(
                self.preprocess_function_batched,
                batched=True,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Runing one tokenization"
            )

            # Split train and dev datasets
            self.train_dataset = self.train_dataset["train"]
            self.eval_dataset = self.eval_dataset["train"]
            self.predict_dataset = self.predict_dataset["train"]
            self.predict_dataset = self.predict_dataset.remove_columns(self.intent_label_set)

            print(f"length train dataset {self.train_dataset.num_rows}")
            print(f"length eval dataset {self.eval_dataset.num_rows}")
            print(f"length test dataset {self.predict_dataset.num_rows}")

        if training_args.do_train:
            if data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))
        if training_args.do_eval:
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))
        if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
            if data_args.max_predict_samples is not None:
                self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))

        if data_args.pad_to_max_length:
            self.data_collator = default_data_collator
        elif training_args.fp16:
            self.data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

        self.metric = load_metric('accuracy')
        self.test_key = "accuracy"


    def preprocess_function_batched(self, examples):
        result = {
            "input_ids": self.tokenizer(examples["text"], padding=self.padding, max_length=self.max_seq_length, truncation=True)["input_ids"],
        }
        return result

    def preprocess_function_test(self, examples):
        result = {}
        result["label"] = []
        result["label_index"] = -1

        for intent in self.intent_label_set:
            if examples[intent] == 1:
                if result["label_index"] == -1:
                    result["label_index"] = len(result["label"])
                result["label"].append(1)
            elif examples[intent] == 0:
                result["label"].append(0)
            else:
                raise Exception("Invalid Value! {} {} {}".format(examples["text"], intent, result["label"]))

        return result

    def preprocess_function_train(self, examples):

        result = {}
        result["label"] = [0] * len(self.intent_label_set)

        if self.model_args.label == 'label':
            intent_name = examples["label"]
            intent_index = self.intent_label_set.index(intent_name)
            result["label"][intent_index] = 1
            result["label_index"] = intent_index

        elif self.model_args.label == 'intent':
            for intent in ["intent_0", "intent_1", "intent_2"]:
                intent_name = examples[intent]
                try:
                    intent_index = self.intent_label_set.index(intent_name)
                    result["label"][intent_index] = 1
                except:
                    pass

        elif self.model_args.label == 'upper':
            result["label"] = []
            for intent in self.intent_label_set:
                if examples[intent] == 1:
                    result["label"].append(1)
                elif examples[intent] == 0:
                    result["label"].append(0)
                else:
                    raise Exception("Invalid Value! {} {} {}".format(examples["text"], intent, result["label"]))

        return result

    def compute_metrics(self, p: EvalPrediction):

        preds = p.predictions
        labels = p.label_ids
        preds_acc = np.array(preds) >= 0.5

        preds_flat = p.predictions.reshape(-1)
        labels_flat = p.label_ids.reshape(-1)
        preds_acc_flat = preds_acc.reshape(-1)

        RocAucScore = roc_auc_score(y_true=labels, y_score=preds, multi_class='ovr', average='micro')
        ExactMatchRatio = accuracy_score(labels, preds_acc).item()
        ZeroOneLoss = zero_one_loss(labels, preds_acc).item()
        accuracy = accuracy_score(y_true=labels_flat, y_pred=preds_acc_flat).item()
        precision = precision_score(y_true=labels, y_pred=preds_acc, average='micro').item()
        recall = recall_score(y_true=labels, y_pred=preds_acc, average='micro').item()
        f1score = f1_score(labels, preds_acc, average='micro').item()
        hammingloss = hamming_loss(labels, preds_acc)
        positive_number = preds_acc.sum()
        positive_number_per_sample = preds_acc.sum() / len(preds)
        positive_rate = positive_number / preds_acc.size

        return {
            "RocAucScore": RocAucScore,
            "ExactMatchRatio": ExactMatchRatio,
            "ZeroOneLoss": ZeroOneLoss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1score": f1score,
            "hammingloss": hammingloss,
            "positive_number": positive_number,
            "positive_rate": positive_rate,
            "positive_number_per_sample": positive_number_per_sample
            }
