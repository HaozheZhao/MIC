from ast import Raise
from datasets import load_dataset, load_metric
import random
import json
import ast
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
import torch
import h5py
from PIL import Image
import torch.nn as nn
import numpy as np
import logging
from datasets import load_metric,Dataset,concatenate_datasets
from collections import defaultdict
from os.path import join
from sklearn.metrics import accuracy_score, zero_one_loss, precision_score, recall_score, f1_score, hamming_loss, roc_auc_score
from torch.nn.functional import pad
from typing import Any, Callable, Dict, List, NewType
from torchmetrics import BLEUScore
from glob import glob
InputDataClass = NewType("InputDataClass", Any)
from collections.abc import Mapping

logger = logging.getLogger(__name__)


class FlickrDataset():
    def __init__(self, processor, model_args, data_args, training_args, config):

        self.rng = random.Random(training_args.seed)

        self.processor = processor
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.config = config

        if self.data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            self.padding = "longest"

        if self.data_args.max_seq_length > processor.tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({processor.tokenizer.model_max_length}). Using max_seq_length={processor.tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, processor.tokenizer.model_max_length)
        if not self.data_args.do_full_training:
            if self.data_args.dataset_name in ['flickr']:
                self.train_dataset = load_dataset('json', data_files=self.data_args.train_file)
                self.eval_dataset = load_dataset('json', data_files=self.data_args.validation_file)
                self.predict_dataset = load_dataset('json', data_files=self.data_args.test_file)

                self.train_dataset = self.train_dataset.shuffle(training_args.seed)

                print(f"debug: train_dataset {self.train_dataset}")

                self.train_dataset = self.train_dataset.map(
                    self.preprocess_function,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Runing one tokenization"
                )
                # self.eval_dataset = self.train_dataset
                # self.predict_dataset = self.train_dataset
                self.eval_dataset = self.eval_dataset.map(
                    self.preprocess_function,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Runing one tokenization"
                )
                self.predict_dataset = self.predict_dataset.map(
                    self.preprocess_function,
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

                print(f"length train dataset {self.train_dataset.num_rows}")
                print(f"length eval dataset {self.eval_dataset.num_rows}")
                print(f"length test dataset {self.predict_dataset.num_rows}")
        else:
                self.train_dataset = self.load_dataset_from_arrow(data_files=self.data_args.train_file)
                self.eval_dataset = self.load_dataset_from_arrow(data_files=self.data_args.validation_file)
                # self.eval_dataset =Dataset.from_file(join(self.data_args.validation_file,"bilp2-temp-val-0.arrow"))
                # self.predict_dataset =Dataset.from_file(join(self.data_args.test_file,"bilp2-temp-test-0.arrow"))
                self.predict_dataset = self.load_dataset_from_arrow(data_files=self.data_args.test_file)

                self.train_dataset = self.train_dataset.shuffle(training_args.seed)

        if training_args.do_train:
            if data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))
        if training_args.do_eval:
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))
        if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
            if data_args.max_predict_samples is not None:
                self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))

        self.data_collator = self.collector
        self.metric = load_metric('accuracy')
        self.test_key = "accuracy"
    def load_dataset_from_arrow(self,data_files):
        files = glob(join(data_files,"bilp2*[0-9].arrow"))
        ds = concatenate_datasets([Dataset.from_file(score) for score in files ])
        return ds
    def preprocess_function_batched(self, examples):
        result= self.processor.tokenizer(examples["input_text"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        # if self.training_args.few_shot:
        #     result['label'] = self.processor.tokenizer(examples["output_text"], padding=self.padding, max_length=5, truncation=True)["input_ids"]
        # else:
        #     result['label'] = self.processor.tokenizer(examples["output_text"], padding=self.padding, max_length=32, truncation=True)["input_ids"]
        result['label'] = self.processor.tokenizer(examples["output_text"], padding=self.padding, max_length=32, truncation=True)["input_ids"]
        
 
        return result
    def read_image(self, postfix,img_path):
        if postfix == 'png':
            image = Image.open(join("/home/haozhezhao/Vision-PromptSource",img_path))
        elif postfix == 'h5':
            image = h5py.File(join("/home/haozhezhao/Vision-PromptSource",img_path), 'r')
        else:
            image = Image.open(join("/home/haozhezhao/Vision-PromptSource", img_path))
        return image
    def preprocess_function(self, examples):
        # if self.training_args.multiple_choice:
        #     candidates = examples["input_text"][examples["input_text"].index('Options: ')+9:].split('\n')
        #     content =  examples["input_text"][:examples["input_text"].index('Options: ')]
        result = {}
        # result["input_text"] = examples["input_text"]
        result["output_text"] = examples["output_text"]
        flag = isinstance(examples["input_image"],list)
        result["pixel_values"] = []
        if flag:
            postfix = examples["input_image"][0][1:].split('.')[-1]
            for img_path in examples["input_image"]:
                img_path = img_path[2:] if img_path[0] == '.' else img_path
                img = self.read_image(postfix,img_path)
                result["pixel_values"].append(self.processor(images = img)["pixel_values"][0])
        else:
            postfix = examples["input_image"][1:].split('.')[-1]
            img_path = img_path[1:] if img_path[0] == '.' else img_path
            img = self.read_image(postfix,img_path)
            result["pixel_values"].append(self.processor(images = img)["pixel_values"][0])


        return result
    def padd_images(self, image, max_length):
        image = torch.tensor(image)
        mask = torch.zeros(max_length).bool()
        pad_len = max_length - image.shape[0]
        mask[:image.shape[0]] = True
        image = pad(image,(0,0,0,0,0,0,0,pad_len)) # padding behind the first dim
        return image,mask


    def collector(self, features: List[InputDataClass]):
        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        first = features[0]
        batch = {}

        if "label" in first and first["label"] is not None:
            label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
            dtype = torch.long if isinstance(label, List) else torch.float
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
        elif "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
                batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)
        ignored_keys = ['input_text', 'input_image', 'output_text', 'output_image','pixel_values']
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str) and k not in ignored_keys :
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
            elif k == 'pixel_values':
                max_image_length = max([len(f[k]) for f in features])
                image_list=[]
                mask_list= []
                for f in features:
                    image,img_mask =self.padd_images(f[k],max_image_length)
                    image_list.append(image)
                    mask_list.append(img_mask)
                batch[k] = torch.stack(image_list)
                batch['img_mask'] = torch.stack(mask_list)
        # for each in batch:
        #     if isinstance(batch[each], torch.Tensor) and (batch[each].dtype == torch.float32):
        #         batch[each] = batch[each].to(dtype= torch.bfloat16)
        return batch

    def compute_metrics(self, p: EvalPrediction):

        preds = p.predictions
        labels = p.label_ids
        preds[preds==-100] = 0
        bleu = BLEUScore()
        bleu_result = []
        accuracy = 0
        dict_return={}
        for i,pred in enumerate(preds):
            p_token = self.processor.tokenizer.decode(pred,skip_special_tokens=True)
            label_token = self.processor.tokenizer.decode(labels[i],skip_special_tokens=True)
            # if self.training_args.multiple_choice:
            #     l = label_token.split(' ')[0]
            #     if l in p_token:
            #         accuracy+=1
            #     bleu_result.append(bleu([p_token],[[label_token]]).item())
            # else:
            #     if p_token == label_token:
            #         accuracy+=1
            l = label_token.split(' ')[0]
            if "option" in l:
                if l in p_token:
                    accuracy+=1
            else:
                if p_token == label_token:
                    accuracy+=1

        # if self.training_args.multiple_choice:
        #     dict_return ={
        #     # "BleuScore": bleu_result,
        #     'Avg_BleuScore':np.array(bleu_result).mean(),
        #     }
        dict_return['accuracy'] = accuracy/len(preds)
        return dict_return
