from ast import Raise
import base64
from io import BytesIO
from datasets import load_dataset, load_metric
import random
import json
import copy
import ast
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
import torch
import h5py
from PIL import Image
import os
import torch.nn as nn
import numpy as np
import logging
from datasets import load_metric,Dataset,concatenate_datasets
import datasets
from collections import defaultdict
from os.path import join
from sklearn.metrics import accuracy_score, zero_one_loss, precision_score, recall_score, f1_score, hamming_loss, roc_auc_score
from torch.nn.functional import pad
from typing import Any, Callable, Dict, List, NewType
from torchmetrics import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from glob import glob
from pycocoevalcap.bleu.bleu_scorer import BleuScorer

from pycocoevalcap.cider.cider_scorer import CiderScorer
InputDataClass = NewType("InputDataClass", Any)
from collections.abc import Mapping
import os
os.environ['TOKENIZERS_PARALLELISM'] = "false"
logger = logging.getLogger(__name__)
IGNORE_INDEX=-100
MASK_INDEX =1
datasets.config.IN_MEMORY_MAX_SIZE = 300 *1024 *1024 *1024

class FlickrDataset():
    def __init__(self, processor, model_args, data_args, training_args, config):

        self.rng = random.Random(training_args.seed)

        self.processor = processor
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.config = config
        self.model_type = model_args.model_type
        self.label_max_length = data_args.label_max_length
        self.save_number = 0
        if model_args.image_place_holder is not None:
            self.image_place_holder = model_args.image_place_holder 
        else:
            self.image_place_holder = "图" if model_args.backbone_model == 'flan-t5' else "<visual_embedding>"
        self.special_visual_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.image_place_holder) 
        self.replace_token = "".join([self.image_place_holder]*32)
        print('self.special_visual_token_id',self.special_visual_token_id) 

        if self.data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            self.padding = "longest"

        if self.data_args.max_seq_length > processor.tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({processor.tokenizer.model_max_length}). Using max_seq_length={processor.tokenizer.model_max_length}."
            )
        # self.max_seq_length = data_args.max_seq_length if data_args.max_seq_length is not None else processor.tokenizer.model_max_length
        self.max_seq_length = processor.tokenizer.model_max_length

        
        if self.data_args.only_evaluate:
            self.predict_dataset = self.load_evaluate_dataset_from_arrow(self.data_args.test_file)
            self.eval_dataset  =None
            self.train_dataset  =None
        else:
            if not self.data_args.done_preprocess: 
                # do your own preprocessing
                if not self.data_args.training_preprocess:
                    if self.data_args.dataset_name in ['flickr']:
                        if self.data_args.load_datatype is not None:
                            self.train_dataset = load_dataset(self.data_args.load_datatype, data_files=self.data_args.train_file, streaming=True,split="train")
                            self.eval_dataset = load_dataset(self.data_args.load_datatype, data_files=self.data_args.validation_file,split="train" )
                            self.predict_dataset = load_dataset(self.data_args.load_datatype, data_files=self.data_args.test_file,split="train")
                        else:
                            self.train_dataset = load_dataset(data_files=self.data_args.train_file, streaming=True,split="train")
                            self.eval_dataset = load_dataset(data_files=self.data_args.validation_file,split="train")
                            self.predict_dataset = load_dataset(data_files=self.data_args.test_file,split="train")

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

                        print(f"debug: train_dataset {self.train_dataset}")

                        self.train_dataset = self.train_dataset.map(
                            self.preprocess_function_batched,
                            batched=True,
                            # load_from_cache_file=not self.data_args.overwrite_cache,
                            # desc="Runing one tokenization"
                        )
                        self.eval_dataset = self.eval_dataset.map(
                            self.preprocess_function_batched,
                            batched=True,
                            batch_size=100,
                            # load_from_cache_file=not self.data_args.overwrite_cache,
                            # desc="Runing one tokenization"
                        )
                        self.predict_dataset = self.predict_dataset.map(
                            self.preprocess_function_batched,
                            batched=True,
                            batch_size=100,
                            # load_from_cache_file=not self.data_args.overwrite_cache,
                            # desc="Runing one tokenization"
                        )

                        # Split train and dev datasets
                        self.train_dataset = self.train_dataset.with_format("torch")
                        self.eval_dataset = self.eval_dataset.with_format("torch")
                        self.predict_dataset = self.predict_dataset.with_format("torch")

                        # print(f"length train dataset {self.train_dataset.num_rows}")
                        print(f"length eval dataset {self.eval_dataset.num_rows}")
                        print(f"length test dataset {self.predict_dataset.num_rows}")
                else:
                    self.train_dataset = load_dataset(self.data_args.load_datatype, data_files=self.data_args.train_file,split="train" )
                    self.eval_dataset = load_dataset(self.data_args.load_datatype, data_files=self.data_args.validation_file,split="train" )
                    self.predict_dataset = load_dataset(self.data_args.load_datatype, data_files=self.data_args.test_file,split="train" )
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
                    self.eval_dataset = self.eval_dataset.map(
                        self.preprocess_function_eval_batched,
                        batched=True,
                        batch_size=100,
                        # load_from_cache_file=not self.data_args.overwrite_cache,
                        # desc="Runing one tokenization"
                        )
                    self.predict_dataset = self.predict_dataset.map(
                        self.preprocess_function_eval_batched,
                        batched=True,
                        batch_size=100,
                        # load_from_cache_file=not self.data_args.overwrite_cache,
                        # desc="Runing one tokenization"
                        )
            else:
                self.train_dataset = self.load_dataset_from_arrow(data_files=self.data_args.train_file)
                self.eval_dataset = self.load_dataset_from_arrow(data_files=self.data_args.validation_file )
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




        
        self.data_collator = self.collector if self.data_args.done_preprocess else self.collector_without_preprocess
        self.metric = load_metric('accuracy')
        self.test_key = "accuracy"

    def load_evaluate_dataset_from_arrow(self,eval_dataset_dir):
        eval_datadir = glob(os.path.join(eval_dataset_dir, "mmicl*.arrow"))
        eval_ds= concatenate_datasets([Dataset.from_file(score) for score in eval_datadir ])
        return eval_ds

    def load_instruct_dataset_from_arrow(self,dataset_folder):  # load the preprocessed data
        train_files = []
        test_files = []
        val_files = []

        for dataset_name in os.listdir(dataset_folder):
            dataset_path = os.path.join(dataset_folder, dataset_name)
            
            for dir in os.listdir(dataset_path):
                folder = os.path.join(dataset_path, dir)
                if 'train' in folder:
                    train_files.extend(glob(os.path.join(folder, "mmicl*.arrow")))
                elif 'test' in folder:
                    test_files.extend(glob(os.path.join(folder, "mmicl*.arrow")))
                elif 'val' in folder:
                    val_files.extend(glob(os.path.join(folder, "mmicl*.arrow")))
        train_ds = concatenate_datasets([Dataset.from_file(score) for score in train_files ])
        test_ds = concatenate_datasets([Dataset.from_file(score) for score in test_files ])
        val_ds = concatenate_datasets([Dataset.from_file(score) for score in val_files ])
        return train_ds,val_ds,test_ds

    def load_dataset_from_arrow(self,data_files):
        files = glob(join(data_files,"mmicl*[0-9].arrow"))
        ds = concatenate_datasets([Dataset.from_file(score) for score in files ])
        return ds
    def load_base64_image(self,base64_str):

        image = base64.b64decode(base64_str)
        image = BytesIO(image)
        image = Image.open(image)

        return image
    
    def preprocess_function_batched(self, examples):
        input_text = [each.replace('图',self.replace_token) for each in examples["input_text"]]
        output_text = examples["output_text"]
        if self.model_args.backbone_model == 'vicuna':
            self.processor.tokenizer.padding_side = "right"
            self.processor.tokenizer.truncation_side = 'left'
            re = self.processor.tokenizer(
                input_text,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                max_length=self.max_seq_length,
            )
            self.processor.tokenizer.truncation_side = 'right'
            out = self.processor.tokenizer(
                output_text,
                padding="longest",
                return_tensors="pt",
                truncation=True,
                max_length=self.label_max_length)
            re, input_part_targets_len = self.concat_text_input_output(
            re['input_ids'],
            re['attention_mask'],
            out['input_ids'],
            out['attention_mask'],
            )

            # do not apply loss to the padding
            targets = copy.deepcopy(re['input_ids'])
            targets[targets == self.processor.tokenizer.pad_token_id] = -100


            # do not apply loss to the text input (i.e., instruction)
            for i, l in enumerate(input_part_targets_len):
                targets[i][:l] = -100

            result= {
                'input_ids': re['input_ids'],
                'attention_mask': re['attention_mask'],
                'label': targets,
                'pixel_values': [],
            }
        else:
            result= self.processor.tokenizer(input_text, padding=self.padding, max_length=self.max_seq_length, truncation=True)
            result['label'] = self.processor.tokenizer(examples["output_text"], padding=self.padding, max_length=self.label_max_length, truncation=True)["input_ids"]
            result["pixel_values"] = []

        for img in examples["input_image"]:
            images=[]
            for img_path in img:
                img_path = img_path[2:] if img_path[0] == '.' else img_path
                pil_img = self.read_image(img_path)
                images.append(self.processor(images = pil_img)["pixel_values"][0])
            result["pixel_values"].append(images)

    def preprocess_function_eval_batched(self, examples):
        input_text = [each.replace('图',self.replace_token) for each in examples["input_text"]]
        output_text = examples["output_text"]
        if self.model_args.backbone_model == 'vicuna':
            self.processor.tokenizer.padding_side = "right"
            self.processor.tokenizer.truncation_side = 'left'
            re = self.processor.tokenizer(
                input_text,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                max_length=self.max_seq_length,
            )
            self.processor.tokenizer.truncation_side = 'right'
            out = self.processor.tokenizer(
                output_text,
                padding="longest",
                return_tensors="pt",
                truncation=True,
                max_length=self.label_max_length)

            out['input_ids'][out['attention_mask']] = IGNORE_INDEX

            result= {
                'input_ids': re['input_ids'].to(torch.long),
                'attention_mask': re['attention_mask'].to(torch.long),
                'labels': out['input_ids'].to(torch.long),
                'pixel_values': [],
            }
        else:
            result= self.processor.tokenizer(input_text, padding=self.padding, max_length=self.max_seq_length,return_tensors="pt", truncation=True)
            label = self.processor.tokenizer(examples["output_text"], padding=self.padding, max_length=self.label_max_length, return_tensors="pt",truncation=True)
            result["pixel_values"] = []
            label["input_ids"][label["attention_mask"]]= IGNORE_INDEX 
            result['labels'] = label["input_ids"]

        for img in examples["input_image"]:
            images=[]
            for img_path in img:
                img_path = img_path[2:] if img_path[0] == '.' else img_path
                pil_img = self.read_image(img_path)
                images.append(self.processor(images = pil_img)["pixel_values"][0])
            result["pixel_values"].append(images)

        max_image_length = 12
        image_list=[]
        mask_list= []
        for f in result["pixel_values"]:
            image,img_mask =self.padd_images(f,max_image_length)
            image_list.append(image)
            mask_list.append(img_mask)
        result['pixel_values'] = torch.stack(image_list)
        result['img_mask'] = torch.stack(mask_list)
        
        return result    
 
    def read_image(self,img_path,postfix='None'):
        if self.data_args.data_dir is not None:
            img_path = join(self.data_args.data_dir,img_path)
        if postfix == 'png':
            image = Image.open(img_path)
        elif postfix == 'h5':
            image = h5py.File(img_path, 'r')
        else:
            image = Image.open(img_path)
        return image
    

    
    def pad_features(self, features,feature_name,dtype=torch.long,pad_token_id=32000, padding= 'pad_2_max_length'):

        # Step 1: Create a list of label tensors
        if isinstance(features[0][feature_name],torch.Tensor):
            padded_labels = [f[feature_name][0] for f in features]
        elif isinstance(features[0][feature_name],np.ndarray):
            padded_labels = [torch.tensor(f[feature_name][0]) for f in features]
        else:
            padded_labels = [torch.tensor(np.array(f[feature_name][0])) for f in features]
        # Step 2: Get the max length of the label tensors
        max_length = max(len(f[feature_name][0]) for idx,f in enumerate(features)) if padding == 'pad_2_max_length' else self.max_seq_length
        if max_length > self.max_seq_length:
            max_length = self.max_seq_length
        # Step 3: Pad the label tensors
        padded_labels = [pad(label, (0, max_length - len(label)), value=pad_token_id) for label in padded_labels]
        padded_labels = torch.stack(padded_labels).to(dtype)
        return padded_labels
    def replicate_values(self,tensor, indices, num_replications):
        new_tensor = tensor.copy()
        for i,index in enumerate(indices):
            value_to_replicate = tensor[index]
            idx = index+i*num_replications
            replicated_values = np.repeat(value_to_replicate, num_replications)
            new_tensor = np.insert(new_tensor, idx+1, replicated_values)
        return new_tensor


    def padding_input_ids(self,feature,sp_token,key='input_ids',num_replications=31,dtype = torch.long):
        pad_input_ids=[]
        length =[]
        diff_length =[]
        for each in feature:
            o_tensor = each[key][0]
            if not isinstance(o_tensor, np.ndarray):
                o_tensor = np.array(o_tensor)
            target_indices = np.where(o_tensor == sp_token)[0]
            new_tensor = self.replicate_values(o_tensor, target_indices, num_replications)
            length.append(len(new_tensor))
            diff_length.append(len(new_tensor)-len(o_tensor))
            pad_input_ids.append(torch.tensor(new_tensor))
        max_length = max(length)
        pad_ids = torch.stack([pad(ids, (0, max_length - length[idx]), value=self.processor.tokenizer.pad_token_id) for idx,ids in enumerate(pad_input_ids)])
        return pad_ids,diff_length
    
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
            if self.model_args.backbone_model == 'flan-t5':
                batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
            else:
                batch["labels"] = self.pad_features(features,'label',dtype,self.processor.tokenizer.pad_token_id)
        elif "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                if self.model_args.backbone_model == 'flan-t5':
                    batch["labels"] = torch.stack([f["label_ids"] for f in features])
                else:
                    batch["labels"] = self.pad_features(features,'label_ids',torch.long,self.processor.tokenizer.pad_token_id)
            else:
                dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
                if self.model_args.backbone_model == 'flan-t5':
                    batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)
                else:
                    batch["labels"] = self.pad_features(features,'label_ids',dtype,self.processor.tokenizer.pad_token_id)
        batch["labels"][torch.where( batch["labels"]==self.processor.tokenizer.pad_token_id)]= IGNORE_INDEX 
        ignored_keys = ['input_text', 'input_image', 'output_text', 'output_image','pixel_values']
        for k, v in first.items():
            if self.model_args.backbone_model == 'flan-t5':
                # expend input tokens to suit for vision token
                if k not in ("label", "label_ids") and v is not None and not isinstance(v, str) and k not in ignored_keys :
                    if isinstance(v, torch.Tensor):
                        batch[k] = torch.stack([f[k] for f in features])
                    elif isinstance(v, np.ndarray):
                        batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                    else:
                        batch[k] = torch.tensor([f[k] for f in features])

            else:
                # expend input tokens to suit for vision token
                if k not in ("label", "label_ids") and v is not None and not isinstance(v, str) and k not in ignored_keys :
                    pad_token_id = self.processor.tokenizer.pad_token_id if k !='attention_mask' else 0
                    batch[k] = self.pad_features(features,k,pad_token_id = pad_token_id)
            if k == 'pixel_values':
                max_image_length = max([len(f[k]) for f in features])
                image_list=[]
                mask_list= []
                for f in features:
                    image,img_mask =self.padd_images(f[k],max_image_length)
                    image_list.append(image)
                    mask_list.append(img_mask)
                batch[k] = torch.stack(image_list)
                batch['img_mask'] = torch.stack(mask_list)

        if self.model_type=='instructblip' and self.training_args.using_instruct_qformer:
            q_former_input = self.processor.tokenizer.batch_decode(batch['input_ids'])
            q_former_input = [inputs.replace('<pad>',"").replace('</s>',"") for inputs in q_former_input]
            q_former_re = self.processor.qformer_tokenizer(q_former_input, padding=self.padding, max_length=self.max_seq_length, truncation=True,return_tensors="pt")
            batch['qformer_input_ids'] = q_former_re['input_ids']
            batch['qformer_attention_mask'] = q_former_re['attention_mask']
        if self.model_args.backbone_model == 'vicuna':
            img_per_row = torch.sum(batch['input_ids'] == self.special_visual_token_id, dim=1)
            pad_length = 32*max(img_per_row)
            batch['input_ids'] = pad(batch['input_ids'], (0, pad_length), value=self.processor.tokenizer.pad_token_id)
            batch['attention_mask'] = pad(batch['attention_mask'], (0, pad_length), value=0)

            
        batch['sp_token'] = self.special_visual_token_id
        if self.training_args.full_bf16_training:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and k in ['pixel_values']:
                    batch[k] = v.to(dtype=torch.bfloat16)
        return batch
    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i,each in enumerate(input_ids):

            this_input_ones = sum(input_atts[i])
            input_part_targets_len.append(this_input_ones)
            if isinstance(each,torch.Tensor):
                llm_tokens['input_ids'].append(
                    torch.concat([
                        each[:this_input_ones],
                        output_ids[i][1:],
                        each[this_input_ones:]
                    ])
                )
                llm_tokens['attention_mask'].append(
                    torch.concat([
                        input_atts[i][:this_input_ones],
                        output_atts[i][1:],
                        input_atts[i][this_input_ones:]
                    ])
                )
            else:
                llm_tokens['input_ids'].append(
                    np.concatenate([
                        each[:this_input_ones],
                        output_ids[i][1:],
                        each[this_input_ones:]
                    ])
                )
                llm_tokens['attention_mask'].append(
                    np.concatenate([
                        input_atts[i][:this_input_ones],
                        output_atts[i][1:],
                        input_atts[i][this_input_ones:]
                    ])
                )
        if isinstance(llm_tokens['input_ids'][0],torch.Tensor):
            llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
            llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        else:
            llm_tokens['input_ids'] = np.stack(llm_tokens['input_ids'])
            llm_tokens['attention_mask'] = np.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len
    def collector_without_preprocess(self, features: List[InputDataClass]):

        if self.data_args.training_preprocess:
            if 'input_ids' in features[0]:
                for k, v in first.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = torch.stack([f[k] for f in features])
                    elif isinstance(v, np.ndarray):
                        batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                    else:
                        batch[k] = torch.tensor([f[k] for f in features])            
            else:
                input_text = [each["input_text"].replace('图',self.replace_token) for each in features]
                output_text = [each["output_text"] for each in features]
                input_image = [each["input_image"] for each in features]
                if self.model_args.backbone_model == 'vicuna':
                    self.processor.tokenizer.padding_side = "right"
                    self.processor.tokenizer.truncation_side = 'left'
                    re = self.processor.tokenizer(
                        input_text,
                        padding="longest",
                        truncation=True,
                        return_tensors="pt",
                        max_length=self.max_seq_length,
                    )
                    self.processor.tokenizer.truncation_side = 'right'
                    out = self.processor.tokenizer(
                        output_text,
                        padding="longest",
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.label_max_length)
                    re, input_part_targets_len = self.concat_text_input_output(
                    re['input_ids'],
                    re['attention_mask'],
                    out['input_ids'],
                    out['attention_mask'],
                    )
                    # do not apply loss to the padding
                    targets = copy.deepcopy(re['input_ids'])
                    targets[targets == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX


                    # do not apply loss to the text input (i.e., instruction)
                    for i, l in enumerate(input_part_targets_len):
                        targets[i][:l] = IGNORE_INDEX

                    batch= {
                        'input_ids': re['input_ids'].to(torch.long),
                        'attention_mask': re['attention_mask'].to(torch.long),
                        'labels': targets.to(torch.long),
                        'pixel_values': [],
                    }
                else:
                    batch= self.processor.tokenizer(input_text, padding=self.padding, max_length=self.max_seq_length, truncation=True,return_tensors="pt")
                    batch['labels'] = self.processor.tokenizer(output_text, padding=self.padding, max_length=self.label_max_length, truncation=True,return_tensors="pt")["input_ids"]
                    batch["labels"][torch.where( batch["labels"]==self.processor.tokenizer.pad_token_id)]= IGNORE_INDEX 
                    batch["pixel_values"] = []


                for img in input_image:
                    images=[]
                    for img_path in img:
                        img_path = img_path[2:] if img_path[0] == '.' else img_path
                        pil_img = self.read_image(img_path)
                        images.append(self.processor(images = pil_img)["pixel_values"][0])
                    batch["pixel_values"].append(images)

                max_image_length = 12
                image_list=[]
                mask_list= []
                for f in batch["pixel_values"]:
                    image,img_mask =self.padd_images(f,max_image_length)
                    image_list.append(image)
                    mask_list.append(img_mask)
                batch['pixel_values'] = torch.stack(image_list)
                batch['img_mask'] = torch.stack(mask_list)
        else:
            if not isinstance(features[0], Mapping):
                features = [vars(f) for f in features]
            first = features[0]
            batch = {}

            if "label" in first and first["label"] is not None:
                label = first["label"][0].item() if isinstance(first["label"], torch.Tensor) else first["label"][0]
                dtype = torch.long  if isinstance(label, int) else torch.float
                batch["labels"] = torch.stack([f["label"] for f in features]).to(dtype)
            elif "label_ids" in first and first["label_ids"] is not None:
                if isinstance(first["label_ids"], torch.Tensor):
                    batch["labels"] = torch.stack([f["label_ids"] for f in features])
                else:
                    dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
                    batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)
            batch["labels"][torch.where( batch["labels"]==self.processor.tokenizer.pad_token_id)]= IGNORE_INDEX 
            ignored_keys = ['input_text', 'input_image', 'output_text', 'output_image','pixel_values']
            for k, v in first.items():
                if k not in ("label", "label_ids") and v is not None and not isinstance(v, str) and k not in ignored_keys :
                    if isinstance(v, torch.Tensor):
                        batch[k] = torch.stack([f[k] for f in features])
                    elif isinstance(v, np.ndarray):
                        batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                    else:
                        batch[k] = torch.tensor([f[k] for f in features])
                if k == 'pixel_values':
                    # max_image_length = max([len(f[k]) for f in features])
                    max_image_length = 8
                    image_list=[]
                    mask_list= []
                    for f in features:
                        image,img_mask =self.padd_images(f[k],max_image_length)
                        image_list.append(image)
                        mask_list.append(img_mask)
                    batch[k] = torch.stack(image_list)
                    batch['img_mask'] = torch.stack(mask_list)
        for k, v in batch.items():
            if not isinstance(v, torch.Tensor):
                batch[k] = torch.tensor(v)
        if self.training_args.full_bf16_training:
            batch['pixel_values'] = batch['pixel_values'].to(dtype=torch.bfloat16)
        return batch

    def save_pred_label(self,p_token_batch,label_token_batch,output_dir):
        with open(join(output_dir,f"eval_result_{self.save_number}.txt"), 'w') as file:
            for i, (pred, label) in enumerate(zip(p_token_batch, label_token_batch)):
                file.write(f"{i+1}:) PREDICTIN: {pred} \t LABEL: {label}\n")
        self.save_number+=self.training_args.eval_steps
    def compute_metrics(self, p: EvalPrediction):

        preds = p.predictions
        labels = p.label_ids
        preds[preds==IGNORE_INDEX] = 0

        labels[labels==IGNORE_INDEX] = 0
        bleu = BLEUScore()
        rouge = ROUGEScore()
        cider_scorer = CiderScorer(n=4, sigma=6)
        bleu_scorer = BleuScorer(n=4)

        bleu_result = []
        accuracy = 0
        dict_return={}
        
        p_token_batch = self.processor.tokenizer.batch_decode(preds,skip_special_tokens=True)
        label_token_batch = self.processor.tokenizer.batch_decode(labels,skip_special_tokens=True)
        try:
            self.save_pred_label(p_token_batch, label_token_batch, self.training_args.output_dir)
            rouge_mertic = rouge(p_token_batch , label_token_batch )
            dict_return.update(rouge_mertic)
        except:
            print("error when compute rouge")
        for i,p_token in enumerate(p_token_batch):

            bleu_result.append(bleu([p_token],[[label_token_batch[i]]]).item()*100)
            cider_scorer+= (p_token,[label_token_batch[i]])
            bleu_scorer+= (p_token,[label_token_batch[i]])
            l = label_token_batch[i].split(' ')[0]
            if p_token == label_token_batch[i] or p_token in label_token_batch[i] or  label_token_batch[i] in p_token:
                accuracy+=1
        cider,_ = cider_scorer.compute_score()
        bleu_score, _ = bleu_scorer.compute_score(option='closest', verbose=1)

        dict_return['bleu_1'] = bleu_score[0]
        dict_return['bleu_2'] = bleu_score[1]
        dict_return['bleu_3'] = bleu_score[2]
        dict_return['bleu_4'] = bleu_score[3]
        dict_return['cider'] = cider
        dict_return['accuracy'] = accuracy/len(preds)
        dict_return['avg_bleuScore'] = np.array(bleu_result).mean()


        return dict_return
