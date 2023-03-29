import json
from os.path import join
from tqdm import tqdm
import transformers
from model.blip2 import Blip2Processor
from model.blip2 import Blip2Config
from PIL import Image
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import h5py
import fsspec
import pickle as pk
import threading
from datasets.arrow_writer import ArrowWriter                                                                                                                                                   
import os

from random import sample
import random
from tqdm import tqdm
data_json={'coco_train' : "prompt_data/coco-promot-0-train_nature.json",
'coco_val' :"prompt_data/coco-promot-0-val_nature.json",
'coco_test' : "prompt_data/coco-promot-0-val_nature.json",

'filckr_train' : "prompt_data/filckr-promot-0-train_nature.json",
'filckr_val' : "prompt_data/filckr-promot-0-val_nature.json",
'filckr_test' : "prompt_data/filckr-promot-0-test_nature.json",

'gqa_train' : "prompt_data/gqa-promot-0-train_nature.json",
'gqa_val' : "prompt_data/gqa-promot-0-val_nature.json",
'gqa_test' : "prompt_data/gqa-promot-0-test_nature.json",

'okvqa_train' : "prompt_data/okvqa-promot-0-train_nature.json",
'okvqa_val' : "prompt_data/okvqa-promot-0-val_nature.json",
'okvqa_test' : "prompt_data/okvqa-promot-0-val_nature.json",

'nlvr2_train' : "prompt_data/nlvr2-promot-0-train_nature.json",
'nlvr2_val' : "prompt_data/nlvr2-promot-0-val_nature.json",
'nlvr2_test' : "prompt_data/nlvr2-promot-0-test_nature.json",

'vqa_train' : "prompt_data/vqa-promot-0-train_nature.json",
'vqa_val' : "prompt_data/vqa-promot-0-val_nature.json",
'vqa_test' : "prompt_data/vqa-promot-0-test_nature.json",

'vcr_train' : "prompt_data/vcr-promot-0-train_nature.json",
'vcr_val' : "prompt_data/vcr-promot-0-test_nature.json",
'vcr_test' : "prompt_data/vcr-promot-0-test_nature.json",}
data_size= {
    "vqa": {
        "train": 60000,
        "test": 627,
        "val": 639
    },
    "gqa": {
        "train": 60000,
        "test": 1000,
        "val": 1000
    },
    "okvqa": {
        "train": 6710,
        "test": 3000,
        "val": 3000
    },
    "nlvr2": {
        "train": 20000,
        "test": 1166,
        "val": 1169
    },
    "coco": {
        "train": 30000,
        "test": 2000,
        "val": 2000
    },
    "filckr": {
        "train": 20000,
        "test": 500,
        "val": 500
    },
    "vcr": {
        "train": 118000,
        "test": 2000,
        "val": 2000
    }
}

def get_json_file(file_path):
    js =[]
    with open(file_path,'r')as f:
        for line in f.readlines():
            js.append(json.loads(line))
    return js
def generate_new_json(data_json,data_size,file_name):
    json_train=[]
    json_test=[]
    json_val=[]
    for each in tqdm(data_size):
        js_train = sample(get_json_file(data_json[f'{each}_train']),data_size[each]['train'])
        js_test = sample(get_json_file(data_json[f'{each}_test']),data_size[each]['test'])
        js_val = sample(get_json_file(data_json[f'{each}_val']),data_size[each]['val'])
        json_train.extend(js_train)
        json_test.extend(js_test)
        json_val.extend(js_val)
    random.shuffle(json_train)
    random.shuffle(json_test)
    random.shuffle(json_val)
    if not os.path.exists(f'prompt_data/'):
        os.makedirs(f'prompt_data/')
    with open(f'prompt_data/{file_name}-train.json','w') as f:
        for dic in json_train:
            f.write(json.dumps(dic, ensure_ascii=False)+"\n") 
    with open(f'prompt_data/{file_name}-test.json','w') as f:
        for dic in json_test:
            f.write(json.dumps(dic, ensure_ascii=False)+"\n") 
    with open(f'prompt_data/{file_name}-val.json','w') as f:
        for dic in json_val:
            f.write(json.dumps(dic, ensure_ascii=False)+"\n") 

path_dir = '/home/haozhezhao/Vision-PromptSource/'
model_name_or_path = '/home/haozhezhao/model/blip2-flan-t5-xl'
def get_json_file(file_path):
    js =[]
    with open(file_path,'r')as f:
        for line in f.readlines():
            js.append(json.loads(line))
    return js

processor = Blip2Processor.from_pretrained(
    model_name_or_path,
)
sp = ["图"]+[f"<image{i}>" for i in range(10)]
sp = sp+processor.tokenizer.additional_special_tokens[len(sp):]
processor.tokenizer.add_special_tokens({'additional_special_tokens':sp})
max_seq_length = min(512, processor.tokenizer.model_max_length)

def save_pickle_img(path,file):
    with open(join(path_dir,path),'ab') as f:
        pk.dump(file,f)

def read_image(postfix,img_path):
    if postfix == 'png':
        image = Image.open(join(path_dir,img_path))
    elif postfix == 'h5':
        image = h5py.File(join(path_dir,img_path), 'r')
    else:
        image = Image.open(join(path_dir, img_path))
    return image
def preprocess_function(input_text,input_image,output_text):
    result = {}
    flag = isinstance(input_image,list)
    result["pixel_values"] = []
    if flag:
        postfix = input_image[0][1:].split('.')[-1]
        for img_path in input_image:
            img_path = img_path[1:] if img_path[0] == '.' and img_path[1] !='/' else img_path
            img = read_image(postfix,img_path)
            result["pixel_values"].append(processor(images = img)["pixel_values"][0])
    else:
        postfix = input_image[1:].split('.')[-1]
        img_path = img_path[1:] if img_path[0] == '.' and img_path[1] !='/'  else img_path
        img = read_image(postfix,img_path)
        result["pixel_values"].append(processor(images = img)["pixel_values"][0])
    return result
def preprocess_function_batched(result,input_text,output_text):
    re= processor.tokenizer(input_text, padding='max_length', max_length=max_seq_length, truncation=True)
    re['input_ids'] = np.array(re['input_ids'],dtype=np.int32)
    re['attention_mask'] = np.array(re['attention_mask'],dtype=np.bool_)
    # result['label'] = np.array(processor.tokenizer(output_text, padding='max_length', max_length=32, truncation=True)["input_ids"],dtype=np.int32)
    out = processor.tokenizer(output_text, padding='max_length', max_length=32, truncation=True)
    result['label'] = np.array(out["input_ids"],dtype=np.int32)
    result['label_attention_mask'] = np.array(out["attention_mask"],dtype=np.bool_)
    result.update(re)
    return result

def process_raw_datajson_to_pickle(json_data,types):
    json_data = get_json_file(json_data)
    with ThreadPoolExecutor(max_workers=10) as executor:
        for each in tqdm(json_data):
            input_text = each['input_text']
            input_imgs = each['input_image']
            output_text = each['output_text']
            temp = preprocess_function(input_text,input_imgs,output_text)
            temp = preprocess_function_batched(temp,input_text,output_text)
            executor.submit(save_pickle_img, f"bilp2-prompt-{types}.pkl",temp)
def save_to_arrow(path,temp):
    with ArrowWriter(path=path) as writer: 
        writer.write_batch(temp) 
        writer.finalize() 
def process_raw_datajson_to_arrow(json_data,types):
    if not os.path.exists(f'arrow_data_{types}'):
        os.makedirs(f'arrow_data_{types}')
    json_data = get_json_file(json_data)
    save_arrow_data={'pixel_values':[], 'label':[], 'input_ids':[], 'attention_mask':[]}
    index_arrow=0
    threads = []
    for idx,each in enumerate(tqdm(json_data)):
        input_text = each['input_text']
        input_imgs = each['input_image']
        output_text = each['output_text']
        temp = preprocess_function(input_text,input_imgs,output_text)
        temp = preprocess_function_batched(temp,input_text,output_text)
        for each in save_arrow_data:
            save_arrow_data[each].append(temp[each])
        if idx %1000 == 0 and idx !=0:
            path = f"arrow_data_{types}/bilp2-temp-{types}-{index_arrow}.arrow"
            t = threading.Thread(target=save_to_arrow, args=(path, save_arrow_data))
            threads.append(t)
            t.start()
            save_arrow_data={'pixel_values':[], 'label':[], 'input_ids':[], 'attention_mask':[]}
            index_arrow+=1
    for t in threads:
        t.join()
    path = f"arrow_data_{types}/bilp2-temp-{types}-{index_arrow}.arrow"
    save_to_arrow(path,save_arrow_data)
 



                                                                                                                                                                 
def to_pickle(file_name):
    train_js =f'prompt_data/{file_name}-train.json'
    test_js =f'prompt_data/{file_name}-test.json'
    val_js =f'prompt_data/{file_name}-val.json'
    print('start process training data')
    process_raw_datajson_to_pickle(train_js,'train')
    print('start process testing data')
    process_raw_datajson_to_pickle(test_js,'test')
    print('start process val data')
    process_raw_datajson_to_pickle(val_js,'val')

def to_arrow(file_name):
    train_js =f'prompt_data/{file_name}-train.json'
    test_js =f'prompt_data/{file_name}-test.json'
    val_js =f'prompt_data/{file_name}-val.json'
    print('start process training data')
    process_raw_datajson_to_arrow(train_js,'train')
    print('start process testing data')
    process_raw_datajson_to_arrow(test_js,'test')
    print('start process val data')
    process_raw_datajson_to_arrow(val_js,'val')
file_name = 'bilp2-prompt-0'
# generate_new_json(data_json,data_size,file_name)

to_arrow(file_name)