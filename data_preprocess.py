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

import json
from random import sample

def getsample_vcr(path,types,split_num = -1):
    with open(path,'r') as f:
        js = json.load(f)
    result =[]
    for each in js['instances']:
        inputs= each['input']
        outputs = each['output']
        input_text = each['input'][0]['text'] 
        if len(each['input']) ==3:
            input_image = [each['input'][1]['image']] +each['input'][2]['bbox_list']
        else:
            input_image = each['input'][1]['image'] 
        if len(input_image) > 10:
            continue
        output_text = each['output'][0]['answer'] 
        output_image = ""
        js_dict ={'input_text':input_text,'input_image':input_image,'output_text':output_text,'output_image':output_image}
        result.append(js_dict)
    if split_num == -1:
        with open(f'vcr-promot-0-{types}_nature.json','w') as f:
            for dic in result:
                f.write(json.dumps(dic, ensure_ascii=False)+"\n") 
    else:
        with open(f'vcr-promot-0-{types}_sample_{split_num}_nature.json','w') as f:
            for dic in sample(result,split_num):
                f.write(json.dumps(dic, ensure_ascii=False)+"\n")


def qa_form_fewshot(instances,text,image,sample_num,nature,output_answer='answer'):
    poss_instances= sample(instances,random.randint(0,sample_num))
    images=[]
    texts = []
    for i,each in enumerate(poss_instances):
        inputs= each['input']
        outputs = each['output']
        input_text = each['input'][0]['text'].replace('<image0>',f'<image{i}>').replace('image 0',f'image {i}')
        input_image = inputs[1]['image'] 
        output_text = outputs[0][output_answer] 
        texts.append(input_text+" "+output_text+"\n")
        images.append(input_image)
    text = text.replace('<image0>',f'<image{i+1}>').replace('image 0',f'image {i+1}')
    texts.append(text)
    images.append(image)
    return "\n".join(texts) , images


def getsample_qa(path,types,split_num = -1,sample_icl=6,nature = False,dataset_name ='vqa',store_path=""):
    with open(path,'r') as f:
        js = json.load(f)
    result =[]
    if split_num == -1:
        instances = js['instances']
    else:
        instances = sample(js['instances'],split_num)
    for each in tqdm(instances):
        inputs= each['input']
        outputs = each['output']
        input_text = each['input'][0]['text'] 
        if len(each['input']) ==3:
            input_image = [each['input'][1]['image']] +each['input'][2]['bbox_list']
        else:
            input_image = each['input'][1]['image'] 
        if len(input_image) > 10 and isinstance(input_image,list):
            continue
        output_text = each['output'][0]['answer'] 
        if output_text =="":
            continue
        output_image = ""
        input_text,input_image = qa_form_fewshot(instances,input_text,input_image,sample_icl,nature)
        js_dict ={'input_text':input_text,'input_image':input_image,'output_text':output_text,'output_image':output_image}
        result.append(js_dict)
    if nature:
        if split_num == -1:    
            filename = f'{dataset_name}-promot-0-{types}_nature.json'
        else:
            filename = f'{dataset_name}-promot-0-{types}_sample_{split_num}_nature.json'
    else:
        if split_num == -1:    
            filename = f'{dataset_name}-promot-0-{types}.json'
        else:
            filename = f'{dataset_name}-promot-0-{types}_sample_{split_num}.json'
    with open(os.path.join(store_path,filename),'w') as f:
        for dic in result:
            f.write(json.dumps(dic, ensure_ascii=False)+"\n") 

def caption_form_fewshot(instances,text,image,sample_num,nature,output_answer='answer'):
    poss_instances= sample(instances,random.randint(0,sample_num))
    images=[]
    texts = []
    for i,each in enumerate(poss_instances):
        inputs= each['input']
        outputs = each['output']
        input_text = each['input'][0]['text'].replace('<image0>',f'<image{i}>').replace('image 0',f'image {i}')
        input_image = inputs[1]['image'] 
        output_text = outputs[0][output_answer] 
        texts.append(input_text+" "+output_text+"\n")
        images.append(input_image)
    text = text.replace('<image0>',f'<image{i+1}>').replace('image 0',f'image {i+1}')
    texts.append(text)
    images.append(image)
    return "\n".join(texts) , images

def getsample_caption(path,types,split_num = -1,sample_icl=6,nature = False,dataset_name ='vqa',store_path=""):
    with open(path,'r') as f:
        js = json.load(f)
    result =[]
    if split_num == -1:
        instances = js['instances']
    else:
        instances = sample(js['instances'],split_num)
    for each in tqdm(instances):
        inputs= each['input']
        outputs = each['output']
        input_text = each['input'][0]['text'] 
        if len(each['input']) ==3:
            input_image = [each['input'][1]['image']] +each['input'][2]['bbox_list']
        else:
            input_image = each['input'][1]['image'] 
        if len(input_image) > 10 and isinstance(input_image,list):
            continue
        output_text = each['output'][0]['caption'] 
        if output_text =="":
            continue
        output_image = ""
        input_text,input_image = caption_form_fewshot(instances,input_text,input_image,sample_icl,nature,'caption')
        js_dict ={'input_text':input_text,'input_image':input_image,'output_text':output_text,'output_image':output_image}
        result.append(js_dict)
    if nature:
        if split_num == -1:    
            filename = f'{dataset_name}-promot-0-{types}_nature.json'
        else:
            filename = f'{dataset_name}-promot-0-{types}_sample_{split_num}_nature.json'
    else:
        if split_num == -1:    
            filename = f'{dataset_name}-promot-0-{types}.json'
        else:
            filename = f'{dataset_name}-promot-0-{types}_sample_{split_num}.json'
    with open(os.path.join(store_path,filename),'w') as f:
        for dic in result:
            f.write(json.dumps(dic, ensure_ascii=False)+"\n") 
def generate_jsonl_data_from_instances():
    all_dataset = ['vqa', 'vcr', 'stvqa', 'okvqa', 'nlvr2' ,'gqa', 'refcoco', 'coco' ,'flickr']
    qa_dataset = ['vqa',  'stvqa', 'okvqa', 'nlvr2' ,'gqa',]
    caption_dataset = [ 'refcoco', 'coco' ,'flickr']
    store_path='/home/haozhezhao/Vision-PromptSource/prompt_data'
    data = {
        'vqa': {'train': '/home/haozhezhao/Vision-PromptSource/tasks/task00001-visual_question_answering-vqa-prompt-0-subset-train.json',
                'val': '/home/haozhezhao/Vision-PromptSource/tasks/task00002-visual_question_answering-vqa-prompt-0-subset-val.json',
                'test': '/home/haozhezhao/Vision-PromptSource/tasks/task00003-visual_question_answering-vqa-prompt-0-subset-test.json'},
        'okvqa': {'train': '/home/haozhezhao/Vision-PromptSource/tasks/task00004-visual_question_answering-okvqa-prompt-0-subset-train.json',
                'val': '/home/haozhezhao/Vision-PromptSource/tasks/task00005-visual_question_answering-okvqa-prompt-0-subset-val.json'},
        'nlvr2': {'train': '/home/haozhezhao/Vision-PromptSource/tasks/task00006-visual_question_answering-nlvr2-prompt-0-subset-train.json',
                'val': '/home/haozhezhao/Vision-PromptSource/tasks/task00007-visual_question_answering-nlvr2-prompt-0-subset-val.json',
                'test': '/home/haozhezhao/Vision-PromptSource/tasks/task00008-visual_question_answering-nlvr2-prompt-0-subset-test.json'},
        'vcr': {'train': '/home/haozhezhao/Vision-PromptSource/tasks/task00012-visual_question_answering-vcr-prompt-0-subset-train.json',
                'val': '/home/haozhezhao/Vision-PromptSource/tasks/task00013-visual_question_answering-vcr-prompt-0-subset-val.json'},
        'refcoco': {'train': '/home/haozhezhao/Vision-PromptSource/tasks/task00014-phrase_grounding-refcoco-prompt-0-subset-train.json'},
        'flickr': {'train': '/home/haozhezhao/Vision-PromptSource/tasks/task00016-image_captioning-flickr-prompt-0-subset-train.json',
                'val': '/home/haozhezhao/Vision-PromptSource/tasks/task00017-image_captioning-flickr-prompt-0-subset-val.json',
                'test': '/home/haozhezhao/Vision-PromptSource/tasks/task00018-image_captioning-flickr-prompt-0-subset-test.json'},
        'coco': {'train': '/home/haozhezhao/Vision-PromptSource/tasks/task00021-image_captioning-coco-prompt-0-subset-train.json',
                'val': '/home/haozhezhao/Vision-PromptSource/tasks/task00022-image_captioning-coco-prompt-0-subset-val.json'},
        'gqa': {'train': '/home/haozhezhao/Vision-PromptSource/tasks/task00023-visual_question_answering-gqa-prompt-0-subset-train.json',
                'val': '/home/haozhezhao/Vision-PromptSource/tasks/task00024-visual_question_answering-gqa-prompt-0-subset-val.json',
                'test': '/home/haozhezhao/Vision-PromptSource/tasks/task00025-visual_question_answering-gqa-prompt-0-subset-test.json'},
        'stvqa':{'train':'/home/haozhezhao/Vision-PromptSource/tasks/task00026-visual_question_answering-stvqa-prompt-0-subset-train.json'}
    }

    for datasetname in qa_dataset:
        print(datasetname)
        files = data[datasetname]
        for each in files:
            getsample_qa(files[each],each,nature=True,dataset_name=datasetname,store_path = store_path)
    for datasetname in caption_dataset:
        print(datasetname)
        files = data[datasetname]
        for each in files:
            getsample_caption(files[each],each,nature=True,dataset_name=datasetname,store_path = store_path)
    print('vcr')
    files = data['vcr']
    for each in files:
        getsample_vcr(files[each],each,nature=True,dataset_name='vcr',store_path = store_path)




data_json={'coco_train' : "/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/coco-promot-0-train_nature.json",
'coco_val' :"/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/coco-promot-0-val_nature.json",
'coco_test' : "/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/coco-promot-0-val_nature.json",

'flickr_train' : "/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/flickr-promot-0-train_nature.json",
'flickr_val' : "/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/flickr-promot-0-val_nature.json",
'flickr_test' : "/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/flickr-promot-0-test_nature.json",

'gqa_train' : "/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/gqa-promot-0-train_nature.json",
'gqa_val' : "/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/gqa-promot-0-val_nature.json",
'gqa_test' : "/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/gqa-promot-0-test_nature.json",

'okvqa_train' : "/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/okvqa-promot-0-train_nature.json",
'okvqa_val' : "/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/okvqa-promot-0-val_nature.json",
'okvqa_test' : "/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/okvqa-promot-0-val_nature.json",

'nlvr2_train' : "/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/nlvr2-promot-0-train_nature.json",
'nlvr2_val' : "/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/nlvr2-promot-0-val_nature.json",
'nlvr2_test' : "/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/nlvr2-promot-0-test_nature.json",

'vqa_train' : "/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/vqa-promot-0-train_nature.json",
'vqa_val' : "/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/vqa-promot-0-val_nature.json",
'vqa_test' : "/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/vqa-promot-0-test_nature.json",

'vcr_train' : "/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/vcr-promot-0-train_nature.json",
'vcr_val' : "/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/vcr-promot-0-val_nature.json",
'vcr_test' : "/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/vcr-promot-0-val_nature.json",

'stvqa_train' : "/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/stvqa-promot-0-train_nature.json",
'stvqa_val' : "/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/stvqa-promot-0-train_nature.json",
'stvqa_test' : "/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/stvqa-promot-0-train_nature.json",

'refcoco_train' : "/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/refcoco-promot-0-train_nature.json",
'refcoco_val' : "/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/refcoco-promot-0-train_nature.json",
'refcoco_test' : "/home/haozhezhao/Vision-PromptSource/prompt_data_multiinst/refcoco-promot-0-train_nature.json",}
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
        "train": 30000,
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
    "flickr": {
        "train": 20000,
        "test": 500,
        "val": 500
    },
    "vcr": {
        "train": 118000,
        "test": 2000,
        "val": 2000
    },
    "stvqa": {
        "train": 40000,
        "test": 0,
        "val": 0
    },
    "refcoco": {
        "train": 50000,
        "test": 0,
        "val": 0
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
    if not os.path.exists(f'/home/haozhezhao/Vision-PromptSource/prompt_data/'):
        os.makedirs(f'/home/haozhezhao/Vision-PromptSource/prompt_data/')
    with open(f'/home/haozhezhao/Vision-PromptSource/prompt_data/{file_name}-train.json','w') as f:
        for dic in json_train:
            f.write(json.dumps(dic, ensure_ascii=False)+"\n") 
    with open(f'/home/haozhezhao/Vision-PromptSource/prompt_data/{file_name}-test.json','w') as f:
        for dic in json_test:
            f.write(json.dumps(dic, ensure_ascii=False)+"\n") 
    with open(f'/home/haozhezhao/Vision-PromptSource/prompt_data/{file_name}-val.json','w') as f:
        for dic in json_val:
            f.write(json.dumps(dic, ensure_ascii=False)+"\n") 

path_dir = '/home/haozhezhao/Vision-PromptSource/'
model_name_or_path = '/home/haozhezhao/models/blip2-flan-t5-xl'
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
        image = Image.open(join("/home/haozhezhao/Vision-PromptSource",img_path))
    elif postfix == 'h5':
        image = h5py.File(join("/home/haozhezhao/Vision-PromptSource",img_path), 'r')
    else:
        image = Image.open(join("/home/haozhezhao/Vision-PromptSource", img_path))
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
def process_raw_datajson_to_arrow(json_data,file_name,types,sub_length = -1):
    if not os.path.exists(f'arrow_data_{file_name}_{types}'):
        os.makedirs(f'arrow_data_{file_name}_{types}')
    if sub_length>0:
        json_data = json_data[:sub_length]
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
            if sub_length>0:
                path = f"arrow_data_{file_name}_{types}/bilp2-temp-{types}-{index_arrow}-length{sub_length}.arrow"
            else:
                path = f"arrow_data_{file_name}_{types}/bilp2-temp-{types}-{index_arrow}.arrow"
            t = threading.Thread(target=save_to_arrow, args=(path, save_arrow_data))
            threads.append(t)
            t.start()
            save_arrow_data={'pixel_values':[], 'label':[], 'input_ids':[], 'attention_mask':[]}
            index_arrow+=1
    for t in threads:
        t.join()
    if sub_length>0:
        path = f"arrow_data_{file_name}_{types}/bilp2-temp-{types}-{index_arrow}-length{sub_length}.arrow"
    else:
        path = f"arrow_data_{file_name}_{types}/bilp2-temp-{types}-{index_arrow}.arrow"
    save_to_arrow(path,save_arrow_data)
 



                                                                                                                                                                 
def to_pickle(file_name):
    train_js =f'/home/haozhezhao/Vision-PromptSource/prompt_data/{file_name}-train.json'
    test_js =f'/home/haozhezhao/Vision-PromptSource/prompt_data/{file_name}-test.json'
    val_js =f'/home/haozhezhao/Vision-PromptSource/prompt_data/{file_name}-val.json'
    print('start process training data')
    process_raw_datajson_to_pickle(train_js,'train')
    print('start process testing data')
    process_raw_datajson_to_pickle(test_js,'test')
    print('start process val data')
    process_raw_datajson_to_pickle(val_js,'val')

def to_arrow(file_name,length=-1,do_train = True):
    train_js =f'/home/haozhezhao/Vision-PromptSource/prompt_data/{file_name}-train.json'
    test_js =f'/home/haozhezhao/Vision-PromptSource/prompt_data/{file_name}-test.json'
    val_js =f'/home/haozhezhao/Vision-PromptSource/prompt_data/{file_name}-val.json'
    if do_train:
        train_js = get_json_file(train_js)
        print('start process training data')
        process_raw_datajson_to_arrow(train_js,file_name,'train',length)
    print('start process testing data')
    test_js = get_json_file(test_js)
    process_raw_datajson_to_arrow(test_js,file_name,'test',length)
    print('start process val data')
    val_js = get_json_file(val_js)
    process_raw_datajson_to_arrow(val_js,file_name,'val',length)

def zero_preprocess_json(json_file):
    re =[]
    for j in json_file:
        m={'output_image':""}
        if 'vcr' in j['input_image'][0]:
            m["input_text"] = "image 0 is <image0>图.\n Given the options below"+j['input_text'].split('Given the options below')[1]
            m["input_image"] = [j["input_image"][0]]
            m['output_text'] = j["output_text"]
        else:
            m["input_text"] = "image 4 is <image4>图.\n"+j['input_text'].split('image 4 is <image4>图.\n')[1]
            m["input_image"] = [j["input_image"][-1]]
            m['output_text'] = j["output_text"]
        re.append(m)
    return re
def zero_shot_to_arrow(file_name,length=-1,do_train = True):
    train_js =f'/home/haozhezhao/Vision-PromptSource/prompt_data/{file_name}-train.json'
    test_js =f'/home/haozhezhao/Vision-PromptSource/prompt_data/{file_name}-test.json'
    val_js =f'/home/haozhezhao/Vision-PromptSource/prompt_data/{file_name}-val.json'
    if do_train:
        train_js = get_json_file(train_js)[:length] if length>0 else get_json_file(train_js)
        zero_train_js = zero_preprocess_json(train_js)
        print('start process training data')
        process_raw_datajson_to_arrow(zero_train_js,file_name,'train_zeroshot',length)
    print('start process testing data')
    test_js = get_json_file(test_js)[:length] if length>0 else get_json_file(test_js)
    zero_test_js = zero_preprocess_json(test_js)
    process_raw_datajson_to_arrow(zero_test_js,file_name,'test_zeroshot',length)
    print('start process val data')
    val_js = get_json_file(val_js)[:length] if length>0 else get_json_file(val_js)
    zero_val_js = zero_preprocess_json(val_js)
    process_raw_datajson_to_arrow(zero_val_js,file_name,'val_zeroshot',length)

file_name = 'bilp2-prompt-allshot-multiinst'

seed= 100
if seed is not None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
generate_new_json(data_json,data_size,file_name)

to_arrow(file_name)

# zero_shot_to_arrow(file_name,1000,False).

# generate_jsonl_data_from_instances()