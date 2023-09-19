import json
from os.path import join
from tqdm import tqdm
import transformers
from model.blip2 import Blip2Processor
from model.blip2 import Blip2Config
from model.instructblip import InstructBlipProcessor
from PIL import Image
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import h5py
import fsspec
import copy
import pickle as pk
import glob
from os.path import join
import threading
from datasets.arrow_writer import ArrowWriter                                                                                                                                                   
import os
from moviepy.editor import VideoFileClip
from PIL import Image
from random import sample
import random
from tqdm import tqdm

import json
from random import sample
import os

def getsample_vcr(path,types,split_num = -1,store_path =''):
    with open(path,'r') as f:
        js = json.load(f)
    result =[]
    for each in tqdm(js['instances']):
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
        with open(os.path.join(store_path,f'vcr-promot-0-{types}_nature.json'),'w') as f:
            for dic in result:
                f.write(json.dumps(dic, ensure_ascii=False)+"\n") 
    else:
        with open(os.path.join(store_path,f'vcr-promot-0-{types}_sample_{split_num}_nature.json'),'w') as f:
            for dic in sample(result,split_num):
                f.write(json.dumps(dic, ensure_ascii=False)+"\n")

def getsample_llava(path,types,split_num = -1,store_path =''):
    with open(path,'r') as f:
        js = json.load(f)
    result =[]
    for each in tqdm(js['instances']):
        inputs= each['input']
        outputs = each['output']
        input_text = each['input'][0]['text'] 
        input_image = each['input'][1]['image'] 
        output_text = each['output'][0]['answer'] 
        output_image = ""
        js_dict ={'input_text':input_text,'input_image':input_image,'output_text':output_text,'output_image':output_image}
        result.append(js_dict)
    if split_num == -1:
        with open(os.path.join(store_path,f'llava-promot-0-{types}_nature.json'),'w') as f:
            for dic in result:
                f.write(json.dumps(dic, ensure_ascii=False)+"\n") 
    else:
        with open(os.path.join(store_path,f'llava-promot-0-{types}_sample_{split_num}_nature.json'),'w') as f:
            for dic in sample(result,split_num):
                f.write(json.dumps(dic, ensure_ascii=False)+"\n")


def qa_form_fewshot(instances,text,image,sample_num,nature,output_answer='answer'):
    poss_instances= sample(instances,random.randint(0,sample_num))
    images=[]
    texts = []
    i=0
    for i,each in enumerate(poss_instances):
        inputs= each['input']
        outputs = each['output']
        input_text = each['input'][0]['text'].replace('<image0>',f'<image{i}>').replace('image 0',f'image {i}')
        input_image = inputs[1]['image'] 
        output_text = outputs[0][output_answer] 
        texts.append(input_text+" "+output_text+"\n")
        images.append(input_image)
    text = text.replace('<image0>',f'<image{i+1}>').replace('image 0',f'image {i+1}') if i>0 else text
    texts.append(text)
    images.append(image)
    return "\n".join(texts) , images


def getsample_qa(path,types,split_num = -1,sample_icl=5,nature = False,dataset_name ='vqa',store_path=""):
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


def getsample_video(path,types,split_num = -1,dataset_name="ivqa",store_path =''):
    with open(path,'r') as f:
        js = json.load(f)
    result =[]
    for each in tqdm(js['instances']):
        inputs= each['input']
        outputs = each['output']
        input_text = each['input'][0]['text'] 
        input_image = each['input'][1]['image'] 
        output_text = each['output'][0]['answer'] 
        output_image = ""
        js_dict ={'input_text':input_text,'input_image':input_image,'output_text':output_text,'output_image':output_image}
        result.append(js_dict)
    if split_num == -1:
        with open(os.path.join(store_path,f'{dataset_name}-promot-0-{types}_nature.json'),'w') as f:
            for dic in result:
                f.write(json.dumps(dic, ensure_ascii=False)+"\n") 
    else:
        with open(os.path.join(store_path,f'{dataset_name}-promot-0-{types}_sample_{split_num}_nature.json'),'w') as f:
            for dic in sample(result,split_num):
                f.write(json.dumps(dic, ensure_ascii=False)+"\n")

def caption_form_fewshot(instances,text,image,sample_num,nature,output_answer='answer'):
    poss_instances= sample(instances,random.randint(0,sample_num))
    images=[]
    texts = []
    i=0
    for i,each in enumerate(poss_instances):
        inputs= each['input']
        outputs = each['output']
        input_text = each['input'][0]['text'].replace('<image0>',f'<image{i}>').replace('image 0',f'image {i}')
        input_image = inputs[1]['image'] 
        output_text = outputs[0][output_answer].split("##")[0]
        texts.append(input_text+" "+output_text+"\n")
        images.append(input_image)
    text = text.replace('<image0>',f'<image{i+1}>').replace('image 0',f'image {i+1}') if i>0 else text
    texts.append(text)
    images.append(image)
    return "\n".join(texts) , images

def getsample_caption(path,types,split_num = -1,sample_icl=5,nature = False,dataset_name ='vqa',store_path=""):
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
def generate_jsonl_data_from_instances(store_path):
    all_dataset = ['llava', 'textvqa', 'diffusiondb','msrvttqa','msrvtt', 'wikiart','nocaps', 'miniimage','vqa', 'vcr', 'stvqa', 'okvqa', 'nlvr2' ,'gqa', 'refcoco', 'coco' ,'flickr']
    qa_dataset = ['vqa',  'stvqa', 'okvqa', 'nlvr2' ,'gqa','textvqa','wikiart','iconqa']
    caption_dataset = [ 'refcoco', 'coco' ,'flickr','diffusiondb','miniimage','nocaps']
    video_dataset=['msrvttqa','msrvtt']


    if not os.path.exists(store_path):
        os.makedirs(store_path)
    data = { 
        'llava':{'train':'Vision-PromptSource/tasks/task00001-visual_dialog-llava-prompt-0-subset-train.json'}  ,
        'textvqa':{'train':'Vision-PromptSource/tasks/task00002-visual_question_answering-textvqa-prompt-0-subset-train.json'}  ,
        'diffusiondb':{'train':'Vision-PromptSource/tasks/task00006-image_captioning-diffusiondb-prompt-0-subset-train.json'}  ,
        'wikiart':{'train':'Vision-PromptSource/tasks/task00007-image_generation-wikiart-prompt-0-subset-train.json',
                   'val':'Vision-PromptSource/tasks/task00008-image_generation-wikiart-prompt-0-subset-val.json'}  ,
        'nocaps':{'val':'Vision-PromptSource/tasks/task00026-image_captioning-nocaps-prompt-0-subset-val.json'}  ,
        'miniimage':{'train':'Vision-PromptSource/tasks/task00027-image_classification-miniimage-prompt-0-subset-train.json',
                     'val': 'Vision-PromptSource/tasks/task00028-image_classification-miniimage-prompt-0-subset-val.json',
                'test': 'Vision-PromptSource/tasks/task00029-image_classification-miniimage-prompt-0-subset-test.json'}  ,

        'vqa': {'train': 'Vision-PromptSource/tasks/task00009-visual_question_answering-vqa-prompt-0-subset-train.json',
                'val': 'Vision-PromptSource/tasks/task00010-visual_question_answering-vqa-prompt-0-subset-val.json',
                'test': 'Vision-PromptSource/tasks/task00011-visual_question_answering-vqa-prompt-0-subset-test.json'},
        'okvqa': {'train': 'Vision-PromptSource/tasks/task00018-visual_question_answering-okvqa-prompt-0-subset-train.json',
                'val': 'Vision-PromptSource/tasks/task00019-visual_question_answering-okvqa-prompt-0-subset-val.json'},
        'nlvr2': {'train': 'Vision-PromptSource/tasks/task00012-visual_question_answering-nlvr2-prompt-0-subset-train.json',
                'val': 'Vision-PromptSource/tasks/task00013-visual_question_answering-nlvr2-prompt-0-subset-val.json',
                'test': 'Vision-PromptSource/tasks/task00014-visual_question_answering-nlvr2-prompt-0-subset-test.json'},
        'vcr': {'train': 'Vision-PromptSource/tasks/task00004-visual_question_answering-vcr-prompt-0-subset-train.json',
                'val': 'Vision-PromptSource/tasks/task00005-visual_question_answering-vcr-prompt-0-subset-val.json'},
        'refcoco': {'train': 'Vision-PromptSource/tasks/task00022-phrase_grounding-refcoco-prompt-0-subset-train.json'},
        'flickr': {'train': 'Vision-PromptSource/tasks/task00023-image_captioning-flickr-prompt-0-subset-train.json',
                'val': 'Vision-PromptSource/tasks/task00024-image_captioning-flickr-prompt-0-subset-val.json',
                'test': 'Vision-PromptSource/tasks/task00025-image_captioning-flickr-prompt-0-subset-test.json'},
        'coco': {'train': 'Vision-PromptSource/tasks/task00020-image_captioning-coco-prompt-0-subset-train.json',
                'val': 'Vision-PromptSource/tasks/task00021-image_captioning-coco-prompt-0-subset-val.json'},
        'gqa': {'train': 'Vision-PromptSource/tasks/task00015-visual_question_answering-gqa-prompt-0-subset-train.json',
                'val': 'Vision-PromptSource/tasks/task00016-visual_question_answering-gqa-prompt-0-subset-val.json',
                'test': 'Vision-PromptSource/tasks/task00017-visual_question_answering-gqa-prompt-0-subset-test.json'},
        'stvqa':{'train':'Vision-PromptSource/tasks/task00003-visual_question_answering-stvqa-prompt-0-subset-train.json'},
        'msrvttqa':{'train':'Vision-PromptSource/tasks/task00030-video_question_answering-msrvttqa-prompt-0-subset-train.json',
                    'val':'Vision-PromptSource/tasks/task00031-video_question_answering-msrvttqa-prompt-0-subset-val.json',
                    'test':'Vision-PromptSource/tasks/task00032-video_question_answering-msrvttqa-prompt-0-subset-test.json'},
        'msrvtt':{'train':'Vision-PromptSource/tasks/task00033-video_question_answering-msrvtt-prompt-0-subset-train.json',
                    'val':'Vision-PromptSource/tasks/task00034-video_question_answering-msrvtt-prompt-0-subset-val.json',
                    'test':'Vision-PromptSource/tasks/task00035-video_question_answering-msrvtt-prompt-0-subset-test.json'},
        'iconqa': {'train':"Vision-PromptSource/tasks/task00044-visual_question_answering-iconqa-prompt-0-subset-train.json",
                   'val':"Vision-PromptSource/tasks/task00045-visual_question_answering-iconqa-prompt-0-subset-val.json",
                   'test':"Vision-PromptSource/tasks/task00046-visual_question_answering-iconqa-prompt-0-subset-test.json",}
    }

    for datasetname in qa_dataset:
        print(datasetname)
        files = data[datasetname]
        for each in files:
            getsample_qa(files[each],each,nature=True,sample_icl=sample_number,dataset_name=datasetname,store_path = store_path)
    for datasetname in caption_dataset:
        print(datasetname)
        files = data[datasetname]
        for each in files:
            getsample_caption(files[each],each,nature=True,sample_icl=sample_number,dataset_name=datasetname,store_path = store_path)
    print('vcr')
    files = data['vcr']
    for each in files:
        getsample_vcr(files[each],each,store_path = store_path)
    files = data['llava']
    for each in files:
        print('llava')
        getsample_llava(files[each],each,store_path = store_path)
    for datasetname in video_dataset:
        print(datasetname)
        files = data[datasetname]
        for each in files:
            getsample_video(files[each],each,dataset_name=datasetname,store_path = store_path)

    # datasetname ='flickr'
    # files = data[datasetname]
    # for each in files:
    #     getsample_caption(files[each],each,sample_icl=0,nature=True,dataset_name=datasetname,store_path = store_path)




# data_json={
# 'coco_train' : "Vision-PromptSource/prompt_data_new_6_12_zero-shot/coco-promot-0-train_nature.json",
# 'coco_val' :"Vision-PromptSource/prompt_data_new_6_12_zero-shot/coco-promot-0-val_nature.json",
# 'coco_test' : "Vision-PromptSource/prompt_data_new_6_12_zero-shot/coco-promot-0-val_nature.json",

# 'flickr_train' : "Vision-PromptSource/prompt_data_new_6_12_zero-shot/flickr-promot-0-train_nature.json",
# 'flickr_val' : "Vision-PromptSource/prompt_data_new_6_12_zero-shot/flickr-promot-0-val_nature.json",
# 'flickr_test' : "Vision-PromptSource/prompt_data_new_6_12_zero-shot/flickr-promot-0-test_nature.json",

# 'gqa_train' : "Vision-PromptSource/prompt_data_new_6_12_zero-shot/gqa-promot-0-train_nature.json",
# 'gqa_val' : "Vision-PromptSource/prompt_data_new_6_12_zero-shot/gqa-promot-0-val_nature.json",
# 'gqa_test' : "Vision-PromptSource/prompt_data_new_6_12_zero-shot/gqa-promot-0-test_nature.json",

# 'okvqa_train' : "Vision-PromptSource/prompt_data_new_6_12_zero-shot/okvqa-promot-0-train_nature.json",
# 'okvqa_val' : "Vision-PromptSource/prompt_data_new_6_12_zero-shot/okvqa-promot-0-val_nature.json",
# 'okvqa_test' : "Vision-PromptSource/prompt_data_new_6_12_zero-shot/okvqa-promot-0-val_nature.json",

# 'nlvr2_train' : "Vision-PromptSource/prompt_data_new_6_12_zero-shot/nlvr2-promot-0-train_nature.json",
# 'nlvr2_val' : "Vision-PromptSource/prompt_data_new_6_12_zero-shot/nlvr2-promot-0-val_nature.json",
# 'nlvr2_test' : "Vision-PromptSource/prompt_data_new_6_12_zero-shot/nlvr2-promot-0-test_nature.json",

# 'vqa_train' : "Vision-PromptSource/prompt_data_new_6_12_zero-shot/vqa-promot-0-train_nature.json",
# 'vqa_val' : "Vision-PromptSource/prompt_data_new_6_12_zero-shot/vqa-promot-0-val_nature.json",
# 'vqa_test' : "Vision-PromptSource/prompt_data_new_6_12_zero-shot/vqa-promot-0-test_nature.json",

# 'miniimage_train':'Vision-PromptSource/prompt_data_new_6_12_zero-shot/miniimage-promot-0-train_nature.json' ,
# 'miniimage_val':'Vision-PromptSource/prompt_data_new_6_12_zero-shot/miniimage-promot-0-val_nature.json' ,
# 'miniimage_test':'Vision-PromptSource/prompt_data_new_6_12_zero-shot/miniimage-promot-0-test_nature.json' ,

# 'wikiart_train':'Vision-PromptSource/prompt_data_new_6_12_zero-shot/wikiart-promot-0-train_nature.json' ,
# 'wikiart_val':'Vision-PromptSource/prompt_data_new_6_12_zero-shot/wikiart-promot-0-val_nature.json' ,
# 'wikiart_test':'Vision-PromptSource/prompt_data_new_6_12_zero-shot/wikiart-promot-0-val_nature.json' ,


# 'stvqa_train' : "Vision-PromptSource/prompt_data_new_6_12_zero-shot/stvqa-promot-0-train_nature.json",
# 'stvqa_val' : "Vision-PromptSource/prompt_data_new_6_12_zero-shot/stvqa-promot-0-train_nature.json",
# 'stvqa_test' : "Vision-PromptSource/prompt_data_new_6_12_zero-shot/stvqa-promot-0-train_nature.json",

# 'refcoco_train' : "Vision-PromptSource/prompt_data_new_6_12_zero-shot/refcoco-promot-0-train_nature.json",
# 'refcoco_val' : "Vision-PromptSource/prompt_data_new_6_12_zero-shot/refcoco-promot-0-train_nature.json",
# 'refcoco_test' : "Vision-PromptSource/prompt_data_new_6_12_zero-shot/refcoco-promot-0-train_nature.json",

# 'textvqa_train':'Vision-PromptSource/prompt_data_new_6_12_zero-shot/textvqa-promot-0-train_nature.json' ,
# 'textvqa_val':'Vision-PromptSource/prompt_data_new_6_12_zero-shot/textvqa-promot-0-train_nature.json' ,
# 'textvqa_test':'Vision-PromptSource/prompt_data_new_6_12_zero-shot/textvqa-promot-0-train_nature.json' ,

# 'diffusiondb_train':'Vision-PromptSource/prompt_data_new_6_12_zero-shot/diffusiondb-promot-0-train_nature.json' ,
# 'diffusiondb_val':'Vision-PromptSource/prompt_data_new_6_12_zero-shot/diffusiondb-promot-0-train_nature.json' ,
# 'diffusiondb_test':'Vision-PromptSource/prompt_data_new_6_12_zero-shot/diffusiondb-promot-0-train_nature.json' ,

# 'nocaps_train':'Vision-PromptSource/prompt_data_new_6_12_zero-shot/nocaps-promot-0-val_nature.json' ,
# 'nocaps_val':'Vision-PromptSource/prompt_data_new_6_12_zero-shot/nocaps-promot-0-val_nature.json' ,
# 'nocaps_test':'Vision-PromptSource/prompt_data_new_6_12_zero-shot/nocaps-promot-0-val_nature.json' ,

# }
# data_json={
# 'coco_train' : "Vision-PromptSource/prompt_data_new_6_5/coco-promot-0-train_nature.json",
# 'coco_val' :"Vision-PromptSource/prompt_data_new_6_5/coco-promot-0-val_nature.json",
# 'coco_test' : "Vision-PromptSource/prompt_data_new_6_5/coco-promot-0-val_nature.json",

# 'flickr_train' : "Vision-PromptSource/prompt_data_new_6_5/flickr-promot-0-train_nature.json",
# 'flickr_val' : "Vision-PromptSource/prompt_data_new_6_5/flickr-promot-0-val_nature.json",
# 'flickr_test' : "Vision-PromptSource/prompt_data_new_6_5/flickr-promot-0-test_nature.json",

# 'gqa_train' : "Vision-PromptSource/prompt_data_new_6_5/gqa-promot-0-train_nature.json",
# 'gqa_val' : "Vision-PromptSource/prompt_data_new_6_5/gqa-promot-0-val_nature.json",
# 'gqa_test' : "Vision-PromptSource/prompt_data_new_6_5/gqa-promot-0-test_nature.json",

# 'okvqa_train' : "Vision-PromptSource/prompt_data_new_6_5/okvqa-promot-0-train_nature.json",
# 'okvqa_val' : "Vision-PromptSource/prompt_data_new_6_5/okvqa-promot-0-val_nature.json",
# 'okvqa_test' : "Vision-PromptSource/prompt_data_new_6_5/okvqa-promot-0-val_nature.json",

# 'nlvr2_train' : "Vision-PromptSource/prompt_data_new_6_5/nlvr2-promot-0-train_nature.json",
# 'nlvr2_val' : "Vision-PromptSource/prompt_data_new_6_5/nlvr2-promot-0-val_nature.json",
# 'nlvr2_test' : "Vision-PromptSource/prompt_data_new_6_5/nlvr2-promot-0-test_nature.json",

# 'vqa_train' : "Vision-PromptSource/prompt_data_new_6_5/vqa-promot-0-train_nature.json",
# 'vqa_val' : "Vision-PromptSource/prompt_data_new_6_5/vqa-promot-0-val_nature.json",
# 'vqa_test' : "Vision-PromptSource/prompt_data_new_6_5/vqa-promot-0-test_nature.json",

# 'vcr_train' : "Vision-PromptSource/prompt_data_new_6_5/vcr-promot-0-train_nature.json",
# 'vcr_val' : "Vision-PromptSource/prompt_data_new_6_5/vcr-promot-0-val_nature.json",
# 'vcr_test' : "Vision-PromptSource/prompt_data_new_6_5/vcr-promot-0-val_nature.json",

# 'miniimage_train':'Vision-PromptSource/prompt_data_new_6_5/miniimage-promot-0-train_nature.json' ,
# 'miniimage_val':'Vision-PromptSource/prompt_data_new_6_5/miniimage-promot-0-val_nature.json' ,
# 'miniimage_test':'Vision-PromptSource/prompt_data_new_6_5/miniimage-promot-0-test_nature.json' ,

# 'wikiart_train':'Vision-PromptSource/prompt_data_new_6_5/wikiart-promot-0-train_nature.json' ,
# 'wikiart_val':'Vision-PromptSource/prompt_data_new_6_5/wikiart-promot-0-val_nature.json' ,
# 'wikiart_test':'Vision-PromptSource/prompt_data_new_6_5/wikiart-promot-0-val_nature.json' ,


# 'stvqa_train' : "Vision-PromptSource/prompt_data_new_6_5/stvqa-promot-0-train_nature.json",
# 'stvqa_val' : "Vision-PromptSource/prompt_data_new_6_5/stvqa-promot-0-train_nature.json",
# 'stvqa_test' : "Vision-PromptSource/prompt_data_new_6_5/stvqa-promot-0-train_nature.json",

# 'refcoco_train' : "Vision-PromptSource/prompt_data_new_6_5/refcoco-promot-0-train_nature.json",
# 'refcoco_val' : "Vision-PromptSource/prompt_data_new_6_5/refcoco-promot-0-train_nature.json",
# 'refcoco_test' : "Vision-PromptSource/prompt_data_new_6_5/refcoco-promot-0-train_nature.json",

# 'llava_train':'Vision-PromptSource/prompt_data_new_6_5/llava-promot-0-train_nature.json' ,
# 'llava_val':'Vision-PromptSource/prompt_data_new_6_5/llava-promot-0-train_nature.json' ,
# 'llava_test':'Vision-PromptSource/prompt_data_new_6_5/llava-promot-0-train_nature.json' ,

# 'textvqa_train':'Vision-PromptSource/prompt_data_new_6_5/textvqa-promot-0-train_nature.json' ,
# 'textvqa_val':'Vision-PromptSource/prompt_data_new_6_5/textvqa-promot-0-train_nature.json' ,
# 'textvqa_test':'Vision-PromptSource/prompt_data_new_6_5/textvqa-promot-0-train_nature.json' ,

# 'diffusiondb_train':'Vision-PromptSource/prompt_data_new_6_5/diffusiondb-promot-0-train_nature.json' ,
# 'diffusiondb_val':'Vision-PromptSource/prompt_data_new_6_5/diffusiondb-promot-0-train_nature.json' ,
# 'diffusiondb_test':'Vision-PromptSource/prompt_data_new_6_5/diffusiondb-promot-0-train_nature.json' ,

# 'nocaps_train':'Vision-PromptSource/prompt_data_new_6_5/nocaps-promot-0-val_nature.json' ,
# 'nocaps_val':'Vision-PromptSource/prompt_data_new_6_5/nocaps-promot-0-val_nature.json' ,
# 'nocaps_test':'Vision-PromptSource/prompt_data_new_6_5/nocaps-promot-0-val_nature.json' ,

# 'msrvtt_train':'Vision-PromptSource/prompt_data_new_6_5/msrvtt-promot-0-train_nature.json',
# 'msrvtt_val':'Vision-PromptSource/prompt_data_new_6_5/msrvtt-promot-0-val_nature.json',
# 'msrvtt_test':'Vision-PromptSource/prompt_data_new_6_5/msrvtt-promot-0-test_nature.json',

# 'msrvttqa_train':'Vision-PromptSource/prompt_data_new_6_5/msrvttqa-promot-0-train_nature.json',
# 'msrvttqa_val':'Vision-PromptSource/prompt_data_new_6_5/msrvttqa-promot-0-val_nature.json',
# 'msrvttqa_test':'Vision-PromptSource/prompt_data_new_6_5/msrvttqa-promot-0-test_nature.json',

# }
data_json={
'coco_train' : "Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/coco-promot-0-train_nature.json",
'coco_val' :"Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/coco-promot-0-val_nature.json",
'coco_test' : "Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/coco-promot-0-val_nature.json",

'flickr_train' : "Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/flickr-promot-0-train_nature.json",
'flickr_val' : "Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/flickr-promot-0-val_nature.json",
'flickr_test' : "Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/flickr-promot-0-test_nature.json",

'gqa_train' : "Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/gqa-promot-0-train_nature.json",
'gqa_val' : "Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/gqa-promot-0-val_nature.json",
'gqa_test' : "Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/gqa-promot-0-test_nature.json",

'okvqa_train' : "Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/okvqa-promot-0-train_nature.json",
'okvqa_val' : "Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/okvqa-promot-0-val_nature.json",
'okvqa_test' : "Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/okvqa-promot-0-val_nature.json",

'nlvr2_train' : "Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/nlvr2-promot-0-train_nature.json",
'nlvr2_val' : "Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/nlvr2-promot-0-val_nature.json",
'nlvr2_test' : "Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/nlvr2-promot-0-test_nature.json",

'vqa_train' : "Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/vqa-promot-0-train_nature.json",
'vqa_val' : "Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/vqa-promot-0-val_nature.json",
'vqa_test' : "Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/vqa-promot-0-test_nature.json",

'vcr_train' : "Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/vcr-promot-0-train_nature.json",
'vcr_val' : "Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/vcr-promot-0-val_nature.json",
'vcr_test' : "Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/vcr-promot-0-val_nature.json",

'miniimage_train':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/miniimage-promot-0-train_nature.json' ,
'miniimage_val':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/miniimage-promot-0-val_nature.json' ,
'miniimage_test':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/miniimage-promot-0-test_nature.json' ,

'wikiart_train':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/wikiart-promot-0-train_nature.json' ,
'wikiart_val':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/wikiart-promot-0-val_nature.json' ,
'wikiart_test':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/wikiart-promot-0-val_nature.json' ,


'stvqa_train' : "Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/stvqa-promot-0-train_nature.json",
'stvqa_val' : "Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/stvqa-promot-0-train_nature.json",
'stvqa_test' : "Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/stvqa-promot-0-train_nature.json",

'refcoco_train' : "Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/refcoco-promot-0-train_nature.json",
'refcoco_val' : "Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/refcoco-promot-0-train_nature.json",
'refcoco_test' : "Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/refcoco-promot-0-train_nature.json",

'llava_train':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/llava-promot-0-train_nature.json' ,
'llava_val':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/llava-promot-0-train_nature.json' ,
'llava_test':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/llava-promot-0-train_nature.json' ,

'textvqa_train':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/textvqa-promot-0-train_nature.json' ,
'textvqa_val':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/textvqa-promot-0-train_nature.json' ,
'textvqa_test':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/textvqa-promot-0-train_nature.json' ,

'diffusiondb_train':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/diffusiondb-promot-0-train_nature.json' ,
'diffusiondb_val':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/diffusiondb-promot-0-train_nature.json' ,
'diffusiondb_test':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/diffusiondb-promot-0-train_nature.json' ,

'nocaps_train':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/nocaps-promot-0-val_nature.json' ,
'nocaps_val':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/nocaps-promot-0-val_nature.json' ,
'nocaps_test':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/nocaps-promot-0-val_nature.json' ,

'msrvtt_train':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/msrvtt-promot-0-train_nature.json',
'msrvtt_val':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/msrvtt-promot-0-val_nature.json',
'msrvtt_test':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/msrvtt-promot-0-test_nature.json',

'msrvttqa_train':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/msrvttqa-promot-0-train_nature.json',
'msrvttqa_val':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/msrvttqa-promot-0-val_nature.json',
'msrvttqa_test':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/msrvttqa-promot-0-test_nature.json',

'iconqa_train':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/iconqa-promot-0-train_nature.json',
'iconqa_val':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/iconqa-promot-0-val_nature.json',
'iconqa_test':'Vision-PromptSource/prompt_data_8_11_vicuna_fewshot/iconqa-promot-0-test_nature.json',
}


data_size= {
    "vqa": {
        "train": 70000,
        "test": 1000,
        "val": 1000
    },
    "gqa": {
        "train": 70000,
        "test": 1000,
        "val": 1000
    },
    "okvqa": {
        "train": 9000,
        "test": 2000,
        "val": 2000
    },
    "nlvr2": {
        "train": 20000,
        "test": 500,
        "val": 500
    },
    "coco": {
        "train": 60000,
        "test": 1000,
        "val": 1000
    },
    "flickr": {
        "train": 40000,
        "test": 1000,
        "val": 1000
    },
    "vcr": {
        "train": 118000,
        "test": 2000,
        "val": 2000
    },
    "stvqa": {
        "train": 30000,
        "test": 0,
        "val": 0
    },
    "refcoco": {
        "train": 60000,
        "test": 0,
        "val": 0
    },

    "miniimage": {
        "train": 15000,
        "test": 500,
        "val": 500
    },
    "wikiart": {
        "train": 8000,
        "test": 500,
        "val": 500
    },
    "llava": {
        "train": 150000,
        "test": 0,
        "val": 0
    },
    "textvqa": {
        "train": 25000,
        "test": 0,
        "val": 0
    },
    "diffusiondb": {
        "train": 15000,
        "test": 0,
        "val": 0
    },
    "nocaps": {
        "train": 0,
        "test": 2500,
        "val": 2500
    },
    "msrvtt": {
        "train": 25000,
        "test": 1000,
        "val": 1000
    },
    "msrvttqa": {
        "train": 30000,
        "test": 1500,
        "val": 1500
    },
    "msrvttqa": {
        "train": 50000,
        "test": 1500,
        "val": 1500
    },
    "iconqa": {
        "train": 15000,
        "test": 1500,
        "val": 1500
    },
    
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
    if not os.path.exists(f'Vision-PromptSource/{save_dir_name}'):
        os.makedirs(f'Vision-PromptSource/{save_dir_name}')
    if not os.path.exists(f'Vision-PromptSource/{save_dir_name}/train'):
        os.makedirs(f'Vision-PromptSource/{save_dir_name}/train')
    if not os.path.exists(f'Vision-PromptSource/{save_dir_name}/val'):
        os.makedirs(f'Vision-PromptSource/{save_dir_name}/val')
    if not os.path.exists(f'Vision-PromptSource/{save_dir_name}/test'):
        os.makedirs(f'Vision-PromptSource/{save_dir_name}/test')
    for each in tqdm(data_size):
        js_train = sample(get_json_file(data_json[f'{each}_train']),data_size[each]['train']) if data_size[each]['train'] != -1 else get_json_file(data_json[f'{each}_train'])
        js_test = sample(get_json_file(data_json[f'{each}_test']),data_size[each]['test']) if data_size[each]['test'] != -1 else get_json_file(data_json[f'{each}_test'])
        js_val = sample(get_json_file(data_json[f'{each}_val']),data_size[each]['val']) if data_size[each]['val'] != -1 else get_json_file(data_json[f'{each}_val'])
        json_train.extend(js_train)
        json_test.extend(js_test)
        json_val.extend(js_val)
        # random.shuffle(json_train)
        # random.shuffle(json_test)
        # random.shuffle(json_val)
        if data_size[each]['train'] !=0:
            sample_num = data_size[each]['train']
            with open(f'Vision-PromptSource/{save_dir_name}/train/{file_name}-{each}-sample_{sample_num}-train.json','w') as f:
                for dic in js_train:
                    f.write(json.dumps(dic, ensure_ascii=False)+"\n") 
        if data_size[each]['test'] !=0:
            sample_num = data_size[each]['test']
            with open(f'Vision-PromptSource/{save_dir_name}/test/{file_name}-{each}-sample_{sample_num}-test.json','w') as f:
                for dic in js_test:
                    f.write(json.dumps(dic, ensure_ascii=False)+"\n") 
        if data_size[each]['val'] !=0:
            sample_num = data_size[each]['val']
            with open(f'Vision-PromptSource/{save_dir_name}/val/{file_name}-{each}-sample_{sample_num}-val.json','w') as f:
                for dic in js_val:
                    f.write(json.dumps(dic, ensure_ascii=False)+"\n") 


    # with open(f'Vision-PromptSource/{save_dir_name}/{file_name}-train.json','w') as f:
    #     for dic in json_train:
    #         f.write(json.dumps(dic, ensure_ascii=False)+"\n") 
    # with open(f'Vision-PromptSource/{save_dir_name}/{file_name}-test.json','w') as f:
    #     for dic in json_test:
    #         f.write(json.dumps(dic, ensure_ascii=False)+"\n") 
    # with open(f'Vision-PromptSource/{save_dir_name}/{file_name}-val.json','w') as f:
    #     for dic in json_val:
    #         f.write(json.dumps(dic, ensure_ascii=False)+"\n") 


def save_pickle_img(path,file):
    with open(join(path_dir,path),'ab') as f:
        pk.dump(file,f)

def extract_frames(video_path, num_frames):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    frame_times = [duration * i / (num_frames + 1) for i in range(1, num_frames + 1)]
    
    frames = []
    for t in frame_times:
        frame = clip.get_frame(t)
        image = Image.fromarray(frame)
        frames.append(image)
    
    clip.close()
    
    return frames

def read_image(postfix,img_path):
    if postfix == 'png':
        image = Image.open(join("Vision-PromptSource",img_path))
    elif postfix == 'h5':
        image = h5py.File(join("Vision-PromptSource",img_path), 'r')
    else:
        image = Image.open(join("Vision-PromptSource", img_path))
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
        img_path = input_image[1:] if input_image[0] == '.' and input_image[1] !='/'  else input_image
        img_path = img_path.replace('_/',"_")
        if postfix =='mp4':
            images = extract_frames(img_path,NUM_FRAMES)
            for img in images:
                result["pixel_values"].append(processor(images = img)["pixel_values"][0])
        else:

            img = read_image(postfix,img_path)
            result["pixel_values"].append(processor(images = img)["pixel_values"][0])
    return result
def concat_text_input_output( input_ids, input_atts, output_ids, output_atts):
    input_part_targets_len = []
    llm_tokens = {"input_ids": [], "attention_mask": []}
    this_input_ones = sum(input_atts)
    input_part_targets_len.append(this_input_ones)
    llm_tokens['input_ids'].append(
        np.concatenate([
            input_ids[:this_input_ones],
            output_ids[1:],
            input_ids[this_input_ones:]
        ])
    )
    llm_tokens['attention_mask'].append(
        np.concatenate([
            input_atts[:this_input_ones],
            output_atts[1:],
            input_atts[this_input_ones:]
        ])
    )
    llm_tokens['input_ids'] = np.stack(llm_tokens['input_ids'])
    llm_tokens['attention_mask'] = np.stack(llm_tokens['attention_mask'])
    return llm_tokens, input_part_targets_len



def preprocess_function_batched(result,input_text,output_text):
    if 'vicuna' in model_type:
        processor.tokenizer.padding_side = "right"
        processor.tokenizer.truncation_side = 'left'
        replace_token = "".join(32*[image_placeholder])
        
        input_text = input_text.replace('图',replace_token)
        re = processor.tokenizer(
            input_text,
            padding="longest",
            truncation=True,
            max_length=max_seq_length,
        )
        processor.tokenizer.truncation_side = 'right'
        out = processor.tokenizer(
            output_text,
            padding="longest",
            truncation=True,
            max_length=256)
        re, input_part_targets_len = concat_text_input_output(
        re['input_ids'],
        re['attention_mask'],
        out['input_ids'],
        out['attention_mask'],
        )
        re['input_ids'] = np.array(re['input_ids'],dtype=np.int32)
        re['attention_mask'] = np.array(re['attention_mask'],dtype=np.bool_)

        # do not apply loss to the padding
        targets = copy.deepcopy(re['input_ids'])
        targets[targets == processor.tokenizer.pad_token_id] = -100


        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        m= {
            'input_ids': re['input_ids'],
            'attention_mask': re['attention_mask'],
            'label': targets,
        }
        result.update(m)

    else:
        re= processor.tokenizer(input_text, padding='max_length', max_length=max_seq_length, truncation=True)
        re['input_ids'] = np.array(re['input_ids'],dtype=np.int32)
        re['attention_mask'] = np.array(re['attention_mask'],dtype=np.bool_)
        # result['label'] = np.array(processor.tokenizer(output_text, padding='max_length', max_length=32, truncation=True)["input_ids"],dtype=np.int32)
        out = processor.tokenizer(output_text, padding='max_length', max_length=128, truncation=True)
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
def process_raw_datajson_to_arrow(json_data,file_name,types,sub_length = -1,big_file_name=None,dataset_name=None):
    if big_file_name is None:
        big_file_name = file_name
    if dataset_name is not None:
        big_file_name = join(big_file_name,dataset_name)
    if not os.path.exists(f'{big_file_name}/arrow_data_{file_name}_{types}'):
        os.makedirs(f'{big_file_name}/arrow_data_{file_name}_{types}')
    if sub_length>0:
        json_data = json_data[:sub_length]
    save_arrow_data={'pixel_values':[], 'label':[], 'input_ids':[], 'attention_mask':[]}
    index_arrow=0
    threads = []
    for idx,each in enumerate(tqdm(json_data)):
        input_text = each['input_text']
        input_imgs = each['input_image']
        output_text = each['output_text']
        try:
            temp = preprocess_function(input_text,input_imgs,output_text)
        except Exception as e:
            print(e)
            continue
        temp = preprocess_function_batched(temp,input_text,output_text)
        for each in save_arrow_data:
            save_arrow_data[each].append(temp[each])
        if idx %1000 == 0 and idx !=0:
            if sub_length>0:
                path = f"{big_file_name}/arrow_data_{file_name}_{types}/bilp2-temp-{types}-{index_arrow}-length{sub_length}.arrow"
            else:
                path = f"{big_file_name}/arrow_data_{file_name}_{types}/bilp2-temp-{types}-{index_arrow}.arrow"
            t = threading.Thread(target=save_to_arrow, args=(path, save_arrow_data))
            threads.append(t)
            t.start()
            save_arrow_data={'pixel_values':[], 'label':[], 'input_ids':[], 'attention_mask':[]}
            index_arrow+=1
    for t in threads:
        t.join()
    if sub_length>0:
        path = f"{big_file_name}/arrow_data_{file_name}_{types}/bilp2-temp-{types}-{index_arrow}-length{sub_length}.arrow"
    else:
        path = f"{big_file_name}/arrow_data_{file_name}_{types}/bilp2-temp-{types}-{index_arrow}.arrow"
    save_to_arrow(path,save_arrow_data)
 



                                                                                                                                                                 
def to_pickle(file_name):
    train_js =f'Vision-PromptSource/prompt_data/{file_name}-train.json'
    test_js =f'Vision-PromptSource/prompt_data/{file_name}-test.json'
    val_js =f'Vision-PromptSource/prompt_data/{file_name}-val.json'
    print('start process training data')
    process_raw_datajson_to_pickle(train_js,'train')
    print('start process testing data')
    process_raw_datajson_to_pickle(test_js,'test')
    print('start process val data')
    process_raw_datajson_to_pickle(val_js,'val')

def to_arrow(file_name,length=-1,do_train = True,convert_file_name=None):
    if convert_file_name is None:
        convert_file_name = file_name
    train_js =f'Vision-PromptSource/prompt_data/{file_name}-train.json'
    test_js =f'Vision-PromptSource/prompt_data/{file_name}-test.json'
    val_js =f'Vision-PromptSource/prompt_data/{file_name}-val.json'
    if do_train:
        train_js = get_json_file(train_js)
        print('start process training data')
        process_raw_datajson_to_arrow(train_js,convert_file_name,'train',length)
    print('start process testing data')
    test_js = get_json_file(test_js)
    process_raw_datajson_to_arrow(test_js,convert_file_name,'test',length)
    print('start process val data')
    val_js = get_json_file(val_js)
    process_raw_datajson_to_arrow(val_js,convert_file_name,'val',length)

def zero_preprocess_json(json_file):
    re =[]
    for j in json_file:
        m={'output_image':""}
        if 'vcr' in j['input_image'][0]:
            m["input_text"] = f"image 0 is <image0>{image_placeholder}.\n"+j['input_text'].split(f'{image_placeholder}.\n')[-1]
            m["input_image"] = [j["input_image"][0]]
            m['output_text'] = j["output_text"]
        else:
            image_id = len(j['input_text'].split('\n'))-1
            m["input_text"] = (f"image 0 is <image0>{image_placeholder}.\n"+j['input_text'].split('\n')[-1]).replace(f'image {image_id}','image 0').replace(f'<image{image_id}>','<image0>')
            m["input_image"] = [j["input_image"][-1]]
            m['output_text'] = j["output_text"]

        replace_token = "".join(32*[image_placeholder])
        m['input_text'] = m['input_text'].replace(image_placeholder,replace_token)
        re.append(m)
    return re
def zero_shot_to_arrow(file_name,length=-1,do_train = True,convert_file_name=None):
    if convert_file_name is None:
        convert_file_name = file_name
    train_js =f'Vision-PromptSource/prompt_data/{file_name}-train.json'
    test_js =f'Vision-PromptSource/prompt_data/{file_name}-test.json'
    val_js =f'Vision-PromptSource/prompt_data/{file_name}-val.json'
    if do_train: 
        train_js = get_json_file(train_js)[:length] if length>0 else get_json_file(train_js)
        zero_train_js = zero_preprocess_json(train_js)
        print('start process training data')
        process_raw_datajson_to_arrow(zero_train_js,convert_file_name,'train_zeroshot',length)
    print('start process testing data')
    test_js = get_json_file(test_js)[:length] if length>0 else get_json_file(test_js)
    zero_test_js = zero_preprocess_json(test_js)
    process_raw_datajson_to_arrow(zero_test_js,convert_file_name,'test_zeroshot',length)
    print('start process val data')
    val_js = get_json_file(val_js)[:length] if length>0 else get_json_file(val_js)
    zero_val_js = zero_preprocess_json(val_js)
    process_raw_datajson_to_arrow(zero_val_js,convert_file_name,'val_zeroshot',length)

# generate_new_json(data_json,data_size,file_name)

# to_arrow(file_name,2000,False,convert_file_name)

# zero_shot_to_arrow(file_name,2000,False,convert_file_name)

# generate_jsonl_data_from_instances()
from multiprocessing import Pool
from functools import partial
def get_json_file(file_path):
    js =[]
    with open(file_path,'r')as f:
        for line in f.readlines():
            js.append(json.loads(line))
    return js
def process_train_data(train,length,big_file_name):
    if '-1' not in train:
        dataset_name = train.split('/')[-1].split('-')[-3]
    else:
        dataset_name = train.split('/')[-1].split('-')[-4]

    train_json = get_json_file(train)
    print(f'start process testing data {dataset_name}')
    process_raw_datajson_to_arrow(train_json, convert_file_name, 'train', length, big_file_name, dataset_name)

def process_val_data(val,length,big_file_name):
    if '-1' not in val:
        dataset_name = val.split('/')[-1].split('-')[-3]
    else:
        dataset_name = val.split('/')[-1].split('-')[-4]

    val_json = get_json_file(val)
    print(f'start process testing data {dataset_name}')
    process_raw_datajson_to_arrow(val_json, convert_file_name, 'val', length, big_file_name, dataset_name)

def process_test_data(test,length,big_file_name):
    if '-1' not in test:
        dataset_name = test.split('/')[-1].split('-')[-3]
    else:
        dataset_name = test.split('/')[-1].split('-')[-4]

    test_json = get_json_file(test)
    print(f'start process testing data {dataset_name}')
    process_raw_datajson_to_arrow(test_json, convert_file_name, 'test', length, big_file_name, dataset_name)


def to_arrowByDataset(file_name,length=-1,do_train = True,convert_file_name=None):
    if convert_file_name is None:
        convert_file_name = file_name
    big_file_name = f"{save_dir_name}/{file_name}"
    train_js =glob.glob(f'Vision-PromptSource/{save_dir_name}/train/*')
    test_js =glob.glob(f'Vision-PromptSource/{save_dir_name}/test/*')
    val_js =glob.glob(f'Vision-PromptSource/{save_dir_name}/val/*')

    # if do_train:
    #     for train in train_js:
    #         # if 'llava' in train or 'okvqa' in train or 'textvqa' in train or 'flickr' in train or 'vcr' in train or 'diffusiondb' in train or 'miniimage' in train or 'refcoco' in train or 'wikiart' in train:
    #         #     # ocr-vqa & miniimage wikiart not finished
    #         #     continue
    #         # if  'msrvtt' in train :
    #         dataset_name = train.split('/')[-1].split('-')[-3]
    #         train_json = get_json_file(train)
    #         print(f'start process training data {dataset_name}')
    #         process_raw_datajson_to_arrow(train_json,convert_file_name,'train',length,big_file_name,dataset_name)
    # for test in test_js:
    #             # ocr-vqa & miniimage not finished
    #             # continue
    #     dataset_name = test.split('/')[-1].split('-')[-4]
    #     # if  'nocaps' == dataset_name:

    #     test_json = get_json_file(test)
    #     print(f'start process testing data {dataset_name}')
    #     process_raw_datajson_to_arrow(test_json,convert_file_name,'test',length,big_file_name,dataset_name)
# 创建进程池

    with Pool() as pool:
        # 使用进程池的map方法并行处理任务
        partial_process_train_data = partial(process_train_data, length=length, big_file_name=big_file_name)
        pool.map(partial_process_train_data, train_js)
    
        partial_process_test_data = partial(process_test_data, length=length, big_file_name=big_file_name)
        pool.map(partial_process_test_data, test_js)
    
        partial_process_val_data = partial(process_val_data, length=length, big_file_name=big_file_name)
        pool.map(partial_process_val_data, val_js)
 
    # for val in val_js:
    #     # if  'msrvttqa' in val or 'msrvtt' in val :
    #             # ocr-vqa & miniimage not finished
    #             # continue
    #     dataset_name = val.split('/')[-1].split('-')[-3]
    #     val_json = get_json_file(val)
    #     print(f'start process val data {dataset_name}')
    #     process_raw_datajson_to_arrow(val_json,convert_file_name,'val',length,big_file_name,dataset_name)
import warnings
warnings.filterwarnings("ignore")
model_type='vicuna'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
path_dir = 'Vision-PromptSource/'
if 'vicuna' in model_type:
    image_placeholder = "<visual_embedding>"
else:
    image_placeholder = "图" 
sample_number = 5
if 'vicuna' in model_type:
    model_name_or_path = 'Salesforce/instructblip-vicuna-7b'
else:
    model_name_or_path = 'Salesforce/instructblip-flan-t5-xxl'


processor = InstructBlipProcessor.from_pretrained(
    model_name_or_path,
)
if 'vicuna' in model_type:
    sp = [image_placeholder]+[f"<image{i}>" for i in range(20)]
    processor.tokenizer.add_special_tokens({'additional_special_tokens':sp})
    max_seq_length = min(2048, processor.tokenizer.model_max_length)
else:
    sp = [image_placeholder]+[f"<image{i}>" for i in range(20)]
    sp = sp+processor.tokenizer.additional_special_tokens[len(sp):]
    processor.tokenizer.add_special_tokens({'additional_special_tokens':sp})
    max_seq_length = min(512, processor.tokenizer.model_max_length)


file_name = 'prompt-allshot-multiinst_final_ver'
convert_file_name = 'prompt-allshot-multiinst_final_ver'

seed= 100
if seed is not None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    

save_dir_name='prompt_data_8_11_max_figure5_vicuna_json'
store_path='prompt_data_8_11_vicuna_fewshot'
generate_jsonl_data_from_instances(store_path)
generate_new_json(data_json,data_size,file_name)
NUM_FRAMES=8
to_arrowByDataset(file_name)
