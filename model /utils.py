'''
Author: JustBluce 972281745@qq.com
Date: 2022-11-24 08:58:23
LastEditors: JustBluce 972281745@qq.com
LastEditTime: 2023-02-18 09:53:29
FilePath: /DialogueVersionConteol/model/utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from transformers import (
    BertModel,
    RobertaModel,
    AlbertModel,
    DebertaV2Model,
    XLNetModel,
    DebertaV2Model,
    AutoConfig
)

from model.blip2.modeling_blip_2 import Blip2ForConditionalGeneration
from model.instructblip.modeling_instructblip import InstructBlipForConditionalGeneration

MODEL_CLASS = {
    "blip-2": Blip2ForConditionalGeneration,
    "instructblip": InstructBlipForConditionalGeneration,

}


def get_model(model_args, config: AutoConfig, fix_bert: bool = False):

    model_class = MODEL_CLASS[config.model_type]
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=config
    )

    for param in model.parameters():
        param.requires_grad = False
        # model.query_tokens.requires_grad = True
    # for param in model.qformer.parameters():
    #     param.requires_grad = True
    for param in model.language_projection.parameters():
        param.requires_grad = True
    # for param in model.vision_model.encoder.layers[-3:].parameters():
    #     param.requires_grad = True
        
    # for param in model.language_model.decoder.block[-3:].parameters():
    #     param.requires_grad = True

    for block in model.language_model.encoder.block:
        block.layer[0].SelfAttention.q.weight.requires_grad=True
        block.layer[0].SelfAttention.v.requires_grad=True

    for block in model.language_model.decoder.block:
        block.layer[0].SelfAttention.q.weight.requires_grad=True
        block.layer[0].SelfAttention.v.requires_grad=True
        block.layer[1].EncDecAttention.q.requires_grad=True
        block.layer[1].EncDecAttention.v.requires_grad=True
    
    all_param = 0
    trained_param=0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad ==True:
            trained_param+=param.numel()
    total_param = all_param 

    print('***** total param is {} *****'.format(total_param))
    print('***** total trained param is {} *****'.format(trained_param))
    return model
