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

MODEL_CLASS = {
    "blip-2": Blip2ForConditionalGeneration
}


def get_model(model_args, config: AutoConfig, fix_bert: bool = False):

    model_class = MODEL_CLASS[config.model_type]
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=config
    )

    bert_param = 0
    if fix_bert:
        if config.model_type == "bert":
            for param in model.bert.parameters():
                param.requires_grad = False
            for _, param in model.bert.named_parameters():
                bert_param += param.numel()
        elif config.model_type == "roberta":
            for param in model.roberta.parameters():
                param.requires_grad = False
            for _, param in model.roberta.named_parameters():
                bert_param += param.numel()

    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
    total_param = all_param - bert_param

    for param in model.parameters():
        param.requires_grad = False
        model.query_tokens.requires_grad = True
    for param in model.qformer.parameters():
        param.requires_grad = True
    for param in model.language_projection.parameters():
        param.requires_grad = True
    # for param in model.vision_model.encoder.layers[-3:].parameters():
    #     param.requires_grad = True
        
    for param in model.language_model.decoder.block[-3:].parameters():
        param.requires_grad = True
    print('***** total param is {} *****'.format(total_param))
    return model
