import torch
from torch._C import NoopLogger
import torch.nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from transformers.modeling_outputs import MaskedLMOutput

from transformers import BertModel, BertPreTrainedModel
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput, Seq2SeqLMOutput

from model.prefix_encoder import PrefixEncoder
from model.prompt_encoder import PromptEncoder

import copy

class BertForMaskedLM(BertPreTrainedModel):
    
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # # Prompt Parameters
        # self.pre_seq_len = config.pre_seq_len
        # self.n_layer = config.num_hidden_layers
        # self.n_head = config.num_attention_heads
        # self.n_embd = config.hidden_size // config.num_attention_heads

        # self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        # self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len, config.hidden_size)

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     processor_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=MaskedLMOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        *args
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class BertPromptForMaskedLM(BertPreTrainedModel):
    
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config, model_args):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        # self.post_init()
        self.init_weights()
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # # Prompt Parameters
        self.config=config
        self.model_args=model_args

        # self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        # self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len, config.hidden_size)
        self.prompt_encoder = PromptEncoder(self.config, self.model_args, self.bert)

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embedding, self.model_args.pre_seq_len
        
    # @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     processor_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=MaskedLMOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        input_ids=None,
        sentences_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None, # [122, 21212, 2121, 222, 2112, 321] trainer eval的时候用的
        label_ids=None, # [122, 21212, 2121, 222, 2112, 321] 计算loss用的
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        label_token_id_list=None, # [123, 9292]
        label_token_id=None, # 123
        label=None, # 0 和labels一样
        mask_index=None # 45
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # raw_embedding = self.bert.get_input_embeddings()(
        #     input_ids=input_ids
        # )
        # prompts = self.get_prompt(batch_size=batch_size)
        # inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)
        # prefix_attention_mask = torch.ones(batch_size, self.model_args.pre_seq_len).to(
        #     self.bert.device
        # )
        # attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        # NOTE: Add code to implement prompt embedding
        # construct query ids
        batch_size = input_ids.shape[0]

        # construct query ids
        if isinstance(label_token_id_list[0][0].item(), list):
            num_mask_token = max(len(length.item()) for length in label_token_id_list[0])
        elif isinstance(label_token_id_list[0][0].item(), int):
            num_mask_token = 1

        if self.model_args.prompt_operation in ["attention"]:
            attention_mask_sentences_ids = sentences_ids != self.model_args.tokenizer.pad_token_id # batch_size * seq_length
            prompts = self.prompt_encoder(sentences_ids, attention_mask_sentences_ids) # batch_size * pre_seq_length * hidden_size
        elif self.model_args.prompt_operation in ["max", "mean", "sum"]:
            prompts = self.prompt_encoder(sentences_ids) # batch_size * max_seq_len
            prompts = torch.ones(batch_size, self.model_args.pre_seq_len, self.config.hidden_size).to(self.bert.device)
            for batch_id in range(batch_size):
                for seq_id in range(self.model_args.pre_seq_len):
                    prompts[batch_id, seq_id, :] = self.prompt_encoder(sentences_ids[batch_id][sentences_ids[batch_id] != self.model_args.tokenizer.pad_token_id])[:] # 这一步已经把pad_token_id排除了。
        elif self.model_args.prompt_operation in ["none"]:
            prompts = torch.ones(batch_size, self.model_args.pre_seq_len, self.config.hidden_size).to(self.bert.device)
            prompts_replace = self.prompt_encoder()
            for batch_id in range(batch_size):
                for seq_id in range(self.model_args.pre_seq_len):
                    prompts[batch_id, seq_id, :] = prompts_replace if self.model_args.pre_seq_len == 1 else prompts_replace[seq_id, :]
        else:
            raise NotImplementedError("The prompt_operation for {} has not been defined.".format(self.model_args.prompt_operation))

        attention_mask = input_ids != self.model_args.tokenizer.pad_token_id # batch_size * seq_length

        # get embedded input
        # inputs_embeds = self.embed_input(input_ids, prompts, self.args) # batch_size * max_seq_length * embedding_dim
        input_for_embedding = input_ids.clone()
        input_for_embedding[(input_ids == self.model_args.tokenizer.additional_special_tokens_ids[0])] = self.model_args.tokenizer.unk_token_id # 转化[PROMPT]_id为[UNK]_id
        inputs_embeds = self.bert.get_input_embeddings()(input_for_embedding) # 转化token_id [batch_size, seq_len]为 embedding [batch_size, seq_len, embedding_dim]

        if self.model_args.prompt_type == "none":
            pass
        else:
            # blocked_indices = (input_ids == self.model_args.tokenizer.additional_special_tokens_ids[0]).nonzero().reshape((batch_size, self.model_args.pre_seq_len, -1))[:, :, 1]  # 把[PROMPT]标记为1，反转，[PROMPT]标记为0，返回的是[PROMPT]的索引
            blocked_indices = torch.nonzero(input_ids == self.model_args.tokenizer.additional_special_tokens_ids[0]).reshape((batch_size, self.model_args.pre_seq_len, -1))[:, :, 1]  # 把[PROMPT]标记为1，反转，[PROMPT]标记为0，返回的是[PROMPT]的索引
            for batch_id in range(batch_size):
                for seq_id in range(self.model_args.pre_seq_len):
                    inputs_embeds[batch_id, blocked_indices[batch_id, seq_id], :] = prompts[batch_id, seq_id, :]

        # get the label input
        # labels = torch.empty_like(batch_queries).fill_(-100).long().to(self.bert.device)  # batch_size * max_sql_length
        
        # label_mask = (input_ids == self.model_args.tokenizer.mask_token_id).nonzero().reshape(batch_size, num_mask_token, -1)[:, :, -1].to(self.bert.device) # batch_size * num_mask
        label_mask = torch.nonzero(input_ids == self.model_args.tokenizer.mask_token_id).reshape(batch_size, num_mask_token, -1)[:, :, -1].to(self.bert.device) # batch_size * num_mask
        
        # labels = torch.empty_like(batch_queries).fill_(-100).long().to(self.bert.device) # batch_size * max_sql_length
        # labels = labels.scatter_(1, label_mask, label_ids) # batch_size * max_sql_length


        outputs = self.bert(
            # input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            # head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            # encoder_hidden_states=encoder_hidden_states,
            # encoder_attention_mask=encoder_attention_mask,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        # 这个地方的logit要取到只有标签
        # 要改成soft label也是在这里改

        # label_token_id
        # label_ids_all = torch.LongTensor([[self.model_args.tokenizer.encode(l)[1:-1] for l in self.labels] for _ in range(batch_size)]).squeeze(2).to(self.bert.device) # batch_size * num_labels

        # label_mask = (input_ids == self.model_args.tokenizer.mask_token_id).nonzero().reshape(batch_size, num_mask_token, -1)[:, :, -1].to(self.bert.device) # batch_size * num_mask
        label_mask = torch.nonzero(input_ids == self.model_args.tokenizer.mask_token_id).reshape(batch_size, num_mask_token, -1)[:, :, -1].to(self.bert.device) # batch_size * num_mask
        
        vocab_size = prediction_scores.shape[-1] # vocab_size
        index = label_mask.unsqueeze(2).repeat(1, 1, vocab_size).long() # batch_size * 1 * vocab_size
        y_pred = torch.gather(prediction_scores, index=index, dim=1).squeeze(1) # batch_size * vocab_size
        y_pred = torch.gather(y_pred, index=label_token_id_list, dim=1) # batch_size  * num_labels
        # y_rank = y_pred.argmax(axis=1) # batch_size
        # 这个地方不用argmax，留下带有两个选项的logit值的tensor就好。因为compute_metric自己会argmax。
        if self.model_args.multiple_choice:
            y_pred = y_pred[:, 0].unsqueeze(1).reshape(batch_size // 2, 2).repeat(1, 2).reshape(batch_size, 2)
        # print(y_pred.shape)

        masked_lm_loss = None
        # TODO: 这个地方可以尝试改成只有label词汇计算loss

        if label_ids is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), label_ids.view(-1))
        # print(masked_lm_loss) # 这个用来看模型train的时候loss有没有降低

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            # logits=prediction_scores,
            logits=y_pred,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.model_args.tokenizer.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.model_args.tokenizer.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

class BertPrefixForMaskedLM(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     processor_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=MaskedLMOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
    
    
