import math
import torch
import torch.nn as nn
from utils.transformers import Encoder, crossEncoder

class EmbeddingEncoder(torch.nn.Module):
    def __init__(self, config, model_args, model):
        super().__init__()
        self.config = config
        self.model_args = model_args
        self.model = model
        self.prompt_encoder = PromptEncoder(self.config, self.model_args, self.model)

    def id2embedding(self, input_ids, sentences_ids):

        input_ids = input_ids.to(self.model.device)
        sentences_ids = sentences_ids.to(self.model.device)

        # construct query ids
        batch_size = input_ids.shape[0]

        if self.model_args.prompt_operation in ["attention", "cross-attention"]:
            attention_mask_sentences_ids = sentences_ids != self.model_args.tokenizer.pad_token_id # batch_size * seq_length
            prompts = self.prompt_encoder(sentences_ids, attention_mask_sentences_ids) # batch_size * pre_seq_length * hidden_size
        elif self.model_args.prompt_operation in ["max", "mean", "sum"]:
            prompts = self.prompt_encoder(sentences_ids) # batch_size * max_seq_len
            prompts = torch.ones(batch_size, self.model_args.pre_seq_len, self.config.hidden_size).to(self.model.device)
            for batch_id in range(batch_size):
                for seq_id in range(self.model_args.pre_seq_len):
                    prompts[batch_id, seq_id, :] = self.prompt_encoder(sentences_ids[batch_id][sentences_ids[batch_id] != self.model_args.tokenizer.pad_token_id])[:] # 这一步已经把pad_token_id排除了。
        elif self.model_args.prompt_operation in ["none"]:
            prompts = torch.ones(batch_size, self.model_args.pre_seq_len, self.config.hidden_size).to(self.model.device)
            prompts_replace = self.prompt_encoder()
            for batch_id in range(batch_size):
                for seq_id in range(self.model_args.pre_seq_len):
                    prompts[batch_id, seq_id, :] = prompts_replace if self.model_args.pre_seq_len == 1 else prompts_replace[seq_id, :]
        else:
            raise NotImplementedError("The prompt_operation for {} has not been defined.".format(self.model_args.prompt_operation))

        attention_mask = input_ids != self.model_args.tokenizer.pad_token_id # batch_size * seq_length

        # get embedded input
        # inputs_embeds = self.embed_input(input_ids, prompts, self.args) # batch_size * max_seq_length * embedding_dim
        input_ids_for_embedding = input_ids.clone()
        input_ids_for_embedding[(input_ids == self.model_args.tokenizer.additional_special_tokens_ids[0])] = self.model_args.tokenizer.unk_token_id # 转化[PROMPT]_id为[UNK]_id
        inputs_embeds = self.model.get_input_embeddings()(input_ids_for_embedding) # 转化token_id [batch_size, seq_len]为 embedding [batch_size, seq_len, embedding_dim]

        if self.model_args.prompt_type == "none":
            pass
        else:
            # blocked_indices = (input_ids == self.model_args.tokenizer.additional_special_tokens_ids[0]).nonzero().reshape((batch_size, self.model_args.pre_seq_len, -1))[:, :, 1]  # 把[PROMPT]标记为1，反转，[PROMPT]标记为0，返回的是[PROMPT]的索引
            blocked_indices = torch.nonzero(input_ids == self.model_args.tokenizer.additional_special_tokens_ids[0]).reshape((batch_size, self.model_args.pre_seq_len, -1))[:, :, 1]  # 把[PROMPT]标记为1，反转，[PROMPT]标记为0，返回的是[PROMPT]的索引
            for batch_id in range(batch_size):
                for seq_id in range(self.model_args.pre_seq_len):
                    inputs_embeds[batch_id, blocked_indices[batch_id, seq_id], :] = prompts[batch_id, seq_id, :]

        return inputs_embeds


class PromptEncoder(torch.nn.Module):
    def __init__(self, config, model_args, model):
        super().__init__()
        self.config = config
        self.model_args = model_args
        self.model = model

        if self.config.model_type == "albert":
            self.config.hidden_size = self.config.embedding_size

        # self.cloze_length = template
        # self.cloze_mask = [
        #     [1] * self.cloze_length[0]  # first cloze
        #     + [1] * self.cloze_length[1]  # second cloze
        #     + [1] * self.cloze_length[2]
        # ]
        self.cloze_mask = [[1] * self.model_args.pre_seq_len] #[[1, 1, 1, 1]]
        self.cloze_mask = torch.LongTensor(self.cloze_mask) #[[True, True, True]]

        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))) # [[0, 1, 2, 3, 4]]
        # embedding
        self.nomal_embeddings = torch.nn.Embedding(self.model_args.pre_seq_len, self.config.hidden_size)
        # embedding
        self.specific_embeddings = self.model.get_input_embeddings()
        # LSTM
        self.lstm_head = torch.nn.LSTM(input_size=self.config.hidden_size,
                                       hidden_size=self.config.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.model_args.hidden_dropout_prob,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.config.hidden_size, self.config.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.config.hidden_size, self.config.hidden_size),
                                      nn.ReLU())
        self.trans = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(config.hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
        )

        if self.model_args.prompt_operation == "attention":
            self.global_embeded = nn.Parameter(torch.Tensor(self.model_args.pre_seq_len, self.config.hidden_size)).to(self.model.device) # pre_seq_len, hidden_size
            nn.init.kaiming_uniform_(self.global_embeded, a=math.sqrt(5))

            self.transformers = Encoder(
                max_seq_len=self.config.max_position_embeddings,
                num_layers=self.model_args.num_attention_layers,
                model_dim=self.config.hidden_size,
                num_heads=self.model_args.num_attention_heads,
                ffn_dim=self.config.intermediate_size,
                dropout=self.model_args.hidden_dropout_prob,
                whether_PositionalEncoding=self.model_args.whether_PositionalEncoding,
                whether_PositionalWiseFeedForward=self.model_args.whether_PositionalWiseFeedForward,
                )
        elif self.model_args.prompt_operation == "cross-attention":
            self.global_embeded = nn.Parameter(torch.Tensor(self.model_args.pre_seq_len, self.config.hidden_size)).to(self.model.device) # pre_seq_len, hidden_size
            nn.init.kaiming_uniform_(self.global_embeded, a=math.sqrt(5))
            self.transformers = crossEncoder(
                max_seq_len=self.config.max_position_embeddings,
                num_layers=self.model_args.num_attention_layers,
                model_dim=self.config.hidden_size,
                num_heads=self.model_args.num_attention_heads,
                ffn_dim=self.config.intermediate_size,
                dropout=self.model_args.hidden_dropout_prob,
                whether_PositionalEncoding=self.model_args.whether_PositionalEncoding,
                whether_PositionalWiseFeedForward=self.model_args.whether_PositionalWiseFeedForward,
            )
        print(f'init {self.model_args.prompt_operation} prompt encoder')

    def forward(self, sentences_encoded=None, attention_mask=None):
        if sentences_encoded != None:
            batch_size = sentences_encoded.size(0)
        if self.model_args.prompt_operation in ["attention"]:
            global_embeded = self.global_embeded.unsqueeze(0).repeat(batch_size, 1, 1)# batch_size * pre_seq_len, hidden_size
            sentences_embeded = self.specific_embeddings(sentences_encoded).detach() # batch_size * seq_len * hidden_size
            cat_embeded = torch.cat((global_embeded, sentences_embeded), dim=1) # batch_size * total_seq_len * hidden_size
            cat_attention = torch.cat((torch.ones(batch_size, self.model_args.pre_seq_len).bool().to(self.model.device), attention_mask), dim=1) # batch_size * total_seq_len * hidden_size
            prompts, _ = self.transformers(cat_embeded, cat_attention) # batch_size * total_seq_len * hidden_size
            if self.model_args.task_type == "sequence_classification":
                prompts = self.trans(prompts)
            result = prompts[:, :self.model_args.pre_seq_len, :] # batch_size * pre_seq_len * hidden_size
            return result
        elif self.model_args.prompt_operation in ["cross-attention"]:
            global_embeded = self.global_embeded.unsqueeze(0).repeat(batch_size, 1, 1)# batch_size * pre_seq_len, hidden_size
            sentences_embeded = self.specific_embeddings(sentences_encoded).detach() # batch_size * seq_len * hidden_size
            prompts, _ = self.transformers(global_embeded, sentences_embeded, attention_mask) # batch_size * total_seq_len * hidden_size
            if self.model_args.task_type == "sequence_classification":
                prompts = self.trans(prompts)
            return prompts
        elif self.model_args.prompt_operation in ["mean", "sum", "max"]:
            sentences_embeded = self.specific_embeddings(sentences_encoded).detach() # seq_len * hidden_size
            output_embeded = self.mlp_head(sentences_embeded).squeeze() # seq_len * hidden_size
            if self.model_args.prompt_operation == "mean":
                result = torch.mean(output_embeded, 0) # hidden_size
            elif self.model_args.prompt_operation == "sum":
                result = torch.sum(output_embeded, 0) # hidden_size
            elif self.model_args.prompt_operation == "max":
                result = torch.max(output_embeded, 0).values # hidden_size
            elif self.model_args.prompt_operation == "attention":
                pass
            else:
                raise NotImplementedError("The prompt_operation for {} has not been defined.".format(self.model_args.prompt_operation))
            return result
        elif self.model_args.prompt_operation in ["none"]:
            input_embeds = self.nomal_embeddings(self.seq_indices.to(self.model.device)).detach().unsqueeze(0) # seq_len * hidden_size
            # LSTM_embeds = self.lstm_head(input_embeds) # seq_len * hidden_size
            output_embeds = self.mlp_head(input_embeds).squeeze() # seq_len * hidden_size
            return output_embeds
        else:
            raise NotImplementedError("The prompt_operation for {} has not been defined.".format(self.model_args.prompt_operation))

