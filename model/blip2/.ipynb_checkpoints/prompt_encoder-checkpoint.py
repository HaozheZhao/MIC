import math
import torch
import torch.nn as nn
from model.transformers import Encoder


class PromptEncoder(torch.nn.Module):
    def __init__(self, config, model_args, model):
        super().__init__()
        self.config = config
        self.model_args = model_args
        self.model = model

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
        self.transformers = Encoder(
            max_seq_len=self.config.max_position_embeddings,
            num_layers=self.model_args.num_attention_layers,
            model_dim=self.config.hidden_size,
            num_heads=self.model_args.num_attention_heads,
            ffn_dim=self.config.intermediate_size,
            dropout=self.model_args.hidden_dropout_prob,
            whether_PositionalEncoding=self.model_args.whether_PositionalEncoding,
            whether_PositionalWiseFeedForward=self.model_args.whether_PositionalWiseFeedForward
        )
        self.trans = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(config.hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
        )
        self.global_embeded = nn.Parameter(torch.Tensor(self.model_args.pre_seq_len, self.config.hidden_size)).to(self.model.device) # pre_seq_len, hidden_size
        nn.init.kaiming_uniform_(self.global_embeded, a=math.sqrt(5))
        print("init prompt encoder...")

    def forward(self, sentences_encoded=None, attention_mask=None):
        if self.model_args.prompt_operation in ["attention"]:
            batch_size = sentences_encoded.size(0)
            global_embeded = self.global_embeded.unsqueeze(0).repeat(batch_size, 1, 1)# batch_size * pre_seq_len, hidden_size
            sentences_embeded = self.specific_embeddings(sentences_encoded).detach() # batch_size * seq_len * hidden_size
            cat_embeded = torch.cat((global_embeded, sentences_embeded), dim=1) # batch_size * total_seq_len * hidden_size
            cat_attention = torch.cat((torch.ones(batch_size, self.model_args.pre_seq_len).bool().to(self.model.device), attention_mask), dim=1) # batch_size * total_seq_len * hidden_size
            prompts, _ = self.transformers(cat_embeded, cat_attention) # batch_size * total_seq_len * hidden_size
            if self.model_args.task_type == "sequence_classification":
                prompts = self.trans(prompts)
            result = prompts[:, :self.model_args.pre_seq_len, :] # batch_size * pre_seq_len * hidden_size
            return result
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

