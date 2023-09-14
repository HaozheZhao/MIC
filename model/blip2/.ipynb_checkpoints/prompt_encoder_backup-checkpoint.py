import torch
import torch.nn as nn


class PromptEncoder(torch.nn.Module):
    def __init__(self, tokenizer, model, hidden_size, pre_seq_len, prompt_operation, hidden_dropout_prob):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.hidden_size = hidden_size
        self.pre_seq_len = pre_seq_len
        self.hidden_dropout_prob = hidden_dropout_prob
        self.prompt_operation = prompt_operation
        
        # self.cloze_length = template
        # self.cloze_mask = [
        #     [1] * self.cloze_length[0]  # first cloze
        #     + [1] * self.cloze_length[1]  # second cloze
        #     + [1] * self.cloze_length[2]
        # ]
        self.cloze_mask = [[1] * self.pre_seq_len] #[[1, 1, 1, 1]]
        self.cloze_mask = torch.LongTensor(self.cloze_mask) #[[True, True, True]]

        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))) # [[0, 1, 2, 3, 4]]
        # embedding
        self.nomal_embeddings = torch.nn.Embedding(self.pre_seq_len, self.hidden_size)
        # embedding
        self.specific_embeddings = self.model.get_input_embeddings()
        # LSTM
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.hidden_dropout_prob,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU())
        print("init prompt encoder...")

    def forward(self, sentences_encoded=None):
        if self.prompt_operation in ["mean", "sum", "max", "attention"]:
            sentences_embeded = self.specific_embeddings(sentences_encoded).detach() # seq_len * hidden_size
            output_embeded = self.mlp_head(sentences_embeded).squeeze() # seq_len * hidden_size
            # for seq_id in range(sentences_embeded.shape[0]):
            #     output_embeded[seq_id, :] = self.mlp_head(sentences_embeded[seq_id, :]).to(self.device)
            if self.prompt_operation == "mean":
                if output_embeded.dim() == 1:
                    result = output_embeded
                else:
                    result = torch.mean(output_embeded, 0) # hidden_size
            elif self.prompt_operation == "sum":
                result = torch.sum(output_embeded, 0) # hidden_size
            elif self.prompt_operation == "max":
                result = torch.max(output_embeded, 0).values # hidden_size
            elif self.prompt_operation == "attention":
                pass
            else:
                raise NotImplementedError("The prompt_operation for {} has not been defined.".format(self.prompt_operation))
            return result
        elif self.prompt_operation in ["none"]:
            input_embeds = self.nomal_embeddings(self.seq_indices.to(self.model.device)).detach().unsqueeze(0) # seq_len * hidden_size
            # LSTM_embeds = self.lstm_head(input_embeds) # seq_len * hidden_size
            output_embeds = self.mlp_head(input_embeds).squeeze() # seq_len * hidden_size
            return output_embeds
        else:
            raise NotImplementedError("The prompt_operation for {} has not been defined.".format(self.prompt_operation))
            
