import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):   
        """
        假设 hidden_size = 768, num_attention_heads = 8, dropout_prob = 0.2
        即隐层维度为768，注意力头设置为8个
        """
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:   # 整除
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        # 参数定义
        self.num_attention_heads = num_attention_heads    # 8
        self.attention_head_size = int(hidden_size / num_attention_heads)  # 96  每个注意力头的维度
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size)   
        # all_head_size = 128 即等于hidden_size, 一般自注意力输入输出前后维度不变
        
        # query, key, value 的线性变换（上述公式2）
        self.query = nn.Linear(hidden_size, self.all_head_size)    # 768, 768
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # dropout
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        # INPUT:  x'shape = [bs, seqlen, hid_size]  假设hid_size=768
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # [bs, seqlen, 8, 96]
        x = x.view(*new_x_shape)   # 
        return x.permute(0, 2, 1, 3)   # [bs, 8, seqlen, 16]

    def forward(self, hidden_states, attention_mask):
        # eg: attention_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])  shape=[bs, seqlen]
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)   # [bs, 1, 1, seqlen] 增加维度
        attention_mask = (1.0 - attention_mask) * -10000.0   # padding的token置为-10000，exp(-1w)=0
        
        # 线性变换
        mixed_query_layer = self.query(hidden_states)   # [bs, seqlen, hid_size]
        mixed_key_layer = self.key(hidden_states)       # [bs, seqlen, hid_size]
        mixed_value_layer = self.value(hidden_states)   # [bs, seqlen, hid_size]

        query_layer = self.transpose_for_scores(mixed_query_layer)    # [bs, 8, seqlen, 96]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)   # [bs, 8, seqlen, 96]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # 计算query与title之间的点积注意力分数，还不是权重（个人认为权重应该是和为1的概率分布）
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, 8, seqlen, seqlen]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)   # [bs, 8, seqlen, seqlen]
        # 除以根号注意力头的数量，可看原论文公式，防止分数过大，过大会导致softmax之后非0即1
        attention_scores = attention_scores + attention_mask
        # 加上mask，将padding所在的表示直接-10000

        # 将注意力转化为概率分布，即注意力权重
        attention_probs = nn.Softmax(dim=-1)(attention_scores)    # [bs, 8, seqlen, seqlen]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        
        # 矩阵相乘，[bs, 8, seqlen, seqlen]*[bs, 8, seqlen, 96] = [bs, 8, seqlen, 96]
        context_layer = torch.matmul(attention_probs, value_layer)   # [bs, 8, seqlen, 16]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()   # [bs, seqlen, 8, 16]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)   # [bs, seqlen, 128]
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer    # [bs, seqlen, 768] 得到输出