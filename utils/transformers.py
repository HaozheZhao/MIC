import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播.

        Args:
            q: Queries张量，形状为[B, L_q, D_q]
            k: Keys张量，形状为[B, L_k, D_k]
            v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
            scale: 缩放因子，一个浮点标量
            attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
            上下文张量和attetention张量
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            # 给需要mask的地方设置一个负无穷
            attention = attention.masked_fill_(attn_mask, float("-inf"))
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        # multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        # 残差连接
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
        query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention
    
    
def residual(sublayer_fn,x):
    return sublayer_fn(x)+x



class LayerNorm(nn.Module):
    """实现LayerNorm。其实PyTorch已经实现啦，见nn.LayerNorm。"""

    def __init__(self, features, epsilon=1e-6):
        """Init.

        Args:
            features: 就是模型的维度。论文默认512
            epsilon: 一个很小的数，防止数值计算的除0错误
        """
        super(LayerNorm, self).__init__()
        # alpha
        self.gamma = nn.Parameter(torch.ones(features))
        # beta
        self.beta = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x):
        """前向传播.

        Args:
            x: 输入序列张量，形状为[B, L, D]
        """
        # 根据公式进行归一化
        # 在X的最后一个维度求均值，最后一个维度就是模型的维度
        mean = x.mean(-1, keepdim=True)
        # 在X的最后一个维度求方差，最后一个维度就是模型的维度
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta
    
    
def padding_mask(seq_k, seq_q):
    # seq_k和seq_q的形状都是[B,L]
    len_q = seq_q.size(1) # L_q
    # `PAD` is 0
    pad_mask = seq_k.eq(0) # batch_size * L_k
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_seq_len):
        """初始化。
        
        Args:
            d_model: 一个标量。模型的维度，论文默认是512
            max_seq_len: 一个标量。文本序列的最大长度
        """
        super(PositionalEncoding, self).__init__()
        
        # 根据论文给的公式，构造出PE矩阵
        position_encoding = np.array([[pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)] for pos in range(max_seq_len)])
        # 偶数列使用sin，奇数列使用cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        # 在PE矩阵的第一行，加上一行全是0的向量，代表这`PAD`的positional encoding
        # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
        # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
        # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
        pad_row = torch.zeros([1, d_model]).double()
        position_encoding = torch.cat((pad_row, torch.tensor(position_encoding)))
        
        # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
        # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)
    def forward(self, input_len, max_len):
        """神经网络的前向传播。

        Args:
          input_len: 一个张量，形状为[BATCH_SIZE, 1]。每一个张量的值代表这一批文本序列中对应的长度。

        Returns:
          返回这一批序列的位置编码，进行了对齐。
        """
        
        # 找出这一批序列的最大长度
        # max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
        # 这里range从1开始也是因为要避开PAD(0)的位置
        input_pos = tensor(
          [list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
        return self.position_encoding(input_pos)

class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output

class EncoderLayer(nn.Module):
    """Encoder的一层。"""

    def __init__(self, model_dim=768, num_heads=8, ffn_dim=3072, dropout=0.0, whether_PositionalWiseFeedForward=True):
        super(EncoderLayer, self).__init__()
        self.whether_PositionalWiseFeedForward = whether_PositionalWiseFeedForward

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):

        # self attention
        output, attention = self.attention(inputs, inputs, inputs, attn_mask)

        # feed forward network
        if self.whether_PositionalWiseFeedForward:
            output = self.feed_forward(output)

        return output, attention


class Encoder(nn.Module):
    """多层EncoderLayer组成Encoder。"""

    def __init__(self,
                 max_seq_len,
                 num_layers=6,
                 model_dim=768,
                 num_heads=8,
                 ffn_dim=3072,
                 dropout=0.0,
                 whether_PositionalEncoding=True,
                 whether_PositionalWiseFeedForward=True
                ):
        super(Encoder, self).__init__()
        self.whether_PositionalEncoding = whether_PositionalEncoding
        self.whether_PositionalWiseFeedForward = whether_PositionalWiseFeedForward

        self.encoder_layers = nn.ModuleList([EncoderLayer(model_dim, num_heads, ffn_dim, dropout, self.whether_PositionalWiseFeedForward) for _ in range(num_layers)])

        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, attention_mask):
        inputs_len = torch.cuda.LongTensor([sum(row).item() for row in attention_mask])
        output = inputs # batch_size * seq_len * hidden_state
        if self.whether_PositionalEncoding:
            output += self.pos_embedding(inputs_len, inputs.size(1)) # batch_size * seq_len * hidden_state

        self_attention_mask = padding_mask(attention_mask, attention_mask) # batch_size * seq_len_q * seq_len_k

        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)

        return output, attentions

class crossEncoderLayer(nn.Module):
    """Encoder的一层。"""

    def __init__(self, model_dim=768, num_heads=8, ffn_dim=3072, dropout=0.0, whether_PositionalWiseFeedForward=True):
        super(crossEncoderLayer, self).__init__()
        self.whether_PositionalWiseFeedForward = whether_PositionalWiseFeedForward

        self.cross_attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, cat_embeded, sentences_embeded, attn_mask=None):

        # self attention
        output, attention = self.cross_attention(sentences_embeded, sentences_embeded, cat_embeded, attn_mask)

        # feed forward network
        if self.whether_PositionalWiseFeedForward:
            output = self.feed_forward(output)

        return output, attention

class crossEncoder(nn.Module):
    """多层EncoderLayer组成Encoder。"""

    def __init__(self,
                 max_seq_len,
                 num_layers=6,
                 model_dim=768,
                 num_heads=8,
                 ffn_dim=3072,
                 dropout=0.0,
                 whether_PositionalEncoding=True,
                 whether_PositionalWiseFeedForward=True
                ):
        super(crossEncoder, self).__init__()
        self.whether_PositionalEncoding = whether_PositionalEncoding
        self.whether_PositionalWiseFeedForward = whether_PositionalWiseFeedForward

        # self.encoder_layers = nn.ModuleList([crossEncoderLayer(model_dim, num_heads, ffn_dim, dropout, self.whether_PositionalWiseFeedForward) for _ in range(num_layers)])
        self.sentence_encoder = EncoderLayer(model_dim, num_heads, ffn_dim, dropout, self.whether_PositionalWiseFeedForward)
        self.prompt_encoder = EncoderLayer(model_dim, num_heads, ffn_dim, dropout, self.whether_PositionalWiseFeedForward)
        self.cross_encoder = crossEncoderLayer(model_dim, num_heads, ffn_dim, dropout, self.whether_PositionalWiseFeedForward)

        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, cat_embeded, sentences_embeded, attention_mask):
        # 给句子端做position embedding
        inputs_len_sentences_embeded = torch.cuda.LongTensor([sum(row).item() for row in attention_mask])
        sentences_embeded = sentences_embeded # batch_size * seq_len * hidden_state
        if self.whether_PositionalEncoding:
            sentences_embeded += self.pos_embedding(inputs_len_sentences_embeded, sentences_embeded.size(1)) # batch_size * seq_len * hidden_state

        # 给prompt端做position embedding
        # inputs_len_cat_embeded = torch.cuda.LongTensor([cat_embeded.shape(1) for row in cat_embeded])
        # cat_embeded = cat_embeded # batch_size * seq_len * hidden_state
        # if self.whether_PositionalEncoding:
        #     sentences_embeded += self.pos_embedding(inputs_len_cat_embeded, cat_embeded.size(1)) # batch_size * seq_len * hidden_state

        corss_attention_mask = padding_mask(attention_mask, torch.ones_like(cat_embeded)) # batch_size * seq_len_q * seq_len_k
        self_attention_mask = padding_mask(attention_mask, attention_mask) # batch_size * seq_len_q * seq_len_k

        output_sentence, _ = self.sentence_encoder(sentences_embeded, self_attention_mask)
        output_prompt, _ = self.prompt_encoder(cat_embeded)
        output, attention = self.cross_encoder(output_prompt, output_sentence, corss_attention_mask)

        return output, attention


"""
参考文档
1. https://github.com/jadore801120/attention-is-all-you-need-pytorch/tree/master/transformer
2. https://github.com/JayParks/transformer
3. http://nlp.seas.harvard.edu/2018/04/03/attention.html
4. https://luozhouyang.github.io/transformer/
5. https://zhuanlan.zhihu.com/p/179959751
"""