import torch
import torch.nn as nn
from typing import Optional
from torch.nn import functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class ContextAwareAttention(nn.Module):

    def __init__(self,
                 dim_model : int,
                 dim_context : int,
                 dropout_rate : Optional[float] = 0.0 ):

        super(ContextAwareAttention, self).__init__()

        self.dim_model = dim_model
        self.dim_context = dim_context
        self.dropout_rate = dropout_rate
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.dim_model,
                                                     num_heads = 1,
                                                     dropout = self.dropout_rate,
                                                     bias = True,
                                                     add_zero_attn=False,
                                                     batch_first=True,
                                                     device=DEVICE
        )

        self.u_k = nn.Linear(self.dim_context, self.dim_model, bias = False)
        self.w1_k = nn.Linear(self.dim_model, 1, bias=False)
        self.w2_k = nn.Linear(self.dim_model, 1, bias=False)

        self.u_v = nn.Linear(self.dim_context, self.dim_model, bias=False)
        self.w1_v = nn.Linear(self.dim_model, 1, bias = False)
        self.w2_v = nn.Linear(self.dim_model, 1, bias = False)

    def forward(self, q, k, v, context):
        # Reshape k to match the expected shape
        k = k.view(k.size(0), k.size(1), -1)

        # print("Context shape : ", context.shape)
        # print("Dim context : ", self.dim_context, " : Dim model : ", self.dim_model)
        key_context = self.u_k(context)
        # print("Context shape below key context : ", key_context.shape)
        value_context = self.u_v(context)

        lambda_k = F.sigmoid(self.w1_k(k) + self.w2_k(key_context))
        lambda_v = F.sigmoid(self.w1_v(v) + self.w2_v(value_context))

        k_cap = (1-lambda_k) * k + (lambda_k) * key_context
        v_cap = (1-lambda_v) * v + (lambda_v) * value_context

        attention_output, _ = self.attention_layer(query = q,
                                                   key = k_cap,
                                                   value = v_cap)

        return attention_output

