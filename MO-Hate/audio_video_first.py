import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from context_aware_attention import ContextAwareAttention
from transformers import BartModel

SOURCE_MAX_LEN = 768
ACOUSTIC_DIM = 768
ACOUSTIC_MAX_LEN = 1000
VISUAL_DIM = 768
VISUAL_MAX_LEN = 100

bart_model = BartModel.from_pretrained('facebook/bart-base')

class MAF_main(nn.Module):
    def __init__(self,
                dim_model,
                dropout_rate):
        super(MAF_main, self).__init__()
        self.dropout_rate = dropout_rate

        self.context_attention = ContextAwareAttention(dim_model=dim_model,
                                                        dim_context=dim_model,
                                                        dropout_rate=dropout_rate)

        self.gate = nn.Linear(2*dim_model, dim_model)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.final_layer_norm = nn.LayerNorm(dim_model)

    def forward(self,
                main_input,
                context_input):

        mixed_out = self.context_attention(q=main_input,
                                            k=main_input,
                                            v=main_input,
                                            context=context_input)

        weight_v = F.sigmoid(self.gate(torch.cat([mixed_out, main_input], dim=-1)))

        output = self.final_layer_norm(weight_v * mixed_out  + main_input)

        return output


class MultimodalClassification(nn.Module):
    def __init__(
        self,
        input_dim: int,
        inned_dim: int,
        num_classes: int,
        pooler_dropout: float
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inned_dim)
        self.dropout = nn.Dropout(p = pooler_dropout)
        self.out_proj = nn.Linear(inned_dim, num_classes)

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states
    

class MultimodalAudio(torch.nn.Module):
  def __init__(self):
    super(MultimodalAudio, self).__init__()
    self.bart_model = bart_model

    self.acoustic_context_transform = nn.Linear(ACOUSTIC_MAX_LEN, SOURCE_MAX_LEN, bias = False)
    self.acoustic_dimension_transform = nn.Linear(ACOUSTIC_DIM, 768, bias = False)

    self.visual_context_transform = nn.Linear(VISUAL_MAX_LEN, SOURCE_MAX_LEN, bias = False)
    self.visual_dimension_transform = nn.Linear(VISUAL_DIM, 768, bias = False)

    self.maf_one = MAF_main(dim_model=768, dropout_rate=0.2)
    self.maf_two = MAF_main(dim_model=768, dropout_rate=0.2)

    self.classification_head = MultimodalClassification(
            768,
            768,
            2,
            0.0
        )

  def forward(self, input_ids, attention_mask, acoustic_input, visual_input, labels):

    if acoustic_input.dim() == 2:
        acoustic_input = acoustic_input.unsqueeze(1)
    # acoustic_input = acoustic_input.permute(0,2,1)
    acoustic_input = acoustic_input.repeat(1, 1, ACOUSTIC_MAX_LEN // 768 + 1)[:, :, :ACOUSTIC_MAX_LEN]
    acoustic_input = acoustic_input.reshape(-1, ACOUSTIC_MAX_LEN)
    acoustic_input = self.acoustic_context_transform(acoustic_input.float())
    # acoustic_input = acoustic_input.permute(0,2,1)
    acoustic_input = acoustic_input.view(acoustic_input.size(0), -1, SOURCE_MAX_LEN)
    acoustic_input = self.acoustic_dimension_transform(acoustic_input)

    if visual_input.dim() == 2:
        visual_input = visual_input.unsqueeze(1)
    visual_input = visual_input.repeat(1, 1, VISUAL_MAX_LEN // visual_input.shape[-1] + 1)[:, :, :VISUAL_MAX_LEN]
    visual_input = visual_input.reshape(-1, VISUAL_MAX_LEN)
    visual_input = self.visual_context_transform(visual_input.float())
    visual_input = visual_input.view(visual_input.size(0), -1, SOURCE_MAX_LEN)
    visual_input = self.visual_dimension_transform(visual_input)

    bart_output = self.bart_model(input_ids, attention_mask)['last_hidden_state']
    output_one = self.maf_one(main_input = bart_output, context_input = torch.zeros_like(visual_input))
    output_two = self.maf_two(main_input = output_one, context_input = acoustic_input)
    
    final_out = output_one[:, -1, :]
    # final_out = output_two[:, -1, :]
    # final_out = acoustic_input[:, -1, :]

    final_out = self.classification_head(final_out)

    loss_fct = CrossEntropyLoss()
    loss = loss_fct(final_out.view(-1, 2), labels.view(-1))

    temp_dict = {}

    temp_dict['logits'] = final_out
    temp_dict['loss'] = loss

    return temp_dict

