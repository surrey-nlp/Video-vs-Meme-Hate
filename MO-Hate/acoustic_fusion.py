import torch
import torch.nn as nn
import torch.nn.functional as F

from context_aware_attention import ContextAwareAttention

SOURCE_MAX_LEN = 768
ACOUSTIC_DIM = 768
ACOUSTIC_MAX_LEN = 1000

class MAF_acoustic(nn.Module):
    def __init__(self,
                dim_model,
                dropout_rate):
        super(MAF_acoustic, self).__init__()
        self.dropout_rate = dropout_rate

        self.acoustic_context_transform = nn.Linear(ACOUSTIC_MAX_LEN, SOURCE_MAX_LEN, bias = False)

        self.acoustic_context_attention = ContextAwareAttention(dim_model=dim_model,
                                                                dim_context=ACOUSTIC_DIM,
                                                                dropout_rate=dropout_rate)

        self.acoustic_gate = nn.Linear(2*dim_model, dim_model)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.final_layer_norm = nn.LayerNorm(dim_model)

    def forward(self,
                text_input,
                acoustic_context):

        # Ensure acoustic_context has three dimensions
        if acoustic_context.dim() == 2:
            acoustic_context = acoustic_context.unsqueeze(1)

        acoustic_context = acoustic_context.repeat(1, 1, ACOUSTIC_MAX_LEN // 768 + 1)[:, :, :ACOUSTIC_MAX_LEN]

        acoustic_context = acoustic_context.reshape(-1, ACOUSTIC_MAX_LEN)
        acoustic_context = self.acoustic_context_transform(acoustic_context.float())
        acoustic_context = acoustic_context.view(text_input.size(0), -1, SOURCE_MAX_LEN)

        audio_out = self.acoustic_context_attention(q=text_input,
                                                    k=text_input,
                                                    v=text_input,
                                                    context=acoustic_context)

        weight_a = F.sigmoid(self.acoustic_gate(torch.cat([text_input, torch.zeros_like(audio_out)], dim=-1)))

        # output = self.final_layer_norm(text_input + weight_a * audio_out + weight_v * video_out)
        output = self.final_layer_norm(text_input + weight_a * audio_out)

        return output