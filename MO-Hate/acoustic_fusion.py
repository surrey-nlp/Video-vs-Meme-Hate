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
        # print("Text input shape (A) : ", text_input.shape)
        # print("Acoustic context shape (A) : ", acoustic_context.shape)

        # Ensure acoustic_context has three dimensions
        if acoustic_context.dim() == 2:
            acoustic_context = acoustic_context.unsqueeze(1)

        acoustic_context = acoustic_context.repeat(1, 1, ACOUSTIC_MAX_LEN // 768 + 1)[:, :, :ACOUSTIC_MAX_LEN]
        # acoustic_context = acoustic_context.permute(0,2,1)
        # # acoustic_context = acoustic_context.expand(-1, 1000, -1)
        # print("Acoustic context shape (B) : ", acoustic_context.shape)
        # # Calculate the correct size for the last dimension
        # correct_last_dim = acoustic_context.numel() // (acoustic_context.size(0) * ACOUSTIC_MAX_LEN)
        # print("Correct last dim : ", correct_last_dim)

        # # Check if the correct_last_dim calculation is valid
        # if correct_last_dim == 0:
        #     # If not, adjust the dimensions to fit the linear layer's expected input
        #     # Here we assume the linear layer can handle the dimension (16, 768) directly
        #     acoustic_context = acoustic_context.squeeze(2)
        # else:
        #     # Reshape acoustic_context to match the expected input dimensions for the linear layer
        #     acoustic_context = acoustic_context.reshape(acoustic_context.size(0), ACOUSTIC_MAX_LEN, correct_last_dim)

        # print("Acoustic context shape (C) : ", acoustic_context.shape)
        acoustic_context = acoustic_context.reshape(-1, ACOUSTIC_MAX_LEN)
        acoustic_context = self.acoustic_context_transform(acoustic_context.float())
        acoustic_context = acoustic_context.view(text_input.size(0), -1, SOURCE_MAX_LEN)
        # acoustic_context = acoustic_context.permute(0,2,1)

        audio_out = self.acoustic_context_attention(q=text_input,
                                                    k=text_input,
                                                    v=text_input,
                                                    context=acoustic_context)
        # print("Audio out (A) : ", audio_out.shape)

        weight_a = F.sigmoid(self.acoustic_gate(torch.cat([text_input, audio_out], dim=-1)))

        # output = self.final_layer_norm(text_input + weight_a * audio_out + weight_v * video_out)
        output = self.final_layer_norm(text_input + weight_a * audio_out)

        return output