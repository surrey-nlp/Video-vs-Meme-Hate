import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartConfig, BartPretrainedModel
from transformers.modeling_outputs import BaseModelOutput
# from transformers.modeling_utils import _expand_mask
from transformers.models.bart.modeling_bart import BartEncoderLayer, BartLearnedPositionalEmbedding

from typing import Optional
import random, math
from acoustic_fusion import MAF_acoustic
from visual_fusion import MAF_visual

def expand_attention_mask(mask: torch.Tensor, dtype: torch.dtype):
    expanded_mask = mask[:, None, None, :]
    # print("Expanded mask shape : ", expanded_mask.shape)
    expanded_mask = torch.logical_not(expanded_mask) * torch.finfo(dtype).min
    return expanded_mask

def expand_attention_mask_memes(mask: torch.Tensor, dtype: torch.dtype):
    mask = mask.squeeze(1) # Remove the second dimension
    expanded_mask = mask[:, None, None, :]
    expanded_mask = torch.logical_not(expanded_mask) * torch.finfo(dtype).min
    return expanded_mask

class MultiModalBartEncoder(BartPretrainedModel):

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_position = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim
        )

        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
         # Create a list to store the layer class names
        self.layer_class_names = []

        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.init_weights()
        self.gradient_checkpointing = False

        # self.fusion_at_layer = [4]
        # self.fusion_at_layer = [3, 4]
        # self.fusion_at_layer4 = [4]
        self.fusion_at_layer4 = [4]
        self.fusion_at_layer5 = [5]

        # self.fusion_of_context = [3]

        self.MAF_layer4 = MAF_acoustic(dim_model=embed_dim, dropout_rate=0.2)

        self.MAF_layer5 = MAF_visual(dim_model=embed_dim, dropout_rate=0.2)

        # self.context_encoder = ContextEncoder(config)

        # self.classification = nn.Linear(embed_dim, 2)


    def forward(self,
            input_ids = None,
            attention_mask = None,
            # context_input_ids = None,
            # context_attention_mask = None,
            acoustic_input = None,
            visual_input = None,
            head_mask = None,
            inputs_embeds = None,
            output_attentions = None,
            output_hidden_states  = None,
            return_dict = None):

            # print("Input ids shape : ", input_ids.shape)
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )

            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You can't specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])

            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                raise ValueError("You have to specify either input_ids or input_embeds")


            if inputs_embeds is None:
                # print("Input ids shape : ", input_ids.shape)
                inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

            input_shape = torch.tensor(input_shape)
            embed_pos = self.embed_positions(input_ids)
            # embed_pos = self.embed_positions(input_shape)


            hidden_states = inputs_embeds + embed_pos
            hidden_states = self.layernorm_embedding(hidden_states)
            hidden_states = F.dropout(hidden_states, p = self.dropout, training=self.training)

            if attention_mask is not None:
                attention_mask = expand_attention_mask(attention_mask, inputs_embeds.dtype)
                # attention_mask = expand_attention_mask_memes(attention_mask, inputs_embeds.dtype)     # uncomment this line for memes dataset

            # print("attention mask shape 4 : ", attention_mask.shape)
            encoder_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None

            if head_mask is not None:
                assert head_mask.size()[0] == (
                    len(self.layers)
                ), f"The head mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

            for idx, encoder_layer in enumerate(self.layers):
                if idx in self.fusion_at_layer4:
                    hidden_states = self.MAF_layer4(text_input = hidden_states,
                                                   acoustic_context = acoustic_input)
                    
                if idx in self.fusion_at_layer5:
                    hidden_states = self.MAF_layer5(text_input = hidden_states,
                                                   visual_context = visual_input)

                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states,)

                dropout_probability = random.uniform(0,1)

                if self.training and (dropout_probability < self.layerdrop):
                    layer_outputs = (None, None)

                else:
                    if self.gradient_checkpointing and self.training:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs, output_attentions)

                            return custom_forward

                        layer_outputs = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(encoder_layer),
                            hidden_states,
                            attention_mask,
                            (head_mask[idx] if head_mask is not None else None),
                        )

                    else:
                        # print("Checking Attention mask shape : ", attention_mask.shape)
                        layer_outputs = encoder_layer(
                            hidden_states,
                            attention_mask,
                            layer_head_mask = (head_mask[idx] if head_mask is not None else None),
                            output_attentions = output_attentions
                        )

                    hidden_states = layer_outputs[0]

                if output_attentions:
                    all_attentions  = all_attentions + (layer_outputs[1],)

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            if not return_dict:
                return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

            return BaseModelOutput(
                last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
            )
