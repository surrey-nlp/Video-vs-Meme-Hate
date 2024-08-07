import torch
import torch.nn as nn
from transformers import BartConfig, BartPretrainedModel
from transformers.modeling_outputs import Seq2SeqModelOutput, BaseModelOutput
from transformers.models.bart.modeling_bart import BartDecoder, shift_tokens_right
from multimodal_bart_encoder import MultiModalBartEncoder

class MultimodalBartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = MultiModalBartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder


    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        # context_input_ids = None,
        # context_attention_mask = None,
        acoustic_input = None,
        visual_input = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        head_mask = None,
        decoder_head_mask = None,
        cross_attn_head_mask = None,
        encoder_outputs = None,
        past_key_values = None,
        inputs_embeds = None,
        decoder_inputs_embeds = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None
    ):

        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id

            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # if attention_mask is not None:
        #     attention_mask = attention_mask.unsqueeze(1).expand(32, 16, -1, -1).to(dtype=next(self.parameters()).dtype)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # print("attention mask shape 2 : ", attention_mask.shape)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids = input_ids,
                attention_mask = attention_mask,
                # context_input_ids = context_input_ids,
                # context_attention_mask = context_attention_mask,
                acoustic_input = acoustic_input,
                visual_input = visual_input,
                head_mask = head_mask,
                inputs_embeds = inputs_embeds,
                output_attentions = output_attentions,
                output_hidden_states = output_hidden_states,
                return_dict = return_dict
            )

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions = encoder_outputs[2] if len(encoder_outputs) > 2 else None
            )

        decoder_outputs = self.decoder(
            input_ids = decoder_input_ids,
            attention_mask = decoder_attention_mask,
            encoder_hidden_states = encoder_outputs[0],
            encoder_attention_mask = attention_mask,
            head_mask = decoder_head_mask,
            cross_attn_head_mask = cross_attn_head_mask,
            past_key_values = past_key_values,
            inputs_embeds = decoder_inputs_embeds,
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions

        )