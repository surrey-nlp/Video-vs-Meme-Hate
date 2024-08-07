import torch
import torch.nn as nn
from transformers import BartPretrainedModel, BartConfig
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput
from typing import Optional, Tuple, Union, List
from torch.nn import CrossEntropyLoss

from multimodal_bart_model import MultimodalBartModel

class MultimodalBartClassification(nn.Module):
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




class MultimodalBartForSequenceClassification(BartPretrainedModel):
    _keys_to_ignore_on_load_missing = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = MultimodalBartModel(config)
        self.classification_head = MultimodalBartClassification(
            config.d_model,
            config.d_model,
            2,
            config.classifier_dropout
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.tensor] = None,
        # context_input_ids : torch.LongTensor = None,
        # context_attention_mask : Optional[torch.tensor] = None,
        acoustic_input = None,
        visual_input = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, Seq2SeqSequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        # print("attention mask shape 1 : ", attention_mask.shape )
        outputs = self.model(
            input_ids,
            attention_mask = attention_mask,
            # context_input_ids = context_input_ids,
            # context_attention_mask = context_attention_mask,
            acoustic_input = acoustic_input,
            visual_input = visual_input,
            decoder_input_ids = decoder_input_ids,
            decoder_attention_mask = decoder_attention_mask,
            head_mask = head_mask,
            decoder_head_mask = decoder_head_mask,
            cross_attn_head_mask = cross_attn_head_mask,
            encoder_outputs = encoder_outputs,
            inputs_embeds = inputs_embeds,
            decoder_inputs_embeds = decoder_inputs_embeds,
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict

        )

        hidden_states = outputs[0]

        eos_mask = input_ids.eq(self.config.eos_token_id).to(hidden_states.device)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have same number of <eos> tokens")

        # sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]
        sentence_representation = hidden_states[eos_mask.any(dim=-1)].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]

        logits = self.classification_head(sentence_representation)

        loss = None

        loss_fct = CrossEntropyLoss()
        # print("logits shape : ", logits.shape)
        if labels is not None:
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        else:
            loss = None
        # print("Loss : ", loss)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output


        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions = outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions
        )