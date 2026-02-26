import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel, AutoConfig

class PCLClassifier(PreTrainedModel):
    def __init__(self, model_name):
        config = AutoConfig.from_pretrained(model_name)
        super().__init__(config)
        self.deberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.deberta.config.hidden_size

        self.seq_classifier   = nn.Linear(hidden_size, 1)
        self.token_classifier = nn.Linear(hidden_size, 1)
        self.fusion           = nn.Linear(2, 1)  # combines cls + token evidence

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state.float()

        # CLS-based sequence logit
        cls_logit = self.seq_classifier(hidden[:, 0, :])  # (batch, 1)

        # token logits - mask padding before pooling
        token_logits = self.token_classifier(hidden)  # (batch, seq_len, 1)
        token_logits = token_logits.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)

        # summarise token evidence into a single score
        span_logit, _ = torch.max(token_logits, dim=1)  # (batch, 1)

        # fuse both signals into the final sequence prediction
        seq_logits = self.fusion(torch.cat([cls_logit, span_logit], dim=1))  # (batch, 1)

        return seq_logits, token_logits.squeeze(-1)