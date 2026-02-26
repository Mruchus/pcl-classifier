import torch.nn as nn
from transformers import AutoModel, PreTrainedModel, AutoConfig

class PCLClassifier(PreTrainedModel):
    def __init__(self, model_name):
        # FIX: use AutoConfig instead of loading the full model twice
        config = AutoConfig.from_pretrained(model_name)
        super().__init__(config)
        self.deberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.deberta.config.hidden_size
        self.seq_classifier   = nn.Linear(hidden_size, 1) # paragraph logit
        self.token_classifier = nn.Linear(hidden_size, 1) # token logits

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state.float()

        seq_logits   = self.seq_classifier(hidden[:, 0, :]) # (batch, 1)
        token_logits = self.token_classifier(hidden).squeeze(-1) # (batch, seq_len)
        token_logits = token_logits * attention_mask # mask padding

        return seq_logits, token_logits