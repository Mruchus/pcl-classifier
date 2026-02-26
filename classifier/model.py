import torch.nn as nn
from transformers import AutoModel, PreTrainedModel

class PCLClassifier(PreTrainedModel):
    def __init__(self, model_name):
        # load the config from the pre‑trained model
        config = AutoModel.from_pretrained(model_name).config
        super().__init__(config)
        self.deberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.deberta.config.hidden_size
        self.seq_classifier = nn.Linear(hidden_size, 1) # paragraph logit
        self.token_classifier = nn.Linear(hidden_size, 1) # token logits

    def forward(self, input_ids, attention_mask, token_labels=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state.float()
        seq_logits = self.seq_classifier(outputs.last_hidden_state[:, 0, :]) # [CLS]
        token_logits = self.token_classifier(outputs.last_hidden_state).squeeze(-1) # (batch, seq_len)
        token_logits = token_logits * attention_mask # mask padding
        return seq_logits, token_logits