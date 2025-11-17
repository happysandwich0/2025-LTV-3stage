import torch
import torch.nn as nn
from transformers import LongformerConfig, LongformerModel

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        return self.fc(last_hidden_state).squeeze(-1)

class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_size)
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)

class LongformerRegressor(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, nlayers, p=0.1, base_rate=0.0314, max_len=4096, tabular_input_dim=0, regression_model_type='mlp'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        config = LongformerConfig(
            vocab_size=vocab_size,
            hidden_size=d_model,
            num_hidden_layers=nlayers,
            num_attention_heads=nhead,
            intermediate_size=d_model * 4,
            attention_probs_dropout_prob=p,
            hidden_dropout_prob=p,
            max_position_embeddings=max_len + 2,
            padding_idx=0,
        )
        self.encoder = LongformerModel(config)
        
        self.regression_model_type = regression_model_type
        
        total_input_dim = d_model + tabular_input_dim

        if regression_model_type == 'mlp':
            self.regressor = MLPRegressor(total_input_dim, d_model * 2)
        elif regression_model_type == 'lstm':
            self.regressor = LSTMRegressor(total_input_dim, d_model * 2)
        else:
            raise ValueError(f"Unsupported regression_model_type: {regression_model_type}")

    def forward(self, input_ids, attention_mask=None, global_attention_mask=None, tabular_features=None, position_ids=None):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            position_ids=position_ids
        )
        pooled_output = out.last_hidden_state[:, 0]
        
        if tabular_features is not None and tabular_features.size(1) > 0:
            combined_features = torch.cat((pooled_output, tabular_features), dim=1)
        else:
            combined_features = pooled_output
        
        if self.regression_model_type == 'mlp':
            logits = self.regressor(combined_features)
        elif self.regression_model_type == 'lstm':
            logits = self.regressor(combined_features.unsqueeze(1))
            
        return logits.squeeze(-1)
