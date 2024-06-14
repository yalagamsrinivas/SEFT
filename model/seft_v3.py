import torch
from torch.optim.lr_scheduler import StepLR
torch.set_default_dtype(torch.float16)
from model.BaseModelWrapper import BaseWrapper

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerTextEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, hidden_dim, device):
        super(TransformerTextEncoder, self).__init__()
        self.device = device
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(device)

    def forward(self, embedded_text):
        batch_size, num_days, num_messages, num_words, embedding_dim = embedded_text.shape
        embedded_text = embedded_text.view(-1, num_messages * num_words, embedding_dim)
        encoded_text = self.transformer_encoder(embedded_text.permute(1, 0, 2))
        encoded_text = encoded_text.permute(1, 0, 2).view(batch_size, num_days, -1, embedding_dim)
        encoded_text = encoded_text.mean(dim=2)
        return encoded_text


class StockPriceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(StockPriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True).to(device)

    def forward(self, technical_features):
        _, (hidden_states, _) = self.lstm(technical_features.view(-1, 1, technical_features.size(-1)))
        hidden_states = hidden_states[-1].view(technical_features.size(0), technical_features.size(1), -1)
        return hidden_states

class Expert(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(Expert, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        ).to(device)

    def forward(self, x):
        return self.layer(x)


class MixtureOfExperts(nn.Module):
    def __init__(self, text_feature_dim, technical_feature_dim, output_dim, num_experts, device):
        super(MixtureOfExperts, self).__init__()
        self.device = device
        self.num_experts = num_experts
        input_dim = text_feature_dim + technical_feature_dim
        self.experts = nn.ModuleList([Expert(input_dim, output_dim, device) for _ in range(num_experts)])
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        ).to(device)

    def forward(self, text_features, technical_features):
        x = torch.cat((text_features, technical_features), dim=-1)
        weights = self.gating_network(x)  # Determine the weight of each expert's output

        expert_outputs = torch.stack([expert(x) for expert in self.experts],
                                     dim=1)  # [batch_size, num_experts, output_dim]
        # Weighted sum of expert outputs
        output = torch.sum(weights.unsqueeze(-1) * expert_outputs, dim=1)
        return output


class SimpleFusionModule(nn.Module):
    def __init__(self, text_feature_dim, technical_feature_dim, output_dim, num_experts, device):
        super(SimpleFusionModule, self).__init__()
        self.moe = MixtureOfExperts(text_feature_dim, technical_feature_dim, output_dim, num_experts, device)

    def forward(self, text_features, technical_features):
        return self.moe(text_features, technical_features)


class SEFT(nn.Module):
    def __init__(self, embedding, embedding_dim, num_heads, num_layers, hidden_dim, technical_feature_dim, output_dim,
                 device,num_experts=5):
        super(SEFT, self).__init__()
        self.device = device
        self.embedding = embedding.to(device)
        self.text_encoder = TransformerTextEncoder(embedding_dim, num_heads, num_layers, hidden_dim, device)
        self.stock_lstm = StockPriceLSTM(technical_feature_dim, hidden_dim, device)
        self.fusion_module = SimpleFusionModule(embedding_dim, hidden_dim, output_dim, num_experts, device)

    def forward(self, inputs):
        text_data,technical_data, _ = inputs

        text_data = text_data.to(self.device)
        technical_data = technical_data.to(self.device)

        text_data = torch.squeeze(text_data, dim=1).long().to(self.device)
        technical_data = torch.squeeze(technical_data, dim=1).to(self.device)

        embedded_text = self.embedding(text_data)
        text_features = self.text_encoder(embedded_text)
        technical_features = self.stock_lstm(technical_data)

        predictions = self.fusion_module(text_features, technical_features)
        return predictions

class SEFTNEWWrapper(BaseWrapper):
    def __init__(self, name, network, optimizer, votes, cfg):
        super(SEFTNEWWrapper, self).__init__(name=name, network=network, optimizer=optimizer, votes=votes, cfg=cfg)

        self.optimizer = 'Adam'
        if self.optimizer == 'Adam':
            # optim = torch.optim.Adam(params=self.network.parameters(), lr=self.cfg['train']['lr'], eps=self.cfg['train']['epsilon'])
            self.optim = torch.optim.Adam(params=self.network.parameters(),
                                          lr=self.cfg['train']['lr'])
        elif self.optimizer == 'SGD':
            self.cfg['train']['momentum'] = 0
            self.optim = torch.optim.SGD(params=self.network.parameters(),
                                         lr=self.cfg['train']['lr'], momentum=self.cfg['train']['momentum'])
        else:
            raise Exception('the optimizer is not supported')

        self.scheduler = StepLR(self.optim,
                                step_size=self.cfg['train']['schedule_step'],
                                gamma=self.cfg['train']['schedule_gamma'])