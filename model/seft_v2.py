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
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(device)

    def forward(self, embedded_text):
        batch_size, num_days, num_messages, num_words, embedding_dim = embedded_text.shape
        # Flatten days, messages, and words into a single dimension for encoding
        embedded_text = embedded_text.view(-1, num_messages * num_words, embedding_dim)
        encoded_text = self.transformer_encoder(embedded_text)
        # Restore num_days dimension and take mean across messages and words
        encoded_text = encoded_text.view(batch_size, num_days, -1, embedding_dim).mean(dim=2)
        return encoded_text


class StockPriceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(StockPriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True).to(device)

    def forward(self, technical_features):
        # No need to loop through days as LSTM can process sequences
        batch_size, num_days, technical_feature_dim = technical_features.shape
        technical_features = technical_features.view(batch_size * num_days, 1, technical_feature_dim)
        _, (hidden_states, _) = self.lstm(technical_features)
        hidden_states = hidden_states[-1].view(batch_size, num_days, -1)
        return hidden_states


class TransformerFusionModule(nn.Module):
    def __init__(self, text_feature_dim, technical_feature_dim, output_dim, num_heads, hidden_dim, device):
        super(TransformerFusionModule, self).__init__()
        self.device = device
        d_model = text_feature_dim + technical_feature_dim
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads,
                                                        dim_feedforward=hidden_dim).to(device)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=1).to(device)
        self.fc = nn.Linear(d_model, output_dim).to(device)

    def forward(self, text_features, technical_features):
        batch_size, num_days, _ = text_features.shape
        # Prepare combined features and a dummy target tensor for the decoder
        combined_features = torch.cat((text_features, technical_features), dim=-1).permute(1, 0, 2)
        tgt = torch.zeros_like(combined_features)
        decoder_output = self.transformer_decoder(tgt, combined_features)
        predictions = self.fc(decoder_output.permute(1, 0, 2))
        return predictions.view(batch_size, num_days, -1)


class SEFT(nn.Module):
    def __init__(self, embedding, embedding_dim, num_heads, num_layers, hidden_dim, technical_feature_dim, output_dim,
                 device):
        super(SEFT, self).__init__()
        self.device = device
        self.embedding = embedding.to(device)
        self.embedding.requires_grad_(False)
        self.text_encoder = TransformerTextEncoder(embedding_dim, num_heads, num_layers, hidden_dim, device)
        self.stock_lstm = StockPriceLSTM(technical_feature_dim, hidden_dim, device)
        self.fusion_module = TransformerFusionModule(embedding_dim, hidden_dim, output_dim, num_heads, hidden_dim,
                                                     device)

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
        # Ensure predictions are reshaped to (batch_size, num_days, 2) if not already
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