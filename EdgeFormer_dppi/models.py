import torch
import torch.nn as nn

class EdgeFormer(nn.Module):
    def __init__(self):
        super(EdgeFormer, self).__init__()

        self.fc0 = nn.Linear(1, 128)
        self.bn0 = nn.BatchNorm1d(128)

        self.ln = nn.LayerNorm(128)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8).cuda()
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4).cuda()

        self.fc1 = nn.Linear(5120, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x1, x2):
        x1 = x1.transpose(0, 1).unsqueeze(2)
        x1 = self.relu(self.fc0(x1))
        x1 = self.transformer_encoder(x1)
        x1 = x1.transpose(0, 1)
        x1 = self.ln(x1)
        x1 = x1.reshape(x1.shape[0], -1)

        x2 = x2.transpose(0, 1).unsqueeze(2)
        x2 = self.relu(self.fc0(x2))
        x2 = self.transformer_encoder(x2)
        x2 = x2.transpose(0, 1)
        x2 = self.ln(x2)
        x2 = x2.reshape(x2.shape[0], -1)

        x = torch.cat([x1, x2], dim=1)

        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.bn3(self.fc3(x))
        return x
