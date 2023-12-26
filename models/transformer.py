import torch
import torch.nn as nn

from .base import BaseNet


class DepressionDetector(BaseNet):

    def __init__(self, d=256, l=6, t_downsample=4):
        super().__init__()
        self.v_downsample = nn.Sequential(
            nn.Conv1d(136, d, kernel_size=16, stride=t_downsample, padding=8),
            nn.BatchNorm1d(d),
        )
        self.v_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d, nhead=4, dim_feedforward=d, batch_first=True
            ),
            num_layers=l, 
        )
        self.a_downsample = nn.Sequential(
            nn.Conv1d(25, d, kernel_size=16, stride=t_downsample, padding=8),
            nn.BatchNorm1d(d),
        )
        self.a_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d, nhead=4, dim_feedforward=d, batch_first=True
            ),
            num_layers=l, 
        )
        self.qa_transform = nn.Linear(d, d)
        self.ka_transform = nn.Linear(d, d)
        self.va_transform = nn.Linear(d, d)
        self.qv_transform = nn.Linear(d, d)
        self.kv_transform = nn.Linear(d, d)
        self.vv_transform = nn.Linear(d, d)
        self.cross_av = nn.MultiheadAttention(
            embed_dim=d, num_heads=4, batch_first=True
        )
        self.cross_va = nn.MultiheadAttention(
            embed_dim=d, num_heads=4, batch_first=True
        )
        self.av_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=2*d, nhead=4, dim_feedforward=2*d, batch_first=True
            ),
            num_layers=l, 
        )
        self.z_dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2*d, 1)

    def feature_extractor(self, x):
        xa = x[:, :, 136:]
        xv = x[:, :, :136]
        xa = self.a_downsample(xa.transpose(1, 2)).transpose(1, 2)
        xv = self.v_downsample(xv.transpose(1, 2)).transpose(1, 2)
        ua = self.a_encoder(xa)
        uv = self.v_encoder(xv)

        qa = self.qa_transform(ua)
        ka = self.ka_transform(ua)
        va = self.va_transform(ua)
        qv = self.qv_transform(uv)
        kv = self.kv_transform(uv)
        vv = self.vv_transform(uv)

        uua = self.cross_av(qa, kv, vv)[0]
        uuv = self.cross_va(qv, ka, va)[0]
        uav = torch.cat((uua, uuv), dim=2)

        z = self.av_encoder(uav)
        z = torch.mean(z, dim=1)
        return self.z_dropout(z)

    def classifier(self, x):
        return self.fc(x)