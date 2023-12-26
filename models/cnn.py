import torch
import torch.nn as nn

from .base import BaseNet


class ResBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size=8,
                padding=7, dilation=2
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(
                out_channels, out_channels, kernel_size=4,
                padding=6, dilation=4
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Conv1d(
                in_channels, out_channels, kernel_size=1, padding=0
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(
                out_channels, out_channels, kernel_size=1, padding=0
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.branch3 = (
            nn.Identity() if in_channels == out_channels else nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, padding=0, 
                ),
            )
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        # x.shape = [B, D, T]
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        return self.activation(x1 + x2 + x3)


class TAMF(nn.Module):

    def __init__(self, d=256):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=d, hidden_size=d, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=d, hidden_size=d, batch_first=True)
        self.proj_v = nn.Linear(2, 2)
        self.proj_h1 = nn.LSTM(
            input_size=2, hidden_size=2, batch_first=True, bidirectional=True
        )
        self.proj_h2 = nn.Linear(4, 2)


    def forward(self, x1, x2):
        # shape = [B, T, D]
        xx1 = self.lstm1(x1)[0]
        xx2 = self.lstm2(x2)[0] # [B, T, D]

        xxx1 = torch.mean(xx1, dim=2, keepdim=True)
        xxx2 = torch.mean(xx2, dim=2, keepdim=True) # [B, T, 1]
        xxx = torch.cat([xxx1, xxx2], dim=2) # [B, T, 2]

        xxxv = self.proj_v(xxx) # [B, T, 2]
        xxxm = self.proj_h2(self.proj_h1(xxxv)[0]) # [B, T, 2]

        aw1 = xxxm[:, :, 0:1]
        aw2 = xxxm[:, :, 1:2] # [B, T, 1]
        return aw1 * xx1 + aw2 * xx2


class TAMFN(BaseNet):

    def __init__(self, d=256, l=6, t_downsample=4):
        super().__init__()
        self.v_downsample = nn.Sequential(
            nn.Conv1d(136, d, kernel_size=16, stride=t_downsample, padding=8),
            nn.BatchNorm1d(d),
        )
        self.a_downsample = nn.Sequential(
            nn.Conv1d(25, d, kernel_size=16, stride=t_downsample, padding=8),
            nn.BatchNorm1d(d),
        )
        self.gtcn_a = nn.ModuleList()
        self.gtcn_v = nn.ModuleList()
        self.ife = nn.ModuleList()
        self.l = l
        for _ in range(l):
            self.gtcn_a.append(ResBlock(d, d))
            self.gtcn_v.append(ResBlock(d, d))
            self.ife.append(ResBlock(d, d))
        self.tamf_av = TAMF(d)
        self.tamf_af = TAMF(d)
        self.tamf_fv = TAMF(d)
        self.feature_dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(d, 1)

    def feature_extractor(self, x):
        xa = x[:, :, 136:]
        xv = x[:, :, :136]
        xa = self.a_downsample(xa.transpose(1, 2))
        xv = self.v_downsample(xv.transpose(1, 2)) # [B, D, T]
        xf = torch.zeros_like(xa)

        for i in range(self.l):
            xa = self.gtcn_a[i](xa)
            xv = self.gtcn_v[i](xv)
            xf = self.ife[i](xf + xa + xv)
        xa = xa.transpose(1, 2)
        xv = xv.transpose(1, 2)
        xf = xf.transpose(1, 2)

        xav = self.tamf_av(xa, xv)
        xaf = self.tamf_af(xa, xf)
        xfv = self.tamf_fv(xf, xv) # [B, T, D]

        xav = torch.mean(xav, dim=1)
        xaf = torch.mean(xaf, dim=1)
        xfv = torch.mean(xfv, dim=1)

        return self.feature_dropout(xav + xaf + xfv)

    def classifier(self, x):
        return self.fc(x)




