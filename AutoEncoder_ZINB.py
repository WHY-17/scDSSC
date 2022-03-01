import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from layers import ZINBLoss, MeanAct, DispAct



class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2,
                 n_dec_1, n_dec_2,
                 n_input, n_z,
                 denoise, sigma):
        super(AE, self).__init__()
        self.n_enc_1 = n_enc_1
        self.n_enc_2 = n_enc_2
        self.n_dec_1 = n_dec_1
        self.n_dec_2 = n_dec_2
        self.n_input = n_input
        self.n_z = n_z
        self.denoise = denoise
        self.sigma = sigma

        self.enc_1 = Linear(self.n_input, self.n_enc_1, bias=True)
        self.enc_2 = Linear(self.n_enc_1, self.n_enc_2, bias=True)
        # self.enc_3 = Linear(n_enc_2, n_enc_3, bias=True)
        self._enc_mu = Linear(self.n_enc_2, self.n_z, bias=True)
        # self.z_layer = Linear(self.n_enc_2, self.n_z, bias=True)

        self.dec_1 = Linear(self.n_z, self.n_dec_1, bias=True)
        self.dec_2 = Linear(self.n_dec_1, self.n_dec_2, bias=True)
        # self.dec_3 = Linear(n_dec_2, n_dec_3)
        self._dec_mean = nn.Sequential(nn.Linear(self.n_dec_2, self.n_input), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(self.n_dec_2, self.n_input), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(self.n_dec_2, self.n_input), nn.Sigmoid())
        # self.x_bar_layer = Linear(self.n_dec_2, self.n_input, bias=True)

    def Encoder(self, x):
        if self.denoise == False:
            enc_h1 = F.relu(self.enc_1(x))
            enc_h2 = F.relu(self.enc_2(enc_h1))
            # enc_h3 = F.relu(self.enc_3(enc_h2))
            z = self._enc_mu(enc_h2)
        else:
            x = x + torch.randn_like(x) * self.sigma
            enc_h1 = F.relu(self.enc_1(x))
            enc_h2 = F.relu(self.enc_2(enc_h1))
            # enc_h3 = F.relu(self.enc_3(enc_h2))
            z = self._enc_mu(enc_h2)
        return z

    def Decoder(self, z):
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        # dec_h3 = F.relu(self.dec_3(dec_h2))
        mean = self._dec_mean(dec_h2)
        disp = self._dec_disp(dec_h2)
        pi = self._dec_pi(dec_h2)
        # x_bar = self.x_bar_layer(dec_h2)
        return mean, disp, pi

    def forward(self, x):
        z = self.Encoder(x)
        mean, disp, pi = self.Decoder(z)
        return z, mean, disp, pi
