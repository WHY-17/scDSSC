import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from torch.autograd import Variable
import h5py
import numpy as np
from layers import ZINBLoss, MeanAct, DispAct
from evaluation import eva
from AutoEncoder_ZINB import AE
from preprocess import read_dataset, normalize
import scanpy as sc
from utils import best_map, thrC, post_proC



class Deep_Sparse_Subspace_Clustering(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_dec_1, n_dec_2,
                 n_input, n_z, denoise, sigma, pre_lr, alt_lr,
                 adata, pre_epoches, alt_epoches,
                 lambda_1, lambda_2):
        super(Deep_Sparse_Subspace_Clustering, self).__init__()
        self.n_enc_1 = n_enc_1
        self.n_enc_2 = n_enc_2
        self.n_dec_1 = n_dec_1
        self.n_dec_2 = n_dec_2
        self.n_input = n_input
        self.n_z = n_z
        self.denoise = denoise
        self.sigma = sigma
        self.pre_lr = pre_lr
        self.alt_lr = alt_lr
        self.adata = adata
        self.pre_epoches = pre_epoches
        self.alt_epoches = alt_epoches
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.model = AE(n_enc_1=self.n_enc_1, n_enc_2=self.n_enc_2,
                        n_dec_1=self.n_dec_1, n_dec_2=self.n_dec_2,
                        n_input=self.n_input, n_z=self.n_z,
                        denoise=self.denoise, sigma=self.sigma)
        self.zinb_loss = ZINBLoss()

        weights = self._initialize_weights()
        self.Coef = weights['Coef']


    def _initialize_weights(self):
        all_weights = dict()
        all_weights['Coef'] = Parameter(1.0e-4 * torch.ones(size=(len(self.adata.X),len(self.adata.X))))
        return all_weights


    def pre_train(self):
        self.model.train()
        log_interval = 1
        optimizer = Adam(self.parameters(), lr=self.pre_lr)
        for epoch in range(1, self.pre_epoches+1):
            x_tensor = Variable(torch.Tensor(adata.X))
            x_raw_tensor = Variable(torch.Tensor(adata.raw.X))
            sf_tensor = Variable(torch.Tensor(adata.obs.size_factors))
            z, mean, disp, pi = self.model(x_tensor)
            z_c = torch.matmul(self.Coef, z)
            loss_reconst = 1 / 2 * torch.sum(torch.pow((x_tensor - mean), 2))
            # loss_reconst = 1 / 2 * torch.sum(torch.pow((x_tensor - x_bar), 2))
            loss_reg = torch.sum(torch.pow(self.Coef, 2))
            loss_selfexpress = 1 / 2 * torch.sum(torch.pow((z - z_c), 2))
            loss_zinb = self.zinb_loss(x=x_raw_tensor, mean=mean, disp=disp, pi=pi, scale_factor=sf_tensor)
            loss = (0.2*loss_reconst + self.lambda_1 * loss_reg + self.lambda_2 * loss_selfexpress)**1/10 + loss_zinb
            # loss = (loss_reconst + self.lambda_1 * loss_reg + self.lambda_2 * loss_selfexpress)**1/10 + loss_zinb
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % log_interval == 0:
                print('Train Epoch: {} ''\tLoss: {:.6f}'.format(epoch, loss.item()))
            if epoch == self.pre_epoches:
                print('Pre-training completed')
        return self.Coef.detach().numpy()



    def alt_train(self):
        # self.model.train()
        log_interval = 1
        optimizer = Adam(self.parameters(), lr=self.alt_lr)
        for epoch in range(1, self.alt_epoches + 1):
            x_tensor = Variable(torch.Tensor(adata.X))
            x_raw_tensor = Variable(torch.Tensor(adata.raw.X))
            sf_tensor = Variable(torch.Tensor(adata.obs.size_factors))
            z, mean, disp, pi = self.model(x_tensor)
            z_c = torch.matmul(self.Coef, z)
            loss_reconst = 1 / 2 * torch.sum(torch.pow((x_tensor - mean), 2))
            # loss_reconst = 1 / 2 * torch.sum(torch.pow((x_tensor - x_bar), 2))
            loss_reg = torch.sum(torch.pow(self.Coef, 2))
            loss_selfexpress = 1 / 2 * torch.sum(torch.pow((z - z_c), 2))
            loss_zinb = self.zinb_loss(x=x_raw_tensor, mean=mean, disp=disp, pi=pi, scale_factor=sf_tensor)
            loss = (0.2*loss_reconst + self.lambda_1 * loss_reg + self.lambda_2 * loss_selfexpress)**1/10 + loss_zinb
            # loss = (loss_reconst + self.lambda_1 * loss_reg + self.lambda_2 * loss_selfexpress)**1/10 + loss_zinb
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % log_interval == 0:
                print('Train Epoch: {} ''\tLoss: {:.6f}'.format(epoch, loss.item()))
            if epoch == self.alt_epoches:
                print('Alt-training completed')

        return self.Coef.detach().numpy()


data_mat = h5py.File('Human1.h5')
x = np.array(data_mat['X'])
y = np.array(data_mat['Y'])
data_mat.close()
print(x.shape)
print(y.shape)

# preprocessing scRNA-seq read counts matrix
adata = sc.AnnData(x)
adata.obs['Group'] = y

adata = read_dataset(adata,
                transpose=False,
                test_split=False,
                copy=True)

adata = normalize(adata,
                size_factors=True,
                normalize_input=True,
                logtrans_input=True,
                select_hvg=True)


print(adata.X.shape)
print(y.shape)

x_sd = adata.X.std(0)
x_sd_median = np.median(x_sd)
print("median of gene sd: %.5f" % x_sd_median)
sd = 2.5

net = Deep_Sparse_Subspace_Clustering(n_enc_1=256, n_enc_2=32, n_dec_1=32, n_dec_2=256, n_input=2000,
                                      n_z=10, denoise=False, sigma=2.0, pre_lr=0.002, alt_lr=0.001,
                                      adata=adata, pre_epoches=200, alt_epoches=100, lambda_1=1.0, lambda_2=0.5)

Coef_1 = net.pre_train()
Coef_2 = net.alt_train()


Coef = thrC(Coef_2, ro=1.0)
pred_label, _ = post_proC(Coef, 14, 11, 7.0)
y = y.astype(np.int64)
pred_label = pred_label.astype(np.int64)
eva(y, pred_label)








