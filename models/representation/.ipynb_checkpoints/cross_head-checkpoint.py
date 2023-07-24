import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

class Covariance(nn.Module):
    def __init__(self,
                remove_mean=True,
                conv=False,
        ):
        super(Covariance, self).__init__()
        self.remove_mean = remove_mean
        self.conv = conv

    def _remove_mean(self, x):
        x = x.transpose(-1, -2)
        _mean = F.adaptive_avg_pool2d(x, (1,1))
        x = x - _mean
        x = x.transpose(-1, -2)
        return x

    def remove_mean_(self, x):
        _mean = F.adaptive_avg_pool2d(x, (1,1))
        x = x - _mean
        return x

    def _cov(self, x):
        batchsize, d, N = x.size()
        x = x.transpose(-1, -2)
        y = (1. / N ) * (x.bmm(x.transpose(1, 2)))
        return y
    
    def _cross_cov(self, x1, x2):
        batchsize1, N1, d1 = x1.size()
        batchsize2, N2, d2 = x2.size()
        assert batchsize1 == batchsize2
        assert N1 == N2
        x1 = x1.transpose(-1, -2)
        x2 = x2.transpose(-1, -2)

        y = (1. / N1) * (x1.bmm(x2.transpose(1, 2)))
        return y
    
    def cross_cov(self, x1, x2):
        batchsize1, d1, h1, w1 = x1.size()
        batchsize2, d2, h2, w2 = x2.size()
        N1 = h1*w1
        N2 = h2*w2
        assert batchsize1 == batchsize2
        assert N1 == N2
        x1 = x1.view(batchsize1, d1, N1)
        x2 = x2.view(batchsize2, d2, N2)

        y = (1. / N1) * (x1.bmm(x2.transpose(1, 2)))
        return y

    def forward(self, x, y=None):
        if self.remove_mean:
            x = self.remove_mean_(x) if self.conv else self._remove_mean(x)
            if y is not None:
                y = self.remove_mean_(y) if self.conv else self._remove_mean(y)          
        if y is not None:
            if self.conv:
                x = self.cross_cov(x, y)
            else:
                x = self._cross_cov(x, y)
        else:
            x = self._cov(x)
        return x

class Down_classifer(nn.Module):
    def __init__(self, cross_type='n-1', in_dim=768, hidden_dim=512, num_heads=4, num_tokens=197, num_blocks=2,num_classes=1000):
        super().__init__()
        self.cross_type = cross_type
        self.num_heads = num_heads

        self.proj = nn.Linear(in_dim, hidden_dim)
        # self.proj_ = nn.Linear(hidden_dim//num_heads, hidden_dim//num_heads)
        # self.norm = nn.LayerNorm(768)
        # self.bn = nn.BatchNorm1d(num_tokens)
        self.ln = nn.LayerNorm(hidden_dim//num_heads)
        # self.ln1 = nn.LayerNorm(hidden_dim//num_heads)
        # self.ln = nn.LayerNorm(hidden_dim)
        self.cov = Covariance()
        # self.drop=nn.Dropout2d(0.1)
        # self.classifier1 = nn.Linear(768, num_classes)
        self.classifier2 = nn.Linear(2883, num_classes)
        # self.classifier2 = nn.Linear(3072, num_classes)
        # self.w1 = nn.parameter.Parameter(torch.ones((3969)), requires_grad=True)
        # self.w2 = nn.parameter.Parameter(torch.ones((3969)), requires_grad=True)

        self.downblocks = nn.ModuleList()
        self.downblocks.append(nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=0, bias=False),
            nn.GELU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=0, bias=False),
            # nn.GELU(),
            # nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=False),
            ))
        
        # for i in range(num_blocks):
        #     self.downblocks.append(nn.Sequential(
        #                 # nn.Conv2d(3, 3, kernel_size=(1,3), stride=(1,2), padding=(0,1), bias=False),
        #                 # nn.Conv2d(3, 3, kernel_size=(1,3), stride=(1,2), padding=(0,0), bias=False),
        #                 # nn.GELU(),
        #                 # nn.Conv2d(3, 3, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=False),
        #                 # nn.Conv2d(3, 3, kernel_size=(3,1), stride=(2,1), padding=(0,0), bias=False),
        #                 nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=0, bias=False),
        #                 )
        #                 )
        #     if i != num_blocks-1:
        #         self.downblocks.append(nn.GELU())
        
        # self.norm = nn.ModuleList()
        # for i in range(num_heads):
        #     self.norm.append(nn.BatchNorm1d(num_tokens))

    def _signed_sqrt(self, x):
        x = torch.mul(x.sign(), torch.sqrt(x.abs()))
        return x

    def _l2norm(self, x):
        x = nn.functional.normalize(x, dim=2)
        return x

    def epn(self, x):
        x = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-6)
        x = torch.nn.functional.normalize(x, dim=2)
        return x

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, D = x.shape
    
        channels_per_group = num_channels // groups

        x = x.view(batchsize, groups,
                channels_per_group, D)
    
        x = torch.transpose(x, 1, 2).contiguous()
    
        # flatten
        x = x.view(batchsize, -1, D)
    
        return x
    
    def forward(self, x):
        # cls_token = self.norm(x[:,0,:])
        # x = self.proj(x[:,1:,:])

        x = self.proj(x)
        B, L, D = x.shape
        # x = self.channel_shuffle(x, 197)
        # cov_list = self.cov(self.ln(x), self.ln(x)).unsqueeze(1)
        # print(x)
                
        # shuffle D
        # x = x.reshape(B, L, self.num_heads, D//self.num_heads).permute(0,1,3,2).reshape(B,L,D)
        
        # divide head
        heads = x.reshape(B, L, self.num_heads, D//self.num_heads).permute(2, 0, 1, 3)
        
        # if self.cross_type == 'n-1':
        #     cov_list = self.cov(self.ln(heads[0]), self.ln(heads[1])).unsqueeze(1)
        #     cov_list = self._l2norm(cov_list)

        #     for i in range(0, self.num_heads):
        #         cov = self.cov(self.ln(heads[i]), self.ln(heads[i])).unsqueeze(1)
        #         cov = self._l2norm(cov)

        #         cov_list = torch.cat([cov_list, cov], dim=1)
                
        if self.cross_type == 'n-1':
            cov_list = self.cov(self.ln(heads[0]), self.ln(heads[1])).unsqueeze(1)
            cov_list = self._l2norm(cov_list)

            for i in range(2, self.num_heads):
                cov = self.cov(self.ln(heads[0]), self.ln(heads[i])).unsqueeze(1)
                cov = self._l2norm(cov)

                cov_list = torch.cat([cov_list, cov], dim=1)

        # elif self.cross_type == 'n/2':
        #     cov_list = self.cov(self.ln(heads[0]), self.ln(heads[1])).unsqueeze(1)
        #     for i in range(1, self.num_heads-1):
        #         cov = self.cov(self.ln(heads[i]), self.ln(heads[i+1])).unsqueeze(1)
        #         cov_list = torch.cat([cov_list, cov], dim=1)
                
            # cov_list = self.cov(self.ln(heads[0]), self.ln(heads[self.num_heads//2])).unsqueeze(1)
            # for i in range(1, self.num_heads//2):
            #     cov = self.cov(self.ln(heads[i]), self.ln(heads[i+self.num_heads//2])).unsqueeze(1)
            #     cov_list = torch.cat([cov_list, cov], dim=1)

        # elif self.cross_type == 'cn2':
        #     cov_list = self.cov(self.bn(heads[0]), self.bn(heads[self.num_heads//2])).unsqueeze(1)
        #     for i in range(1, self.num_heads//2):
        #         cov = self.cov(self.bn(heads[i]), self.bn(heads[i+self.num_heads//2])).unsqueeze(1)
        #         cov_list = torch.cat([cov_list, cov], dim=1)
        # else:
        #     assert 0, 'Please choose from [n-1, n/2, cn2] !'

        
        for layer in self.downblocks:
            cov_list = layer(cov_list)

        # cross_cov = cov_list.mean(dim=1).view(B, -1)
        cross_cov = cov_list.view(B, -1)
        # cross_cov = self.w1 * cross_cov[ :, 0, :] + self.w2 * cross_cov[ :, 1, :]
        
        # cls_token = self.classifier1(cls_token)
        cross_cov = self.classifier2(cross_cov)

        # return (cls_token + cross_cov)/2
        return cross_cov

if __name__ == '__main__':
    # load model
    x = torch.randn((2,197,768))
    model = Down_classifer(in_dim=768,)
    y = model(x)
    print(y.shape)