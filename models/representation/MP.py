import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict
from timm.models.layers import SelectAdaptivePool2d
from timm.models.fx_features import register_notrace_module

def _is_contiguous(tensor: torch.Tensor) -> bool:
    # jit is oh so lovely :/
    # if torch.jit.is_tracing():
    #     return True
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)

@register_notrace_module
class LayerNorm2d(nn.LayerNorm):
    r""" LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        if _is_contiguous(x):
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        else:
            s, u = torch.var_mean(x, dim=1, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x

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

class Moment_Probing_ViT(nn.Module):
    def __init__(self, cross_type='near', in_dim=768, hidden_dim=512, num_heads=4, num_tokens=197, num_blocks=2, num_classes=1000):
        super().__init__()
        self.cross_type = cross_type
        self.num_heads = num_heads

        self.proj = nn.Linear(in_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim//num_heads)
        self.cov = Covariance()
        self.classifier1 = nn.Linear(in_dim, num_classes)
        self.classifier2 = nn.Linear(2883, num_classes)
        
        self.downblocks = nn.ModuleList()
        self.downblocks.append(nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=0, bias=False),
            nn.GELU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=0, bias=False),
            ))

    def _l2norm(self, x):
        x = nn.functional.normalize(x, dim=2)
        return x
    
    def forward(self, cls_token, x):
        x = self.proj(x[:,1:,:])

        B, L, D = x.shape

        # divide head
        heads = x.reshape(B, L, self.num_heads, D//self.num_heads).permute(2, 0, 1, 3)
                
        if self.cross_type == 'n/2':
            cov_list = self.cov(self.ln(heads[0]), self.ln(heads[self.num_heads//2])).unsqueeze(1)
            cov_list = self._l2norm(cov_list)

            for i in range(1, self.num_heads//2):
                cov = self.cov(self.ln(heads[i]), self.ln(heads[i+self.num_heads//2])).unsqueeze(1)
                cov = self._l2norm(cov)
                cov_list = torch.cat([cov_list, cov], dim=1)

        elif self.cross_type == 'near':
            cov_list = self.cov(self.ln(heads[0]),self.ln(heads[1])).unsqueeze(1)
            cov_list = self._l2norm(cov_list)
            
            for i in range(1, self.num_heads-1):
                cov = self.cov(self.ln(heads[i]), self.ln(heads[i+1])).unsqueeze(1)
                cov = self._l2norm(cov)
                cov_list = torch.cat([cov_list, cov], dim=1)

        elif self.cross_type == 'cn2':
            cov_list = self.cov(self.ln(heads[0]), self.ln(heads[1])).unsqueeze(1)
            for i in range(0, self.num_heads-1):
                for j in range(i+1, self.num_heads):
                    cov = self.cov(self.ln(heads[i]), self.ln(heads[j])).unsqueeze(1)
                    cov_list = torch.cat([cov_list, cov], dim=1)
                    
            cov_list = cov_list[:,1:]
        else:
            assert 0, 'Please choose from [one, near, cn2] !'

        
        for layer in self.downblocks:
            cov_list = layer(cov_list)

        cross_cov = cov_list.view(B, -1)
        
        cls_token = self.classifier1(cls_token)
        cross_cov = self.classifier2(cross_cov)

        return (cls_token + cross_cov)/2

class Moment_Probing_CNN(nn.Module):
    def __init__(self, cross_type='near', in_dim=1280, hidden_dim=512, num_heads=4, num_tokens=197, num_blocks=2,num_classes=1000):
        super().__init__()
        self.cross_type = cross_type
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.proj = nn.Conv2d(in_dim, hidden_dim, 1, 1, 0)
        self.ln = nn.LayerNorm(hidden_dim//num_heads)
        self.cov = Covariance()
        norm_layer = partial(LayerNorm2d, eps=1e-6)
        self.cls_head = nn.Sequential(OrderedDict([
            ('global_pool', SelectAdaptivePool2d(pool_type='avg')),
            ('norm', norm_layer(self.in_dim)),
            ('flatten', nn.Flatten(1) if 'avg' else nn.Identity()),
            ('drop', nn.Dropout(0)),
            ('fc', nn.Linear(self.in_dim, num_classes) if num_classes > 0 else nn.Identity())
        ]))

        self.classifier2 = nn.Linear(2883, num_classes)
        
        self.downblocks = nn.ModuleList()
        self.downblocks.append(nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=0, bias=False),
            nn.GELU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=0, bias=False),
            ))

    def _l2norm(self, x):
        x = nn.functional.normalize(x, dim=2)
        return x
    
    def forward(self, x):
        b, c, h, w = x.shape
        cls_token = self.cls_head(x)
        x = self.proj(x)
        x = x.reshape(b, self.hidden_dim, -1)
        B, C, D = x.shape
        # divide head
        heads = x.reshape(B, C//self.num_heads, self.num_heads, D).permute(2, 0, 3, 1)
                        
        if self.cross_type == 'one':
            cov_list = self.cov(self.ln(heads[0]), self.ln(heads[1])).unsqueeze(1)
            cov_list = self._l2norm(cov_list)

            for i in range(2, self.num_heads):
                cov = self.cov(self.ln(heads[0]), self.ln(heads[i])).unsqueeze(1)
                cov = self._l2norm(cov)
                cov_list = torch.cat([cov_list, cov], dim=1)

        elif self.cross_type == 'near':
            cov_list = self.cov(self.ln(heads[0]), self.ln(heads[1])).unsqueeze(1)
            cov_list = self._l2norm(cov_list)

            for i in range(1, self.num_heads-1):
                cov = self.cov(self.ln(heads[i]), self.ln(heads[i+1])).unsqueeze(1)
                cov = self._l2norm(cov)
                cov_list = torch.cat([cov_list, cov], dim=1)

        elif self.cross_type == 'cn2':
            cov_list = self.cov(self.bn(heads[0]), self.bn(heads[self.num_heads//2])).unsqueeze(1)
            cov_list = self._l2norm(cov_list)

            for i in range(1, self.num_heads//2):
                cov = self.cov(self.bn(heads[i]), self.bn(heads[i+self.num_heads//2])).unsqueeze(1)
                cov_list = self._l2norm(cov_list)
                cov_list = torch.cat([cov_list, cov], dim=1)
        else:
            assert 0, 'Please choose from [one, near, cn2] !'

        
        for layer in self.downblocks:
            cov_list = layer(cov_list)
        cross_cov = cov_list.view(B, -1)
        
        cross_cov = self.classifier2(cross_cov)

        return (cls_token + cross_cov)/2

class Moment_Probing_MLP(nn.Module):
    def __init__(self, cross_type='near', in_dim=1280, hidden_dim=512, num_heads=4, num_tokens=197, num_blocks=2, num_classes=100):
        super().__init__()
        self.cross_type = cross_type
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.proj = nn.Conv2d(in_dim, hidden_dim, 1, 1, 0)
        self.ln = nn.LayerNorm(hidden_dim//num_heads)
        self.cov = Covariance()
        self.cls_head = nn.Sequential(OrderedDict([
            ('global_pool', SelectAdaptivePool2d(pool_type='avg')),
            ('flatten', nn.Flatten(1) if 'avg' else nn.Identity()),
            ('drop', nn.Dropout(0)),
            ('fc', nn.Linear(self.in_dim, num_classes) if num_classes > 0 else nn.Identity())
        ]))

        self.classifier2 = nn.Linear(2883, num_classes)
        
        self.downblocks = nn.ModuleList()
        self.downblocks.append(nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=0, bias=False),
            nn.GELU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=0, bias=False),
            ))
        
    def _l2norm(self, x):
        x = nn.functional.normalize(x, dim=2)
        return x
    
    def forward(self, x):
        b, c, h, w = x.shape
        cls_token = self.cls_head(x)
        x = self.proj(x)
        x = x.reshape(b, self.hidden_dim, -1)
        B, C, D = x.shape
        # divide head
        heads = x.reshape(B, C//self.num_heads, self.num_heads, D).permute(2, 0, 3, 1)
                        
        if self.cross_type == 'one':
            cov_list = self.cov(self.ln(heads[0]), self.ln(heads[1])).unsqueeze(1)
            cov_list = self._l2norm(cov_list)

            for i in range(2, self.num_heads):
                cov = self.cov(self.ln(heads[0]), self.ln(heads[i])).unsqueeze(1)
                cov = self._l2norm(cov)
                cov_list = torch.cat([cov_list, cov], dim=1)

        elif self.cross_type == 'near':
            cov_list = self.cov(self.ln(heads[0]), self.ln(heads[1])).unsqueeze(1)
            cov_list = self._l2norm(cov_list)

            for i in range(1, self.num_heads-1):
                cov = self.cov(self.ln(heads[i]), self.ln(heads[i+1])).unsqueeze(1)
                cov = self._l2norm(cov)
                cov_list = torch.cat([cov_list, cov], dim=1)

        elif self.cross_type == 'cn2':
            cov_list = self.cov(self.bn(heads[0]), self.bn(heads[self.num_heads//2])).unsqueeze(1)
            cov_list = self._l2norm(cov_list)

            for i in range(1, self.num_heads//2):
                cov = self.cov(self.bn(heads[i]), self.bn(heads[i+self.num_heads//2])).unsqueeze(1)
                cov_list = self._l2norm(cov_list)
                cov_list = torch.cat([cov_list, cov], dim=1)
        else:
            assert 0, 'Please choose from [one, near, cn2] !'

        
        for layer in self.downblocks:
            cov_list = layer(cov_list)
        cross_cov = cov_list.view(B, -1)
        
        cross_cov = self.classifier2(cross_cov)

        return (cls_token + cross_cov)/2

if __name__ == '__main__':
    # load model
    cls_token = torch.randn((2,768))
    x = torch.randn((2,197,768))
    model = Moment_Probing_ViT(in_dim=768,)
    y = model(cls_token, x)
    print(y.shape)