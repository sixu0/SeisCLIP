# Revised from Open-CLIP
# @Time    : 14/9/23
# @Author  : Xu Si
# @Affiliation  : University of Science and Technolog of China
# @Email   : xusi@mail.ustc.edu.cn

from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .ast_models import *


from typing import List
from typing import Tuple
from typing import Union
from typing import Optional

ClipFeatures = Tuple[
    Optional[torch.Tensor],  # audio
    Optional[torch.Tensor]   # text
]


ClipOutput = Tuple[
    Tuple[ClipFeatures],
    Optional[torch.Tensor],   # loss
    Optional[torch.Tensor]   # loss
]

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        # self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class FullyConnectedLayer(nn.Module):
    def __init__(self, in_features, out_features, activation=QuickGELU(), use_batchnorm=True):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        self.use_batchnorm = use_batchnorm
        
        if self.use_batchnorm:
            self.batchnorm = LayerNorm(out_features)
        
    def forward(self, x):
        out = self.linear(x)
        
        if self.use_batchnorm:
            out = self.batchnorm(out)
        
        out = self.activation(out)
        
        return out
    
class Info_embedding(nn.Module):
    def __init__(self, width:int, hid_feature:int, layers:int, out_dim: int):
        super().__init__() 
        self.width = width
        self.hid_feature = hid_feature
        self.layers = layers
        self.out_dim = out_dim
        
        self.FCN_input = FullyConnectedLayer(width, hid_feature)
        self.FCN = nn.Sequential(*[FullyConnectedLayer(hid_feature, hid_feature) for _ in range(layers)])
        self.proj = nn.Parameter(torch.randn(hid_feature, out_dim))
        
        
    def forward(self, x: torch.Tensor):
        x = self.FCN_input(x)
        x = self.FCN(x)
        if self.proj is not None:
            x = x @ self.proj
        return x
        

class AUDIO_CLIP(nn.Module):
    def __init__(self,
                 device_name: str,
                 embed_dim: int,
                 # text
                 text_input: int,
                 text_width: int,
                 text_layers: int,
                 spec_fdim: int = 50,
                 spec_tdim: int = 120,
                 spec_tstr: int = 10,
                 spec_fstr: int = 10,
                 spec_model_size: str = 'base224',
                 imagenet_pretrain: bool = True,
                 audioset_pretrain: bool = False,
                 load_pretrain_patch: int = 120
                 ):
        super().__init__()
        
        self.device = device_name
        
    
        self.info = Info_embedding(
            width = text_input,
            hid_feature = text_width,
            layers = text_layers,
            out_dim = embed_dim
        )
        
        
        self.spec = ASTModel(
            input_fdim = spec_fdim, 
            input_tdim = spec_tdim, 
            tstride = spec_tstr,
            fstride = spec_fstr,
            model_size = spec_model_size,
            imagenet_pretrain= imagenet_pretrain, 
            audioset_pretrain= audioset_pretrain,
            load_pretrain_patch = load_pretrain_patch
            
        )
        


        # self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale_at = torch.nn.Parameter(torch.log(torch.ones([]) * 100))

        # self.initialize_parameters()

#     def initialize_parameters(self):


#         proj_std = (self.signal.width ** -0.5) * ((2 * self.signal.layers) ** -0.5)
#         attn_std = self.signal.width ** -0.5
#         fc_std = (2 * self.signal.width) ** -0.5
#         for block in self.signal.resblocks:
#             nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
#             nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
#             nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
#             nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.signal.width ** -0.5)

    # def build_attention_mask(self):
    #     # lazily create causal attention mask, with full attention between the vision tokens
    #     # pytorch uses additive attention mask; fill with -inf
    #     mask = torch.empty(self.context_length, self.context_length)
    #     mask.fill_(float("-inf"))
    #     mask.triu_(1)  # zero out the lower diagonal
    #     return mask

    @property
    def dtype(self):
        return self.spec.v.head.weight.dtype
    
    def update(self,t_dim,f_dim):
        self.spec.update_position_embed(t_dim,f_dim)
    
    def encode_audio(self, audio):
        feature,_ = self.spec(audio.type(self.dtype))
        return feature
    
    def get_audio_total_feature(self, audio):
        _,total_feature = self.spec(audio.type(self.dtype))
        return total_feature
    
    def encode_text(self, text):
        return self.info(text.type(self.dtype))
    

    def forward(self, text, audio):
        text_features = self.encode_text(text)
        audio_features = self.encode_audio(audio)
        
        
        if audio is not None:
            audio_features = self.encode_audio(audio)
            audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)

        if text is not None:
            text_features = self.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        features: ClipFeatures = (audio_features, text_features)
        
        logit_scale_at = torch.clamp(self.logit_scale_at.exp(), min=1.0, max=100.0)
        
        # print('features from model',signal_features,text_features)

        if (audio_features is not None) and (text_features is not None):
            logits_audio_text = logit_scale_at * audio_features @ text_features.T


        loss = self.loss_fn(logits_audio_text)
        # if audio is not None and loss is not None:
        #     loss = loss + self.audio.loss_ttf(self.device)

        return (features), logits_audio_text, loss

    def loss_fn(self, logits_audio_text):

        if logits_audio_text is not None:
            batch_size = logits_audio_text.shape[0]
        else:
            return None

        reference = torch.arange(
            batch_size,
            dtype=torch.int64,
            device=self.device
        )

        loss = torch.tensor(0.0, dtype=self.dtype)

        num_modalities: int = 0
        scale = torch.tensor(1.0, dtype=self.dtype)

        if logits_audio_text is not None:
            loss_at = F.cross_entropy(
                logits_audio_text, reference
            ) + F.cross_entropy(
                logits_audio_text.transpose(-1, -2), reference
            )
            loss = loss + loss_at
            num_modalities += 1

        for idx in range(num_modalities):
            scale = scale * (idx + 1)

        return loss / scale

    @property
    def loss_fn_name(self) -> str:
        return 'Cross Entropy'
    
