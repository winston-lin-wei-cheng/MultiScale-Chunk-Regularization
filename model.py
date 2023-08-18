#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Wei-Cheng (Winston) Lin

Defined Cross-Modal Attention Transformers.
"""
import torch 
from torch import nn


class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class MaskSelfAttenLayer(nn.Module):
    
    def __init__(self, d_model, num_heads, dim_feedforward, dropout_rate):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        # feed-forward module
        self.dropout = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()
    
    def forward(self, x, x_mask):
        # with intermediate maskings
        x_h1 = self.self_attn(query=x, key=x, value=x, key_padding_mask=x_mask)[0]
        x_mask = ~x_mask                    # reverse boolean values
        x_mask = x_mask.long()              # obtain the binary 1/0 masking matrix
        x_h1 = x_h1 * x_mask.unsqueeze(-1)  # apply masks on the hidden outputs
        x_h1 = self.dropout(x_h1)
        x_h1 = self.norm1(x + x_h1)
        x_h1 = x_h1 * x_mask.unsqueeze(-1)  # apply masks on the hidden outputs
        x_h2 = self.dropout(self.linear2(self.dropout(self.activation(self.linear1(x_h1)))))
        x_h2 = self.norm2(x_h1 + x_h2)
        return x_h2

class MaskCoAttenLayer(nn.Module):
    
    def __init__(self, d_model, num_heads, dim_feedforward, dropout_rate):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        # feed-forward module
        self.dropout = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()
    
    def forward(self, x1, x2, x1_mask):
        # with intermediate maskings
        x_h1 = self.self_attn(query=x2, key=x1, value=x1, key_padding_mask=x1_mask)[0]
        x1_mask = ~x1_mask                  # reverse boolean values
        x1_mask = x1_mask.long()            # obtain the binary 1/0 masking matrix
        x_h1 = x_h1 * x1_mask.unsqueeze(-1) # apply masks on the hidden outputs
        x_h1 = self.dropout(x_h1)
        x_h1 = self.norm1(x1 + x_h1)
        x_h1 = x_h1 * x1_mask.unsqueeze(-1) # apply masks on the hidden outputs
        x_h2 = self.dropout(self.linear2(self.dropout(self.activation(self.linear1(x_h1)))))
        x_h2 = self.norm2(x_h1 + x_h2)
        return x_h2

class StackedENCLayers(nn.Module):

    def __init__(self, d_model, num_heads, dim_feedforward, dropout_rate):
        super().__init__()
        # speech Transformer (3*SelfAttenL + 2*CoAttenL)
        self.l1_sph_sph = MaskSelfAttenLayer(d_model, num_heads, dim_feedforward, dropout_rate)
        self.l2_sph_sph = MaskSelfAttenLayer(d_model, num_heads, dim_feedforward, dropout_rate)
        self.l3_sph_sph = MaskSelfAttenLayer(d_model, num_heads, dim_feedforward, dropout_rate)
        self.l4_sph_txt = MaskCoAttenLayer(d_model, num_heads, dim_feedforward, dropout_rate)
        self.l5_sph_txt = MaskCoAttenLayer(d_model, num_heads, dim_feedforward, dropout_rate)
        
        # text Transformer (3*SelfAttenL + 2*CoAttenL)
        self.l1_txt_txt = MaskSelfAttenLayer(d_model, num_heads, dim_feedforward, dropout_rate)
        self.l2_txt_txt = MaskSelfAttenLayer(d_model, num_heads, dim_feedforward, dropout_rate)
        self.l3_txt_txt = MaskSelfAttenLayer(d_model, num_heads, dim_feedforward, dropout_rate)
        self.l4_txt_sph = MaskCoAttenLayer(d_model, num_heads, dim_feedforward, dropout_rate)
        self.l5_txt_sph = MaskCoAttenLayer(d_model, num_heads, dim_feedforward, dropout_rate)

    def forward(self, xs, xt, xs_mask, xt_mask):
        # speech self-atten path
        xs = self.l1_sph_sph(xs, xs_mask)
        xs = self.l2_sph_sph(xs, xs_mask)
        xs = self.l3_sph_sph(xs, xs_mask)
        # text self-atten path
        xt = self.l1_txt_txt(xt, xt_mask)
        xt = self.l2_txt_txt(xt, xt_mask)
        xt = self.l3_txt_txt(xt, xt_mask)
        # cross co-atten path
        xs_co1 = self.l4_sph_txt(xs, xt, xs_mask)
        xt_co1 = self.l4_txt_sph(xt, xs, xt_mask)
        xs_co2 = self.l5_sph_txt(xs_co1, xt_co1, xs_mask)
        xt_co2 = self.l5_txt_sph(xt_co1, xs_co1, xt_mask)
        return xs_co2, xt_co2

class TransformerENC_CoAtten(torch.nn.Module): 
    def __init__(self, hidden_dim, nhead, max_len):
        super(TransformerENC_CoAtten, self).__init__()
        
        # init feature-dim
        inp_featD_sph = 1024*7 # audio-dim after TPP
        inp_featD_txt = 768    # text-dim
        
        # input projection layer & PE layer
        self.proj_sph = nn.Linear(inp_featD_sph, hidden_dim, bias=False)
        self.proj_txt = nn.Linear(inp_featD_txt, hidden_dim, bias=False)
        self.PE = PositionalEncoding(num_hiddens=hidden_dim, dropout=0.1, max_len=max_len)
        
        # main Transformer model
        self.trans = StackedENCLayers(d_model=hidden_dim, num_heads=nhead, dim_feedforward=2*hidden_dim, dropout_rate=0.3)
        
        # MTL output heads
        self.top_layer_act = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(hidden_dim, 1))
        self.top_layer_dom = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(hidden_dim, 1))
        self.top_layer_val = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(hidden_dim, 1))

    def forward(self, inp_sph, inp_txt, msk_sph, msk_txt):
        # proj input sequence + PE info
        inp_sph = self.PE(self.proj_sph(inp_sph))
        inp_txt = self.PE(self.proj_txt(inp_txt))
        out_sph, out_txt = self.trans(inp_sph, inp_txt, msk_sph, msk_txt)
        # time-step 0 is the global token for sentence-level emotion predictions
        out_sph_rep = out_sph[:,0,:]
        out_txt_rep = out_txt[:,0,:]
        # simple concat fusion
        out_embd = torch.cat((out_sph_rep, out_txt_rep), dim=1)
        # MTL predictions
        pred_act = self.top_layer_act(out_embd)
        pred_dom = self.top_layer_dom(out_embd)
        pred_val = self.top_layer_val(out_embd)
        return pred_act.squeeze(1), pred_dom.squeeze(1), pred_val.squeeze(1)
