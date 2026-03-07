import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .custom_transformer_blocks import Block

class MultimodalDecoder(nn.Module):
    def __init__(self, use_target_context, embed_dim=128, future_steps=60, k=6) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.future_steps = future_steps
        self.k = k

        self.attn_depth = 3
        dpr = [x.item() for x in torch.linspace(0, 0.2, self.attn_depth)]
        
        self.ctx_blks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=8,
                mlp_ratio=4.0,
                qkv_bias=False,
                drop_path=dpr[i],
                cross_attn=True,
                kdim=embed_dim,
                vdim=embed_dim
            )
            for i in range(self.attn_depth)
        )

        if use_target_context:
            self.target_blks = nn.ModuleList(
            Block(
                    dim=embed_dim,
                    num_heads=8,
                    mlp_ratio=4.0,
                    qkv_bias=False,
                    drop_path=dpr[i],
                    cross_attn=True,
                    kdim=embed_dim,
                    vdim=embed_dim
                )
                for i in range(self.attn_depth)
            )

        self.pi = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2, 1),
        )

        self.loc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2, future_steps * 2),
        )

        self.pi_norm = nn.Softmax(dim=-1)

        self.mode_embed = nn.Embedding(self.k, embedding_dim=embed_dim)  
        #self.new_mode_embed = nn.Embedding(self.k, embedding_dim=embed_dim)  

        self.initialize_weights()
        return


    def initialize_weights(self):
        nn.init.normal_(self.mode_embed.weight, std=0.02)
        #nn.init.normal_(self.new_mode_embed.weight, std=0.02)
        self.apply(self._init_weights)
        return


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        return


    def forward(self, x, x_encoder, key_padding_mask, N, aux):
        B = x.shape[0]

        kv = x_encoder
        kv_ctx = kv
        mask_ctx =  key_padding_mask

        intention_query = self.mode_embed.weight.view(1, self.k, self.embed_dim).repeat(B, 1, 1)
        for ali in range(self.attn_depth):
            # cross attention to agent-centric features
            intention_query = self.ctx_blks[ali](intention_query, k=kv_ctx, v=kv_ctx, key_padding_mask=mask_ctx)
            # if auxiliary data contains target-centric features
            if len(aux) > 3:
                intention_query = intention_query.view(-1, 1, self.embed_dim)
                target_kv = aux[3].view(B*self.k, -1, self.embed_dim)
                target_key_padding_mask = aux[4].view(B*self.k, -1)   
                intention_query = self.target_blks[ali](intention_query, k=target_kv, v=target_kv, key_padding_mask=target_key_padding_mask)
                intention_query = intention_query.view(B, self.k, self.embed_dim) 

        #  output heads
        loc = self.loc(intention_query).view(B, self.k, self.future_steps, 2)
        pi = self.pi(intention_query).squeeze(2)
        return loc, pi, intention_query
