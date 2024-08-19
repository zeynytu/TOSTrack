import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lib.models.ostrack.utils import MLP
from timm.models.layers import trunc_normal_


class SimilarityModel(nn.Module):
    def __init__(self, num_heads=12, hidden_dim=768, nlayer_head=3):
        super().__init__()
        self.num_heads = num_heads
        self.score_head = MLP(hidden_dim, hidden_dim, 1, nlayer_head)
        self.scale = hidden_dim ** -0.5

        self.proj_q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.proj_k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.proj_v = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.score_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        trunc_normal_(self.score_token, std=.02)
       

    def forward(self, search_feat, template_feat):
        """
        :param search_box: with normalized coords. (x0, y0, x1, y1)
        :return:
        """
        b, hw, c, = search_feat.shape
        x = self.score_token.expand(b, -1, -1)
        x = self.norm1(x)

        # only 1 decoder used
        # search_box_feat = rearrange(search_feat, 'b c h w -> b (h w) c')
        # template_feat = rearrange(template_feat, 'b c h w -> b (h w) c')
        merged_feature = torch.cat((template_feat, search_feat), dim=1)
        q = rearrange(self.proj_q(x), 'b t (n d) -> b n t d', n=self.num_heads)
        k = rearrange(self.proj_k(merged_feature), 'b t (n d) -> b n t d', n=self.num_heads)
        v = rearrange(self.proj_v(merged_feature), 'b t (n d) -> b n t d', n=self.num_heads)

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')   # (b, 1, c)
        x = self.proj(x)
        x = self.norm1(x)
        out_scores = self.score_head(x)  # (b, 1, 1)

        return out_scores