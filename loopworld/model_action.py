"""Action-conditioned LoopWorld (maze v2). New file; LoopWorldModel unchanged.

Action embedding (zero-init) is added to the timestep embedding, so it enters every
block via AdaLN. At init the model is exactly the loaded checkpoint.
Actions: 0 noop, 1 forward 0.4 m, 2 turn left 22.5 deg, 3 turn right 22.5 deg.
"""
from __future__ import annotations
import torch, torch.nn as nn
from .model import LoopWorldModel

N_ACTIONS = 4
ACTION_NAMES = ["noop", "forward", "left", "right"]


class ActionLoopWorldModel(LoopWorldModel):
    def __init__(self, *a, n_actions: int = N_ACTIONS, **kw):
        super().__init__(*a, **kw)
        self.action_embed = nn.Embedding(n_actions, self.width)
        self.action_mlp = nn.Sequential(nn.SiLU(), nn.Linear(self.width, self.width))
        nn.init.normal_(self.action_embed.weight, std=0.02)
        nn.init.zeros_(self.action_mlp[-1].weight); nn.init.zeros_(self.action_mlp[-1].bias)
        self.cur_action = None  # (B,) long, set before forward

    def forward_exits(self, target_xt, t, target_plucker, context_rgbd, context_plucker, context_valid,
                      num_loops=4, return_all_exits=True):
        tokens, pad = self._embed_views(target_xt, target_plucker, context_rgbd, context_plucker, context_valid)
        t_emb = self.t_embed(t)
        if self.cur_action is not None:
            t_emb = t_emb + self.action_mlp(self.action_embed(self.cur_action)).to(t_emb.dtype)
        rec = bool(self.record_attn_logit)
        h = self.pre(tokens, t_emb, key_padding_mask=pad, record_attn_logit=rec)
        exits, n = [], self.n_tokens_per_view
        for k in range(num_loops):
            for block in self.core:
                h = block(h, t_emb, key_padding_mask=pad, record_attn_logit=rec)
            if return_all_exits or k == num_loops - 1:
                z = self.post(h, t_emb, key_padding_mask=pad, record_attn_logit=rec)
                exits.append(self.unpatchify(self.head(z[:, -n:, :], t_emb), self.in_channels))
        return exits
