"""Shared compact encoder with WeakSATD auxiliary outputs."""
import torch
from torch import nn


def pooling(hidden, mask):
    masked = mask.unsqueeze(-1)
    average = (hidden * masked).sum(1) / mask.sum(1,keepdim=True)
    maximum = hidden.masked_fill(~masked.bool(), -torch.inf).amax(1)
    return torch.cat((average, maximum),1)


class RuleEncoder(nn.Module):
    def __init__(self, encoder, conditional=False):
        super().__init__()
        self.encoder = encoder
        self.width = encoder.width
        self.rule_head = nn.Sequential(nn.LayerNorm(2*self.width),
                                       nn.Linear(2*self.width,22 if conditional else 20))

    def forward(self,tokens,mask,side):
        hidden = self.encoder.token_states(tokens,mask)
        main = self.encoder.readout(hidden,mask,side)
        aux = self.rule_head(pooling(hidden,mask))
        return torch.cat((main,aux),1)


