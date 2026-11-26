"""Conservatively skip token attention when a function cannot reach the scanner."""
import torch


class GateFilter:
    def __init__(self, recipe, route):
        base,policy,heads=recipe
        self.policy=policy
        self.gate=base['gate']
        self.route=route
        self.positions=torch.tensor([[0,1],[0,2],[0,3]],device='cuda')
        stack=[heads[k] for k in ('w','negative','positive')]
        self.coefficients=torch.tensor([h.coef_[0].tolist() for h in stack],device='cuda',dtype=torch.float64)
        self.intercept=torch.tensor([float(h.intercept_[0]) for h in stack],device='cuda',dtype=torch.float64)

    def __call__(self, p, aux):
        raw=p[:,0]
        data=torch.column_stack((raw,aux[:,0],aux[:,-2:]))
        # Re-read rows must keep their attention scores until the alternative view is known.
        canonical=(raw>self.route['lower']-.0001)&(raw<self.route['upper']+.0001)
        logits=torch.logit(data.clamp(1e-5,1.-1e-5)).double()
        estimation=(logits[:,self.positions]*self.coefficients).sum(2)+self.intercept
        w,u,v=estimation.sigmoid().T
        unqueried=(1.-self.policy.mix)*raw
        no=unqueried+self.policy.mix*u
        yes=unqueried+self.policy.mix*v
        prior=(1.-w)*no+w*yes
        decision=self.gate['base']
        if self.policy.method=='margin':
            value=(yes-no).abs()*w*(1.-w)/(.08+(prior-decision).abs())
        else:
            value=(1.-w)*(no-decision).clamp_min(0.)+w*(yes-decision).clamp_min(0.)
            value-=(prior-decision).clamp_min(0.)
        return canonical | (value>self.gate['utility']-.0001)
