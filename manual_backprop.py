from torch import nn
from torch.autograd import Function
import torch
import torch.nn.functional as F

import pdb

def d_sigmoid(z):
    return (1 - z) * z

def d_tanh(z):
    return 1 - (z * z)

def d_softplus(x):
    return 1 / (1 + torch.exp(-x))

def d_cos(x):
    return -1 * torch.sin(x)


def d_att_func(name, z):
    if name == 'sigmoid':
        return d_sigmoid(z)
    elif name == 'tanh':
        return d_tanh(z)

def d_r_func(name, x):
    if name == 'sigmoid':
        z = torch.sigmoid(x)
        return d_sigmoid(z)
    elif name == 'tanh':
        z = torch.tanh(x)
        return d_tanh(z)
    elif name == 'cos':
        return d_cos(x)
    elif name == 'softplus':
        return d_softplus(x)

def d_th_func(name, x):
    if name == 'softplus':
        return d_softplus(x)
    elif name == 'cos':
        return d_cos(x)

class GATOFunction(Function):
    @staticmethod 
    def forward(ctx, x, hidden, wx_w, wx_b, wr_w, r_func_name, th_func_name, att_func_name, scale):
        r, th = torch.chunk(hidden, 2, dim=-1)

        fx = F.linear(x, wx_w, wx_b).squeeze()
        fx_r1, fx_r2, fx_th = fx.chunk(3,1)

        repeated_r = torch.cat(3 * [r], dim=-1)
        fr = (wr_w * repeated_r).squeeze()
        fr_r1, fr_r2, fr_th = fr.chunk(3,1)

        r_func = getattr(F, r_func_name)
        th_func = getattr(F, th_func_name)
        att_func = getattr(F, att_func_name)        

        r1 = fx_r1 + fr_r1
        r_tilde = r_func(r1)

        att_tilde = att_func(fx_r2 + fr_r2)

        attractor = att_tilde * r
        dr = scale * attractor + 1 * r_tilde

        th_c = fx_th + fr_th
        th_tilde = th_func(th_c)

        dth = th + th_tilde 
        hidden = torch.cat((dr, dth), dim=-1)

        ctx.save_for_backward(
            r1, att_tilde, th_c, r, x, wr_w,
        )
        ctx.scale = scale
        ctx.r_func = r_func_name
        ctx.th_func = th_func_name
        ctx.att_func = att_func_name

        return hidden

    @staticmethod
    def backward(ctx, g_hy):
        r1, att_tilde, th_c, r, x, wr_w = ctx.saved_variables[:6]

        old_g_r, old_g_th = torch.chunk(g_hy, 2, dim=-1)

        repeated_r = torch.cat(3 * [r], dim=-1)

        g_fx_r1 = old_g_r * d_r_func(ctx.r_func, r1)
        g_fx_r2 = old_g_r * d_att_func(ctx.att_func, att_tilde) * r * ctx.scale
        g_fx_th = old_g_th * d_th_func(ctx.th_func, th_c)


        g_fx = torch.cat([g_fx_r1, g_fx_r2, g_fx_th], dim=1)
        g_r1, g_r2, g_r3 = (g_fx * wr_w).chunk(3,1)

        g_r = old_g_r * att_tilde * ctx.scale + g_r1 + g_r2 + g_r3

        # g_th = old_g_th
        g_h = torch.cat([g_r, old_g_th], dim=1)

        g_wx_w = g_fx.t().mm(x) # correct
        g_wx_b = g_fx.sum(dim=0, keepdim=True) # correct

        g_wr_w = torch.sum(g_fx * repeated_r, dim=0) # correct

        return None, g_h, g_wx_w, g_wx_b, g_wr_w, None, None, None, None
