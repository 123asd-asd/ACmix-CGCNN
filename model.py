from __future__ import print_function, division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch_geometric.nn as geom_nn
from torch_geometric.data import Data

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

import torch.nn.functional as F

def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc

def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]

def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)

class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride

        pad_att_h = min(self.padding_att, h - 1)
        pad_att_w = min(self.padding_att, w - 1)

        if h >= self.kernel_att and w >= self.kernel_att:
            pe = self.conv_p(position(h, w, x.is_cuda))

            q_att = q.view(b * self.head, self.head_dim, h, w) * scaling
            k_att = k.view(b * self.head, self.head_dim, h, w)
            v_att = v.view(b * self.head, self.head_dim, h, w)

            if self.stride > 1:
                q_att = stride(q_att, self.stride)
                q_pe = stride(pe, self.stride)
            else:
                q_pe = pe

            dynamic_pad = nn.ReflectionPad2d((pad_att_w, pad_att_w, pad_att_h, pad_att_h))

            unfold_k = self.unfold(dynamic_pad(k_att)).view(b * self.head, self.head_dim,
                                                            self.kernel_att * self.kernel_att, h_out,
                                                            w_out)
            unfold_rpe = self.unfold(dynamic_pad(pe)).view(1, self.head_dim, self.kernel_att * self.kernel_att, h_out,
                                                           w_out)

            att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(
                1)
            att = self.softmax(att)

            out_att = self.unfold(dynamic_pad(v_att)).view(b * self.head, self.head_dim,
                                                           self.kernel_att * self.kernel_att,
                                                           h_out, w_out)
            out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        else:
            q_att = q.view(b, self.head, self.head_dim, h * w) * scaling
            k_att = k.view(b, self.head, self.head_dim, h * w)
            v_att = v.view(b, self.head, self.head_dim, h * w)

            att = torch.matmul(q_att.transpose(2, 3), k_att)
            att = self.softmax(att)

            out_att = torch.matmul(att, v_att.transpose(2, 3))
            out_att = out_att.transpose(2, 3).contiguous().view(b, self.out_planes, h_out, w_out)

        f_all = self.fc(torch.cat(
            [q.view(b, self.head, self.head_dim, h * w), k.view(b, self.head, self.head_dim, h * w),
             v.view(b, self.head, self.head_dim, h * w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        out_conv = self.dep_conv(f_conv)

        return self.rate1 * out_att + self.rate2 * out_conv

class ConvLayer(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len):
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

        self.ACmix = ACmix(in_planes=atom_fea_len, out_planes=atom_fea_len)

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)

        out = out.unsqueeze(-1).unsqueeze(-1)
        
        if out.size(2) == 1 and out.size(3) == 1:
            out = F.interpolate(out, size=(3, 3), mode='nearest')
            out = self.ACmix(out)
            out = F.interpolate(out, size=(1, 1), mode='nearest')
        else:
            out = self.ACmix(out)

        out = out.squeeze(-1).squeeze(-1)

        return out

class CrystalGraphConvNet(nn.Module):
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False,dropout_prob=0.2, n_extra_features=None):
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])

        self.extra_feat_net = nn.Sequential(
            nn.Linear(n_extra_features, h_fea_len // 4),
            nn.BatchNorm1d(h_fea_len // 4, eps=1e-3, momentum=0.05),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.conv_to_fc = nn.Linear(atom_fea_len + (h_fea_len // 4), h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()

        self.dropout = nn.Dropout(p=dropout_prob)

        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])

            self.dropouts = nn.ModuleList([nn.Dropout(p=dropout_prob)
                                         for _ in range(n_h-1)])

        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, extra_fea):
        atom_fea = atom_fea.float()
        nbr_fea = nbr_fea.float()
        extra_fea = extra_fea.float()

        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)

        crys_fea = self.pooling(atom_fea, crystal_atom_idx)

        extra_fea = self.extra_feat_net(extra_fea)

        combined_fea = torch.cat([crys_fea, extra_fea], dim=1)
        crys_fea = self.conv_to_fc(combined_fea)

        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)

        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)

    def visualize_feature_processing(self, extra_fea):
        print("\nFeature Processing Visualization:")
        print(f"Input features shape: {extra_fea.shape}")
        intermediate = self.extra_feat_net[0](extra_fea)
        print(f"After first linear layer: {intermediate.shape}")
        intermediate = self.extra_feat_net[1](intermediate)
        print(f"After batch norm: {intermediate.shape}")
        intermediate = self.extra_feat_net[2](intermediate)
        print(f"After ReLU: {intermediate.shape}")
        output = self.extra_feat_net(extra_fea)
        print(f"Final extra feature output shape: {output.shape}")
        return {'input_stats': {'mean': extra_fea.mean(dim=0).tolist(), 'std': extra_fea.std(dim=0).tolist()},
                'output_stats': {'mean': output.mean(dim=0).tolist(), 'std': output.std(dim=0).tolist()}}
