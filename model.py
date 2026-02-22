from __future__ import print_function, division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch_geometric.nn as geom_nn
from torch_geometric.data import Data


# 定义h_sigmoid激活函数，这是一种硬Sigmoid函数
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)  # 使用ReLU6实现

    def forward(self, x):
        return self.relu(x + 3) / 6  # 公式为ReLU6(x+3)/6，模拟Sigmoid激活函数


# 定义h_swish激活函数，这是基于h_sigmoid的Swish函数变体
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)  # 使用上面定义的h_sigmoid

    def forward(self, x):
        return x * self.sigmoid(x)  # 公式为x * h_sigmoid(x)


import torch.nn.functional as F


# 定义一个函数来生成位置编码，返回一个包含位置信息的张量
def position(H, W, is_cuda=True):
    # 生成宽度和高度的位置信息，范围在-1到1之间
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)  # 为宽度生成线性间距的位置信息并复制到GPU
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)  # 为高度生成线性间距的位置信息并复制到GPU
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)  # 在CPU上为宽度生成线性间距的位置信息
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)  # 在CPU上为高度生成线性间距的位置信息
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)  # 合并宽度和高度的位置信息，并增加一个维度
    return loc


# 定义一个函数实现步长操作，用于降采样
def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]  # 通过步长来降低采样率


# 初始化函数，将张量的值填充为0.5
def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)  # 使用0.5来填充张量


# 初始化函数，将张量的值填充为0
def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


# 定义ACmix模块的类
class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()  # 调用父类的构造函数
        # 初始化模块参数
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))  # 注意力分支权重
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))  # 卷积分支权重
        self.head_dim = self.out_planes // self.head  # 每个头的维度

        # 定义用于特征变换的卷积层
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)  # 位置编码的卷积层

        # 定义自注意力所需的padding和展开操作
        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        # 定义用于生成动态卷积核的全连接层和深度可分离卷积层
        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=stride)  # 深度可分离卷积层，用于应用动态卷积核

        self.reset_parameters()  # 参数初始化

    def reset_parameters(self):
        init_rate_half(self.rate1)  # 初始化注意力分支权重为0.5
        init_rate_half(self.rate2)  # 初始化卷积分支权重为0.5
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)  # 设置为可学习参数
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)  # 初始化偏置为0

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)  # 应用转换层
        scaling = float(self.head_dim) ** -0.5  # 缩放因子，用于自注意力计算
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride  # 计算输出的高度和宽度

        # 动态计算合适的填充大小
        pad_att_h = min(self.padding_att, h - 1)
        pad_att_w = min(self.padding_att, w - 1)

        # 根据特征图尺寸选择不同的处理策略
        if h >= self.kernel_att and w >= self.kernel_att:
            # 正常处理：特征图足够大
            pe = self.conv_p(position(h, w, x.is_cuda))  # 生成位置编码

            # 为自注意力机制准备q, k, v
            q_att = q.view(b * self.head, self.head_dim, h, w) * scaling
            k_att = k.view(b * self.head, self.head_dim, h, w)
            v_att = v.view(b * self.head, self.head_dim, h, w)

            if self.stride > 1:  # 如果步长大于1，则对q和位置编码进行降采样
                q_att = stride(q_att, self.stride)
                q_pe = stride(pe, self.stride)
            else:
                q_pe = pe

            # 使用动态填充
            dynamic_pad = nn.ReflectionPad2d((pad_att_w, pad_att_w, pad_att_h, pad_att_h))

            # 展开k和位置编码，准备自注意力计算
            unfold_k = self.unfold(dynamic_pad(k_att)).view(b * self.head, self.head_dim,
                                                            self.kernel_att * self.kernel_att, h_out,
                                                            w_out)
            unfold_rpe = self.unfold(dynamic_pad(pe)).view(1, self.head_dim, self.kernel_att * self.kernel_att, h_out,
                                                           w_out)

            # 计算注意力权重
            att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(
                1)  # (b*head, head_dim, 1, h_out, w_out) * (b*head, head_dim, k_att^2, h_out, w_out) -> (b*head, k_att^2, h_out, w_out)
            att = self.softmax(att)

            # 应用注意力权重
            out_att = self.unfold(dynamic_pad(v_att)).view(b * self.head, self.head_dim,
                                                           self.kernel_att * self.kernel_att,
                                                           h_out, w_out)
            out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        else:
            # 简化处理：特征图太小，使用标准自注意力
            # 将特征图reshape为序列形式
            q_att = q.view(b, self.head, self.head_dim, h * w) * scaling
            k_att = k.view(b, self.head, self.head_dim, h * w)
            v_att = v.view(b, self.head, self.head_dim, h * w)

            # 计算标准自注意力
            att = torch.matmul(q_att.transpose(2, 3), k_att)  # [b, head, h*w, h*w]
            att = self.softmax(att)

            # 应用注意力
            out_att = torch.matmul(att, v_att.transpose(2, 3))  # [b, head, h*w, head_dim]
            out_att = out_att.transpose(2, 3).contiguous().view(b, self.out_planes, h_out, w_out)

        # 动态卷积核部分（保持不变）
        f_all = self.fc(torch.cat(
            [q.view(b, self.head, self.head_dim, h * w), k.view(b, self.head, self.head_dim, h * w),
             v.view(b, self.head, self.head_dim, h * w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        out_conv = self.dep_conv(f_conv)

        # 将注意力分支和卷积分支的输出相加
        return self.rate1 * out_att + self.rate2 * out_conv


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len  # 卷积层中原子特征维度
        self.nbr_fea_len = nbr_fea_len  # 邻居特征维度
        self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len,
                                 2 * self.atom_fea_len)  # 全连接层，聚合中心原子与邻居原子特征的核心线性层
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

        self.ACmix = ACmix(in_planes=atom_fea_len, out_planes=atom_fea_len)

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):

        N, M = nbr_fea_idx.shape  # N：当前批次中原子的总数量（每个原子作为 "中心原子"）。M：每个中心原子的邻居数量（固定值，不足时可能用填充）。例如，若有 1000 个原子，每个原子最多 6 个邻居，则 nbr_fea_idx 形状为 (1000, 6)，存储每个原子的 6 个邻居的索引。
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)  # 变为三维
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len * 2)).view(N, M, self.atom_fea_len * 2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)  # 变为二维
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)

        # 应用SAFM模块
        out = out.unsqueeze(-1).unsqueeze(-1)  # [4648, 32, 1, 1]

        # 如果特征图太小，先进行上采样
        if out.size(2) == 1 and out.size(3) == 1:
            # 上采样到3x3或更大的尺寸
            out = F.interpolate(out, size=(3, 3), mode='nearest')
            out = self.ACmix(out)
            # 如果需要，可以再下采样回1x1
            out = F.interpolate(out, size=(1, 1), mode='nearest')
        else:
            out = self.ACmix(out)

        out = out.squeeze(-1).squeeze(-1)

        return out


class GraphComplexityModule(nn.Module):
    """
    图复杂度计算模块
    简单但全面的多维度度量
    """

    def __init__(self, max_atoms=100, max_elements=10, max_neighbors=12):
        super(GraphComplexityModule, self).__init__()
        self.max_atoms = max_atoms
        self.max_elements = max_elements
        self.max_neighbors = max_neighbors

        # ==================== 新增：自适应融合权重参数 ====================
        # 可学习的三个维度融合权重（初始化为[0.4, 0.3, 0.3]）
        self.fusion_weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.3]))

        # 可学习的权重映射参数（初始化为0.8和0.7）
        self.weight_base = nn.Parameter(torch.tensor(0.8))  # 基础权重
        self.weight_range = nn.Parameter(torch.tensor(0.7))  # 权重范围
        # ==================== 新增结束 ====================

        # 用于数值稳定性
        self.register_buffer('eps', torch.tensor(1e-6))

    def forward(self, atom_fea, nbr_fea_idx, crystal_atom_idx):
        """
        计算每个晶体的图复杂度
        Args:
            atom_fea: [N, atom_fea_len] 原子特征
            nbr_fea_idx: [N, M] 邻居索引
            crystal_atom_idx: list 每个晶体的原子索引列表
        Returns:
            complexities: [batch_size] 复杂度值
        """
        batch_size = len(crystal_atom_idx)
        complexities = torch.zeros(batch_size, device=atom_fea.device, dtype=torch.float32)

        for i, atom_indices in enumerate(crystal_atom_idx):
            num_atoms = len(atom_indices)

            if num_atoms == 0:
                complexities[i] = 0.0
                continue

            # 1. 规模复杂度（原子数） - 40%
            # 修改：使用对数归一化处理原子数，避免极端值
            scale_complexity = torch.log1p(torch.tensor(num_atoms, dtype=torch.float32, device=atom_fea.device)) / \
                               torch.log1p(torch.tensor(self.max_atoms, dtype=torch.float32, device=atom_fea.device))

            # 2. 化学多样性（基于原子特征方差） - 30%
            crystal_atom_fea = atom_fea[atom_indices]
            if num_atoms > 1:
                # 修改：使用标准差而不是方差，更稳定
                chem_diversity = torch.std(crystal_atom_fea, dim=0, unbiased=False).mean()
                # 修改：使用sigmoid归一化到[0,1]
                chem_diversity = torch.sigmoid(chem_diversity - 0.5)
            else:
                chem_diversity = torch.tensor(0.0, device=atom_fea.device)

            # 3. 连接复杂度（平均有效邻居数） - 30%
            crystal_nbr_idx = nbr_fea_idx[atom_indices]
            valid_neighbors = (crystal_nbr_idx >= 0).sum().float()
            avg_connectivity = valid_neighbors / float(num_atoms)
            connect_complexity = min(avg_connectivity / self.max_neighbors, 1.0)

            # ==================== 修改：自适应加权融合 ====================
            # 使用softmax确保权重和为1且为正数
            normalized_weights = F.softmax(self.fusion_weights, dim=0)

            # 自适应加权融合
            complexity = (
                    normalized_weights[0] * scale_complexity +
                    normalized_weights[1] * chem_diversity +
                    normalized_weights[2] * connect_complexity
            )
            # ==================== 修改结束 ====================

            # 确保复杂度在[0, 1]范围内
            complexity = torch.clamp(complexity, 0.0, 1.0)
            complexities[i] = complexity

        return complexities


class CrystalGraphConvNet(nn.Module):
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False, dropout_prob=0.2, n_extra_features=None):
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)  # 将原始特征映射到更高维度
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                              nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])  # 图卷积层，通过多次聚合邻居信息，逐步捕捉更复杂的局部结构。

        # 修改
        self.extra_feat_net = nn.Sequential(
            nn.Linear(n_extra_features, h_fea_len // 4),
            # 将原始额外特征（n_extra_features维度，如 19）线性压缩到h_fea_len//4维度（例如h_fea_len=256时，压缩到 64 维），减少计算量并统一特征尺度。
            nn.BatchNorm1d(h_fea_len // 4, eps=1e-3, momentum=0.05),  # 对压缩后的特征做批归一化，加速训练收敛，增强稳定性（eps和momentum是数值稳定性参数）。
            nn.ReLU(),
            nn.Dropout(0.2)
        )  # 额外特征处理部分

        # -------------------- 修改：融合卷积特征与额外特征（核心改动） --------------------
        # 输入维度 = 卷积特征维度(atom_fea_len) + 处理后的额外特征维度(h_fea_len//4)
        self.conv_to_fc = nn.Linear(atom_fea_len + (h_fea_len // 4),
                                    h_fea_len)  # 将 "晶体结构特征" 与 "处理后的额外特征" 融合，并转换到全连接层的输入维度将1和2拼接后的维度映射为3
        self.conv_to_fc_softplus = nn.Softplus()  # 保持原有激活函数

        self.dropout = nn.Dropout(p=dropout_prob)

        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(
                    n_h - 1)])  # 创建 n_h-1 个维度不变的全连接隐藏层，用于对融合后的特征进行深度加工，通过多次线性变换和非线性激活，捕捉更复杂的特征模式，
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h - 1)])  # 每个隐藏层后的Softplus激活函数，增加非线性。

            # 为每个隐藏层添加Dropout
            self.dropouts = nn.ModuleList([nn.Dropout(p=dropout_prob)
                                           for _ in range(n_h - 1)])

        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)  ## 回归，输出1维
        if self.classification:  # 分类任务归一化
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        self.complexity_module = GraphComplexityModule()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, extra_fea, return_complexity=False):

        atom_fea = atom_fea.float()
        nbr_fea = nbr_fea.float()
        extra_fea = extra_fea.float()

        atom_fea = self.embedding(atom_fea)  # 将atom_fea映射到更高维度
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)  # 通过多层图卷积（ConvLayer）迭代更新原子特征。将每个原子的特征与周围邻居的特征进行聚合
            # 此时atom_fea以包含原子的局部结构信息

        crys_fea = self.pooling(atom_fea, crystal_atom_idx)  # 将单原子级特征（atom_fea）聚合为整个晶体的全局特征（crys_fea）

        # crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        # crys_fea = self.conv_to_fc_softplus(crys_fea)

        # crys_fea = self.dropout(crys_fea)

        extra_fea = self.extra_feat_net(extra_fea)  # 额外特征处理，处理方法见上述内容

        combined_fea = torch.cat([crys_fea, extra_fea],
                                 dim=1)  # 将crys_fea, extra_fea在列方向dim=1拼接，拼接后的维度为atom_fea_len + h_fea_len//4
        crys_fea = self.conv_to_fc(combined_fea)  # 通过线性层将拼接后的融合特征映射到全连接层的隐藏维度（h_fea_len，如 128 或 256），作为后续全连接网络的输入

        if self.classification:
            crys_fea = self.dropout(crys_fea)  # 分类任务
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))  # 全连接层特征变换，通过多层全连接层进一步提炼特征。
        out = self.fc_out(crys_fea)  # 通过输出层（self.fc_out）将最终特征映射到预测结果维度。
        if self.classification:
            out = self.logsoftmax(out)

        if return_complexity:
            # 计算复杂度
            complexities = self.complexity_module(atom_fea, nbr_fea_idx, crystal_atom_idx)
            # ==================== 新增：获取自适应权重 ====================
            # 使用可学习的参数计算权重
            weight_base = self.complexity_module.weight_base
            weight_range = self.complexity_module.weight_range
            adaptive_weights = weight_base + weight_range * complexities
            return out, complexities, adaptive_weights
            # ==================== 新增结束 ====================
        else:
            return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) == \
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
