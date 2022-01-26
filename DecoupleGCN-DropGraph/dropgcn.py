import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import paddle.fluid as fluid
import math
from ..registry import BACKBONES
from paddle.nn.initializer import TruncatedNormal, Constant, Normal
from einopspaddle.einops.layers.paddle import einsum

class DropBlock_Ske(nn.Layer):
    def __init__(self, num_point, block_size=7):
        super(DropBlock_Ske, self).__init__()
        self.keep_prob = 0.0
        self.block_size = block_size
        self.num_point = num_point

    def forward(self, input, keep_prob, A):  # n,c,t,v
        self.keep_prob = keep_prob
        if not self.training or self.keep_prob == 1:
            return input
        n, c, t, v = input.shape
        input_abs = fluid.layers.reduce_mean(fluid.layers.reduce_mean(fluid.layers.abs(input), dim=2), dim=1).detach()
        input_abs = input_abs / fluid.layers.reduce_sum(input_abs) * paddle.numel(input_abs)
        gamma = (1. - self.keep_prob) / (1 + 1.92)
        ig = input_abs * gamma
        ig = ig.min()
        M_seed = paddle.bernoulli(fluid.layers.clip(input_abs * gamma, min=ig, max=1.0))
        M = fluid.layers.matmul(M_seed, A)
        M[M > 0.001] = 1.0
        M[M < 0.5] = 0.0
        mask = fluid.layers.reshape(1 - M, (n, 1, 1, self.num_point))

        return input * mask * paddle.numel(mask) / fluid.layers.reduce_sum(mask)


class DropBlockT_1d(nn.Layer):
    def __init__(self, block_size=7):
        super(DropBlockT_1d, self).__init__()
        self.keep_prob = 0.0
        self.block_size = block_size

    def forward(self, input, keep_prob):
        self.keep_prob = keep_prob
        if not self.training or self.keep_prob == 1:
            return input
        n,c,t,v = input.shape

        input_abs = fluid.layers.reduce_mean(fluid.layers.reduce_mean(fluid.layers.abs(input), dim=3), dim=1).detach()
        input_abs = fluid.layers.reshape(input_abs/fluid.layers.reduce_sum(input_abs)*paddle.numel(input_abs),(n,1,t))
        gamma = (1. - self.keep_prob) / self.block_size
        input1 = fluid.layers.reshape(input.transpose((0,1,3,2)), (n,c*v,t))
        ig = input_abs * gamma
        ig = ig.min()
        M = paddle.tile(paddle.bernoulli(fluid.layers.clip(input_abs * gamma, min=ig, max=1.0)), (1,c*v,1))
        Msum = F.max_pool1d(M, kernel_size=[self.block_size], stride=1, padding=self.block_size // 2)
        mask = 1 - Msum
        return fluid.layers.reshape(input1 * mask * paddle.numel(mask) / fluid.layers.reduce_sum(mask), (n,c,v,t)).transpose((0,1,3,2))


class unit_tcn(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, num_point=25, block_size=41):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),stride=(stride, 1))

        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()

        self.dropS = DropBlock_Ske(num_point=num_point)
        self.dropT = DropBlockT_1d(block_size=block_size)

    def forward(self, x, keep_prob, A):
        x = self.bn(self.conv(x))
        x = self.dropT(self.dropS(x, keep_prob, A), keep_prob)
        return x


class unit_tcn_skip(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn_skip, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Layer):
    def __init__(self, in_channels, out_channels, A, groups, num_point, num_subset=3):
        super(unit_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_point = num_point
        self.groups = groups
        self.num_subset = num_subset
        self.DecoupleA = self.create_parameter(shape=(3, groups, num_point, num_point), default_initializer=fluid.initializer.NumpyArrayInitializer(np.reshape(A.astype(np.float32), [3, 1, num_point, num_point]).repeat(groups, axis=1)))
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, 1),
                nn.BatchNorm2D(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn0 = nn.BatchNorm2D(out_channels * num_subset)
        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()

        self.Linear_weight = self.create_parameter(shape=(in_channels, out_channels * num_subset), default_initializer=fluid.initializer.Normal(0, math.sqrt(0.5 / (out_channels * num_subset))))
        self.Linear_bias = self.create_parameter(shape=(1, out_channels * num_subset, 1, 1), default_initializer=fluid.initializer.Constant(1e-6))


        eye_array = []
        for i in range(out_channels):
            eye_array.append(fluid.layers.eye(num_point))

        self.eyes = fluid.dygraph.to_variable(fluid.layers.stack(eye_array))
    def norm(self, A):
        b, c, h, w = A.shape
        A = fluid.layers.reshape(A, (c, self.num_point, self.num_point))
        D_list = fluid.layers.reshape(paddle.sum(A, 1), (c, 1, self.num_point))
        D_list_12 = (D_list + 0.001)**(-1)
        D_12 = self.eyes * D_list_12
        A = fluid.layers.reshape(paddle.tensor.bmm(A, D_12), (b, c, h, w))
        return A

    def forward(self, x0):
        learn_A = paddle.tile(self.DecoupleA, (1, self.out_channels // self.groups, 1, 1))
        norm_learn_A = fluid.layers.concat([self.norm(learn_A[0:1, :, :, :]), self.norm(learn_A[1:2, :, :, :]), self.norm(learn_A[2:3, :, :, :])], 0)
        x = einsum('nctw,cd->ndtw', x0, self.Linear_weight)
        x = x + self.Linear_bias
        x = self.bn0(x)

        n, kc, t, v = x.shape
        x = fluid.layers.reshape(x, (n, self.num_subset, kc // self.num_subset, t, v))
        x = einsum('nkctv,kcvw->nctw', x, norm_learn_A)

        x = self.bn(x)
        x += self.down(x0)
        x = self.relu(x)
        return x


class TCN_GCN_unit(nn.Layer):
    def __init__(self, in_channels, out_channels, A, groups, num_point, block_size, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, groups, num_point)
        self.tcn1 = unit_tcn(out_channels, out_channels,
                             stride=stride, num_point=num_point)
        self.relu = nn.ReLU()

        self.A = fluid.dygraph.to_variable(np.sum(np.reshape(A.astype(np.float32), [3, num_point, num_point]), axis=0))
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn_skip(in_channels, out_channels, kernel_size=1, stride=stride)
        self.dropSke = DropBlock_Ske(num_point=num_point)
        self.dropT_skip = DropBlockT_1d(block_size=block_size)

    def forward(self, x, keep_prob):
        x = self.tcn1(self.gcn1(x), keep_prob, self.A) + self.dropT_skip(self.dropSke(self.residual(x), keep_prob, self.A), keep_prob)
        return self.relu(x)


@BACKBONES.register()
class DropGCN(nn.Layer):

    def __init__(self, num_point=25, groups=8, block_size=41, A=np.load("/home/aistudio/work/graph.npy"), **kwargs):
        super(DropGCN, self).__init__()

        self.data_bn = nn.BatchNorm1D(25 * 2)

        self.l1 = TCN_GCN_unit(2, 64, A, groups, num_point, block_size, residual=False,**kwargs)
        self.l2 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size,**kwargs)
        self.l3 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size,**kwargs)
        self.l4 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size,**kwargs)
        self.l5 = TCN_GCN_unit(64, 128, A, groups, num_point, block_size, stride=2,**kwargs)
        self.l6 = TCN_GCN_unit(128, 128, A, groups, num_point, block_size,**kwargs)
        self.l7 = TCN_GCN_unit(128, 128, A, groups, num_point, block_size,**kwargs)
        self.l8 = TCN_GCN_unit(128, 256, A, groups, num_point, block_size, stride=2,**kwargs)
        self.l9 = TCN_GCN_unit(256, 256, A, groups, num_point, block_size,**kwargs)
        self.l10 = TCN_GCN_unit(256, 256, A, groups, num_point, block_size,**kwargs)

        self.pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))

    def forward(self, x):

        N, C, T, V, M = x.shape
        x = fluid.layers.reshape(x.transpose((0, 4, 3, 1, 2)), (N, M * V * C, T))
        if self.data_bn:
            x.stop_gradient = False
        x = self.data_bn(x)
        x = fluid.layers.reshape(fluid.layers.reshape(x, (N, M, V, C, T)).transpose((0, 1, 3, 4, 2)), (N * M, C, T, V))

        x = self.l1(x, 1.0)
        x = self.l2(x, 1.0)
        x = self.l3(x, 1.0)
        x = self.l4(x, 1.0)
        x = self.l5(x, 1.0)
        x = self.l6(x, 1.0)
        x = self.l7(x, 0.9)
        x = self.l8(x, 0.9)
        x = self.l9(x, 0.9)
        x = self.l10(x, 0.9)

        x = self.pool(x)  # NM,C,T,V --> NM,C,1,1
        C = x.shape[1]
        x = paddle.reshape(x, (N, M, C, 1, 1)).mean(axis=1)  # N,C,1,1

        return x

