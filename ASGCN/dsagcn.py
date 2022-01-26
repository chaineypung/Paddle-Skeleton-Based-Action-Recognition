import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import paddle.fluid as fluid
from ..registry import BACKBONES
from paddle.nn.initializer import TruncatedNormal, Constant, Normal

class unit_gcn(nn.Layer):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3, adaptive=True, attention=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = num_subset
        num_jpts = A.shape[-1]

        

        self.conv_d = nn.LayerList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2D(in_channels, out_channels, 1))

        if adaptive:
            # self.PA = fluid.dygraph.Layer.add_parameter(A, fluid.dygraph.to_variable(A.astype(np.float32)))
            self.PA = self.create_parameter(shape=(3, 25, 25), default_initializer=fluid.initializer.NumpyArrayInitializer(A))
      
            # self.PA = fluid.dygraph.Layer.add_parameter(A, fluid.dygraph.to_variable(A.astype(np.float32)))
            # self.alpha = fluid.dygraph.Layer.parameters(paddle.zeros(1))
            self.alpha = self.create_parameter(shape=(1, 1), default_initializer=fluid.initializer.Constant(0.))

            self.conv_a = nn.LayerList()
            self.conv_b = nn.LayerList()
            for i in range(self.num_subset):
                self.conv_a.append(nn.Conv2D(in_channels, inter_channels, 1))
                self.conv_b.append(nn.Conv2D(in_channels, inter_channels, 1))
        # else:
        #     self.A = Variable(fluid.dygraph.to_variable(A.astype(np.float32)), requires_grad=False)
        self.adaptive = adaptive

        if attention:

            self.conv_ta = nn.Conv1D(out_channels, 1, 9, padding=4)


            # s attention
            ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
            pad = (ker_jpt - 1) // 2
            self.conv_sa = nn.Conv1D(out_channels, 1, ker_jpt, padding=pad)


            # channel attention
            rr = 2
            self.fc1c = nn.Linear(out_channels, out_channels // rr)
            self.fc2c = nn.Linear(out_channels // rr, out_channels)


        self.attention = attention

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, 1),
                nn.BatchNorm2D(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2D(out_channels)
        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()


    def forward(self, x):
        N, C, T, V = x.shape

        y = None
        if self.adaptive:
            A = self.PA
            # A = A + self.PA
            for i in range(self.num_subset):
                A1 = fluid.layers.reshape(self.conv_a[i](x).transpose((0, 3, 1, 2)), (N, V, self.inter_c * T))
                # print(A1.shape[-1])
                A2 = fluid.layers.reshape(self.conv_b[i](x), (N, self.inter_c * T, V))
                # print(A2.shape)
                A1 = self.tan(fluid.layers.matmul(A1, A2, transpose_x=False, transpose_y=False, alpha=1.0, name=None)/ A1.shape[-1])  # N V V
                A1 = A[i] + A1 * self.alpha
                # print(self.alpha)
                A2 = fluid.layers.reshape(x, (N, C * T, V))
                z = self.conv_d[i](fluid.layers.reshape(fluid.layers.matmul(A2, A1, transpose_x=False, transpose_y=False, alpha=1.0, name=None), (N, C, T, V)))
                y = z + y if y is not None else z
        # else:
        #     A = self.A.cuda(x.get_device()) * self.mask
        #     for i in range(self.num_subset):
        #         A1 = A[i]
        #         A2 = fluid.layers.reshape(x, (N, C * T, V))
        #         z = self.conv_d[i](fluid.layers.reshape(paddle.matmul(A2, A1), (N, C, T, V)))
        #         y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        if self.attention:
            # spatial attention
            se = y.mean(-2)  # N C V
            se1 = self.sigmoid(self.conv_sa(se))
            y = y * se1.unsqueeze(-2) + y


            # temporal attention
            se = y.mean(-1)
            se1 = self.sigmoid(self.conv_ta(se))
            y = y * se1.unsqueeze(-1) + y


            # channel attention
            se = y.mean(-1).mean(-1)
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y

        return y

# class GCN(nn.Layer):
#     def __init__(self, in_channels, out_channels, vertex_nums=23, stride=1):###################
#         super(GCN, self).__init__()
#         self.conv1 = nn.Conv2D(in_channels=in_channels,
#                                out_channels=3 * out_channels,
#                                kernel_size=1,
#                                stride=1)
#         self.conv2 = nn.Conv2D(in_channels=vertex_nums * 3,
#                                out_channels=vertex_nums,
#                                kernel_size=1)
#
#     def forward(self, x):
#         # x --- N,C,T,V
#         x = self.conv1(x)  # N,3C,T,V
#         N, C, T, V = x.shape
#         x = paddle.reshape(x, [N, C // 3, 3, T, V])  # N,C,3,T,V
#         x = paddle.transpose(x, perm=[0, 1, 2, 4, 3])  # N,C,3,V,T
#         x = paddle.reshape(x, [N, C // 3, 3 * V, T])  # N,C,3V,T
#         x = paddle.transpose(x, perm=[0, 2, 1, 3])  # N,3V,C,T
#         x = self.conv2(x)  # N,V,C,T
#         x = paddle.transpose(x, perm=[0, 2, 3, 1])  # N,C,T,V
#         return x



class Block(paddle.nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 adaptive = True,
                 attention = True,
                 vertex_nums=25,######################
                 temporal_size=9,
                 stride=1,
                 residual=True,
                 ):
        super(Block, self).__init__()
        self.residual = residual
        self.out_channels = out_channels
        # self.dropout = nn.Dropout2D(p=0.2)
        self.bn_res = nn.BatchNorm2D(out_channels)
        self.conv_res = nn.Conv2D(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=1,
                                  stride=(stride, 1))
        self.gcn = unit_gcn(in_channels, out_channels, A, adaptive=adaptive, attention=attention)
        self.tcn = nn.Sequential(
            nn.BatchNorm2D(out_channels),
            nn.ReLU(),
            nn.Conv2D(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=(temporal_size, 1),
                      padding=((temporal_size - 1) // 2, 0),
                      stride=(stride, 1)),
            nn.BatchNorm2D(out_channels),
        )

    def forward(self, x):
        if self.residual:
            y = self.conv_res(x)
            y = self.bn_res(y)
        x = self.gcn(x)
        # x = self.dropout(x)
        x = self.tcn(x)
        out = x + y if self.residual else x
        out = F.relu(out)
        return out


@BACKBONES.register()
class DSAGCN(nn.Layer):
    """
    AGCN model improves the performance of ST-GCN using
    Adaptive Graph Convolutional Networks.
    Args:
        in_channels: int, channels of vertex coordinate. 2 for (x,y), 3 for (x,y,z). Default 2.
    """
    def __init__(self, in_channels=2, A=np.load("/home/aistudio/work/graph.npy"), **kwargs):
        super(DSAGCN, self).__init__()



        self.data_bn = nn.BatchNorm1D(25 * 2)#########################
        self.agcn = nn.Sequential(
            Block(in_channels=in_channels,
                  out_channels=64,
                  A=A,
                  residual=False,
                  **kwargs),
            Block(in_channels=64, out_channels=64, A=A,**kwargs),
            Block(in_channels=64, out_channels=64, A=A,**kwargs),
            Block(in_channels=64, out_channels=64, A=A,**kwargs),
            Block(in_channels=64, out_channels=128,A=A, stride=2, **kwargs),
            Block(in_channels=128, out_channels=128,A=A, **kwargs),
            Block(in_channels=128, out_channels=128, A=A,**kwargs),
            Block(in_channels=128, out_channels=256, A=A,stride=2, **kwargs),
            Block(in_channels=256, out_channels=256,A=A, **kwargs),
            Block(in_channels=256, out_channels=256,A=A, **kwargs))

        self.pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))

    def forward(self, x):
        # data normalization
        N, C, T, V, M = x.shape

        x = x.transpose((0, 4, 1, 2, 3))  # N, M, C, T, V
        x = x.reshape((N * M, C, T, V))

        x = self.agcn(x)

        x = self.pool(x)  # NM,C,T,V --> NM,C,1,1
        C = x.shape[1]
        x = paddle.reshape(x, (N, M, C, 1, 1)).mean(axis=1)  # N,C,1,1

        return x






