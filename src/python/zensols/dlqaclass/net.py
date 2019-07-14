import logging
import torch
from torch import nn
import torch.nn.functional as F
from zensols.dltools import ConvolutionLayerFactory
from zensols.dlqaclass import Features

logger = logging.getLogger(__name__)


class Net(nn.Module):
    def __init__(self, para_shape, ques_shape, para_f_shape, ques_f_shape,
                 n_labels, cnf, debug=False):
        super(Net, self).__init__()
        self.debug = debug

        bilinear_out = cnf.bilinear_out
        feat_out = None#1.0
        fch_prop = 1.0
        self.add_common = True
        self.conv_dropout = True
        self.batch_norm = False
        self.conv_relu = False

        pc = ConvolutionLayerFactory(
            *para_shape, D=1, K=cnf.filter_depth,
            F=(para_shape[0], cnf.filter_width), S=cnf.convolution_stride, P=1)
        qc = ConvolutionLayerFactory(
            *ques_shape, D=1, K=cnf.filter_depth,
            F=(ques_shape[0], cnf.filter_width), S=cnf.convolution_stride, P=1)

        pc_flat = pc.flatten()
        qc_flat = qc.flatten()
        pc_pool = pc_flat.clone(F=(1, cnf.max_pool_fz), S=cnf.max_pool_stride)
        qc_pool = qc_flat.clone(F=(1, cnf.max_pool_fz), S=cnf.max_pool_stride)

        self.para_conv1 = pc.conv1d()
        self.ques_conv1 = qc.conv1d()

        if self.batch_norm:
            self.para_batch = pc.batch_norm2d()
            self.ques_batch = qc.batch_norm2d()

        self.para_pool = pc_pool.max_pool1d()
        self.ques_pool = qc_pool.max_pool1d()

        bl_shape = (pc_pool.flatten_dim, qc_pool.flatten_dim)
        self.match = torch.nn.Bilinear(*bl_shape, bilinear_out)

        join_shape = bl_shape[0] + bl_shape[1] + bilinear_out
        self.add_feats = (feat_out is not None)
        if self.add_feats:
            pc = ConvolutionLayerFactory(
                *para_f_shape, D=1, K=cnf.filter_depth,
                F=(para_f_shape[0], cnf.filter_width),
                S=cnf.convolution_stride, P=1)
            qc = ConvolutionLayerFactory(
                *ques_f_shape, D=1, K=cnf.filter_depth,
                F=(ques_f_shape[0], cnf.filter_width),
                S=cnf.convolution_stride, P=1)

            pc_flat = pc.flatten()
            qc_flat = qc.flatten()
            pc_pool = pc_flat.clone(
                F=(1, cnf.max_pool_fz), S=cnf.max_pool_stride)
            qc_pool = qc_flat.clone(
                F=(1, cnf.max_pool_fz), S=cnf.max_pool_stride)

            self.para_feat_conv1 = pc.conv1d()
            self.ques_feat_conv1 = qc.conv1d()
            self.para_feat_pool = pc_pool.max_pool1d()
            self.ques_feat_pool = qc_pool.max_pool1d()

            bl_shape = (pc_pool.flatten_dim, qc_pool.flatten_dim)
            self.feat_match = torch.nn.Bilinear(*bl_shape, bilinear_out)

            join_shape += pc_pool.flatten_dim + qc_pool.flatten_dim + 1

        if self.add_common:
            join_shape += Features.COMMON_FEATURES_SHAPE

        self.add_hidden = (fch_prop is not None)
        if self.add_hidden:
            n_param = int(fch_prop * join_shape)
            self.fc_join = nn.Linear(join_shape, n_param)
            self.fch = nn.Linear(n_param, n_labels)
        else:
            self.fc_join = nn.Linear(join_shape, n_labels)
        self.join_shape = join_shape

        self.dropout = nn.Dropout(cnf.dropout)

    def forward(self, mats):
        batch_size = mats[0].shape[0]
        if self.debug:
            logger.debug(f'batch_size: {batch_size}')

        para_x, ques_x = map(lambda x: x[:, None, :], mats[:2])
        self._shape_debug('semantic', para_x, ques_x)

        if self.add_feats:
            #para_f_x, ques_f_x = mats[2:]
            para_f_x, ques_f_x = map(lambda x: x[:, None, :], mats[2:])
            self._shape_debug('feats', para_f_x, ques_f_x)

        para_x = self.para_conv1(para_x)
        ques_x = self.ques_conv1(ques_x)
        self._shape_debug('conv', para_x, ques_x)

        if self.conv_relu:
            para_x = F.relu(para_x)
            ques_x = F.relu(ques_x)

        if self.batch_norm:
            para_x = self.para_batch(para_x)
            ques_x = self.ques_batch(ques_x)
            self._shape_debug('batch norm', para_x, ques_x)

        para_x = para_x.view(batch_size, 1, -1)
        ques_x = ques_x.view(batch_size, 1, -1)
        self._shape_debug('flatten', para_x, ques_x)

        para_x = self.para_pool(para_x)
        ques_x = self.ques_pool(ques_x)
        self._shape_debug('pool', para_x, ques_x)

        if self.conv_dropout:
            para_x = self.dropout(para_x)
            ques_x = self.dropout(ques_x)

        x_sim = self.match(para_x, ques_x)
        if self.debug:
            logger.debug(f'x_sim: {x_sim.shape}')

        join_mats = [para_x, ques_x, x_sim]

        if self.add_feats:
            para_f_x = self.para_feat_conv1(para_f_x)
            ques_f_x = self.ques_feat_conv1(ques_f_x)
            self._shape_debug('conv feat', para_f_x, ques_f_x)

            para_f_x = para_f_x.view(batch_size, 1, -1)
            ques_f_x = ques_f_x.view(batch_size, 1, -1)
            self._shape_debug('flatten feat', para_f_x, ques_f_x)

            para_f_x = self.para_feat_pool(para_f_x)
            ques_f_x = self.ques_feat_pool(ques_f_x)
            self._shape_debug('pool feat', para_f_x, ques_f_x)

            if self.conv_dropout:
                para_f_x = self.dropout(para_f_x)
                ques_f_x = self.dropout(ques_f_x)

            f_x_sim = self.match(para_x, ques_x)
            if self.debug:
                logger.debug(f'f_x_sim: {f_x_sim.shape}')

            join_mats.extend((para_f_x, ques_f_x, f_x_sim))

        if self.add_common:
            common_x = mats[4][:, None, :]
            if self.debug:
                logger.debug(f'common_x: {common_x.shape}')
            join_mats.append(common_x)

        join = torch.cat(join_mats, 2)

        if self.debug:
            logger.debug(f'join in: {join.shape} -> {self.join_shape}')

        x = join
        x = self.fc_join(x)
        logger.debug(f'join out: {x.shape}')
        if self.add_hidden:
            x = self.dropout(x)
            x = F.relu(x)
            x = self.fch(x)
            if self.debug:
                logger.debug(f'hidden out: {x.shape}')
        x = x.squeeze(1)
        if self.debug:
            logger.debug(f'squeeze: {x.shape}')

        x = F.log_softmax(x, dim=1)
        if self.debug:
            logger.debug('-' * 30)
            logger.debug(f'out: {x.shape}')

        if self.debug:
            raise ValueError('network disabled in debug mode')

        return x

    def _shape_debug(self, msg, para_x, ques_x):
        if self.debug:
            logger.debug(f'{msg}: para: {para_x.shape}, ques: {ques_x.shape}')
