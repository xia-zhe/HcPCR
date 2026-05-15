# Copyright 2023 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import numpy as np
import tensorly as tl
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorly.decomposition import tucker
from tensorly.tenalg import inner as tl_inner

import hyptorch.pmath as hypmath
from hutil import get_children_np2, hype_triplet_losses
from src.fns import (index_points, preprocess_img,
                     shrink_points_beyond_threshold, square_distance)
from src.model.decoder_anchor import DecoderPredictCenters
from src.model.decoder_feature import FeatureAggregator
from src.model.encoder import MCCEncoder

tl.set_backend('pytorch')
class TRL(nn.Module):
    def __init__(self, input_size, ranks, output_size, verbose=1, **kwargs):
        super(TRL, self).__init__(**kwargs)
        self.ranks = list(ranks)
        self.verbose = verbose

        if isinstance(output_size, int):
            self.input_size = [input_size]
        else:
            self.input_size = list(input_size)

        if isinstance(output_size, int):
            self.output_size = [output_size]
        else:
            self.output_size = list(output_size)

        self.n_outputs = int(np.prod(output_size[1:]))

        # Core of the regression tensor weights
        self.core = nn.Parameter(tl.zeros(self.ranks), requires_grad=True)
        #self.core = nn.Parameter(torch.tensor(tl.zeros(self.ranks)), requires_grad=True)
        self.bias = nn.Parameter(tl.zeros(1), requires_grad=True)
        #self.bias = nn.Parameter(torch.tensor(tl.zeros(1)), requires_grad=True)
        weight_size = list(self.input_size[1:]) + list(self.output_size[1:])

        # Add and register the factors
        self.factors = []
        for index, (in_size, rank) in enumerate(zip(weight_size, ranks)):
            #self.factors.append(nn.Parameter(tl.zeros((in_size, rank)), requires_grad=True))
            self.factors.append(nn.Parameter(torch.tensor(tl.zeros((in_size, rank))), requires_grad=True))

            self.register_parameter('factor_{}'.format(index), self.factors[index])

        # FIX THIS
        self.core.data.uniform_(-0.1, 0.1)
        for f in self.factors:
            f.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        tucker_tensor = (self.core, self.factors)
        regression_weights = tl.tucker_to_tensor(tucker_tensor)
        return tl_inner(x, regression_weights, n_modes=tl.ndim(x) - 1) + self.bias

    def penalty(self, order=2):
        penalty = tl.norm(self.core, order)
        for f in self.factors:
            penalty = penalty + tl.norm(f, order)
        return penalty


def tl_inner(x, w, n_modes=2):
    return tl.tenalg.inner(x, w, n_modes=n_modes)

class TRL2(nn.Module):
    def __init__(self, input_size, ranks, output_size, verbose=1):
        super(TRL2, self).__init__()
        self.ranks = list(ranks)
        self.verbose = verbose

        if isinstance(input_size, int):
            input_size = (input_size,)
        if isinstance(output_size, int):
            output_size = (output_size,)

        self.input_size = tuple(input_size)
        self.output_size = tuple(output_size)

        self.n_outputs = int(np.prod(self.output_size))
        weight_size = self.input_size + self.output_size

        self.core = nn.Parameter(tl.zeros(self.ranks), requires_grad=True)

        self.bias = nn.Parameter(torch.zeros(self.output_size[-1]), requires_grad=True)

        self.factors = nn.ParameterList()
        for i, (sz, r) in enumerate(zip(weight_size, self.ranks)):
            factor_i = nn.Parameter(torch.zeros(sz, r), requires_grad=True)
            factor_i.data.uniform_(-0.1, 0.1)
            self.factors.append(factor_i)

        self.core.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        tucker_tensor = (self.core, list(self.factors))
        regression_weights = tl.tucker_to_tensor(tucker_tensor)
        out = tl_inner(x, regression_weights, n_modes=len(self.input_size))
        out = out + self.bias
        return out

    def penalty(self, order=2):
        penalty_val = tl.norm(self.core, order=order)
        for f in self.factors:
            penalty_val = penalty_val + tl.norm(f, order=order)
        return penalty_val

class AttentionPooling(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionPooling, self).__init__()
        self.w = nn.Parameter(torch.randn(feature_dim))
        nn.init.normal_(self.w, mean=0.0, std=0.01)

    def forward(self, x):
        x_tanh = torch.tanh(x)
        alpha = torch.sum(self.w * x_tanh, dim=-1)  # (B, N)
        alpha = F.softmax(alpha, dim=-1)  # (B, N)
        alpha = alpha.unsqueeze(-1)          # (B, N, 1)
        out = torch.sum(x * alpha, dim=1)    # (B, C)
        
        return out


class HcPCRPP(nn.Module):
    """HcPCR++ model with adaptive hyperbolic constraints.
    """
    def __init__(self, args=None):
        super().__init__()

        self.args = args
        self.encoder = MCCEncoder(args=args)
        self.decoderl1 = DecoderPredictCenters(args=args)
        self.decoderl2 = FeatureAggregator(nneigh=args.nneigh, args=args)
        self.fc_out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 1 + 256*3)
        )

        # self.trl_v = TRL(ranks=(400, 1, 1, 300), input_size=(256, 512, 1, 1), output_size=(16, 1 + 256*3))
        # self.trl_a = TRL(ranks=(400, 1, 1, 300), input_size=(256, 512, 1, 1), output_size=(16, 1 + 256*3))

        self.attn_pool = AttentionPooling(1 + 256*3)
        self.attn_pool2 = AttentionPooling(1 + 256*3)

        self.trl_1 = TRL2(input_size=(550, 769), ranks=(300, 769, 769), output_size=(769,)) # rank 最大为(550,769,769)，我不建议设为最大
        self.trl_2 = TRL2(input_size=(400, 769), ranks=(300, 769, 769), output_size=(769,)) 

        self.scale = 0.1
        self.num_curvature = 2
        self.fusenet1 = nn.Linear(in_features=2*(1 + 256*3), out_features=self.num_curvature)
        self.fusenet2 = nn.Sigmoid()

    def repulsive(self, points):
        pts = points.clone()
        k = min(16+1, pts.shape[1]) # plus itself
        dists = square_distance(pts, pts)
        sort_dist, sort_idx = dists.sort()
        knn_dist = sort_dist[:,:,:k] # 1, n, k
        knn_dist = torch.clamp(knn_dist, min=0.001)
        knn_idx = sort_idx[:,:,:k] # 1, n, k

        knn_points = index_points(pts, knn_idx) # 1, n, k, 3
        d = pts[:, :, None] - knn_points # 1, n, k, 3

        const = 0.001
        repulsive = d / (knn_dist[...,None])**2 # 1, n, k, 3 #^2
        repulsive = torch.sum(repulsive, dim=2) * const # 1, n, 3
        repulsive = torch.clamp(repulsive, min=-0.03, max=0.03)

        return repulsive



    def move_points(self, points, seen_xyz, valid_seen_xyz, fea, up_grid_fea, n_iter=2):
        points.requires_grad = True

        for i in range(n_iter):
            pred = self.decoderl2(points, seen_xyz, valid_seen_xyz, fea, up_grid_fea)
            pred = self.fc_out(pred)

            pred_udf = pred[:,:,0]

            grad_x = torch.autograd.grad(
            outputs=pred_udf.sum(),  
            inputs=points,
            retain_graph=True,
            create_graph=False
        )[0]

            gradient = grad_x.detach()
            points = points.detach() # 1, n, 3
            pred_udf = pred_udf.detach()

            points = points - F.normalize(gradient, dim=2) * pred_udf.unsqueeze(-1).repeat(1, 1, 3)
            points = points.detach()

            # repulsive force
            if self.args.repulsive==1:
                points += self.repulsive(points)

            points.requires_grad = True

        return points
        
    def forward(self, seen_images, seen_xyz, query_xyz, valid_seen_xyz, seen_xyz_hr=None, valid_seen_xyz_hr=None):
        
        # print(seen_images.shape, seen_xyz.shape, query_xyz.shape, valid_seen_xyz.shape)
        query_xyz = shrink_points_beyond_threshold(query_xyz, self.args.shrink_threshold)

        seen_images_hr = None
        if seen_xyz_hr != None:
            seen_images_hr = preprocess_img(seen_images.clone(), res=self.args.xyz_size)
            seen_xyz_hr = shrink_points_beyond_threshold(seen_xyz_hr, self.args.shrink_threshold)

        seen_images = preprocess_img(seen_images)
        seen_xyz = shrink_points_beyond_threshold(seen_xyz, self.args.shrink_threshold)

        with torch.cuda.amp.autocast():
            latent, up_grid_fea = self.encoder(seen_images, seen_xyz, valid_seen_xyz, up_grid_bypass=seen_images_hr)
        fea = self.decoderl1(latent)

        if seen_xyz_hr == None:
            net = self.decoderl2(query_xyz, seen_xyz, valid_seen_xyz, fea, up_grid_fea)
        else:
            net = self.decoderl2(query_xyz, valid_seen_xyz_hr, fea, up_grid_fea)
            
        out = self.fc_out(net)  # [16, 550, 769]

        # pred_udf = F.relu(out[:,:,:1]).reshape((-1, 1)) # nQ, 1
        # pred_udf = torch.clamp(pred_udf, max=0.5) 
        # t = self.args.udf_threshold
        # pos = (pred_udf < t).squeeze(-1) # (nQ, )
        # points = query_xyz.reshape((-1, 3)) # (nQ, 3)
        # points = points[pos].unsqueeze(0) # (1, n, 3)
        # points = points.reshape((query_xyz.shape[0],query_xyz.shape[1],3))
        points = self.move_points(query_xyz.clone(), seen_xyz, valid_seen_xyz, fea, up_grid_fea, n_iter=self.args.udf_n_iter//2)
        points.requires_grad = False  

        # pred_udf = F.relu(out[:,:,:1])
        # pred_udf = torch.clamp(pred_udf, max=0.5) 
        # t = self.args.udf_threshold
        # pos = (pred_udf < t)
        # points = query_xyz.clone()
        # points = points[pos.repeat(1, 1, 3)]
        # points = self.move_points(points, seen_xyz, valid_seen_xyz, fea, up_gr                id_fea, n_iter=self.args.udf_n_iter//2)
        points2 = points.permute(0, 2, 1)
        mar, points_patial, out_p_patial = get_children_np2(points2, out_f=out.clone(), starting=points2.shape[-1]-1, kmin = 300, kmax = 400) 
        # mar, points_patial, out_p_patial = get_children_np2(points2, out_f=out.clone(), starting=points2.shape[-1]-1, kmin = points2.shape[-1]//2, kmax = points2.shape[-1]) 

####(B,N1,C) out (B,N2,C) out_f_partial 
###################################################
        # out_f = out.permute(0, 2, 1)
        # out_f = F.adaptive_max_pool1d(out_f, 1).view(out_f.shape[0], -1)

        # out_f_patial = out_p_patial.permute(0, 2, 1)
        # out_f_patial = F.adaptive_max_pool1d(out_f_patial, 1).view(out_f_patial.shape[0], -1)

        # B = out_f.shape[0]
        # out_f = out_f.view(-1, out_f.shape[-1])
        # out_f_patial = out_f_patial.view(-1, out_f_patial.shape[-1])

        # out_f = self.attn_pool(out)
        # out_f_patial = self.attn_pool2(out_p_patial)
    
        out_f = self.trl_1(out)
        out_f_patial = self.trl_2(out_p_patial)

#(B,C) (B,C)
#############################################
#(B,C) (B,C)
        curvature = self.fusenet1(torch.cat((out_f, out_f_patial), 1))
        curvature = self.scale * self.fusenet2(curvature)
        l1 = 0
        l2 = 0
        for i in range(self.num_curvature):
            # self.w = hypmath.logmap0(hypmath.project(out, c=curvature[:, i].unsqueeze(-1)),
            #                                    c=curvature[:, i].unsqueeze(-1))
            self.w = hypmath.project(out_f, c=curvature[:, i].unsqueeze(-1))
            # self.p = hypmath.logmap0(hypmath.project(out_patial, c=curvature[:, i].unsqueeze(-1)),
            #                                    c=curvature[:, i].unsqueeze(-1))
            self.p = hypmath.project(out_f_patial, c=curvature[:, i].unsqueeze(-1))

            pn, posn, pd, nd, t_loss, h_loss = hype_triplet_losses(self.w, self.p, hier_margin=mar, contr_margin=4, ball_dim = 256)
            l1 = l1 + t_loss
            l2 = l2 + h_loss

        l1 = l1 / self.num_curvature
        l2 = l2 / self.num_curvature
        # mar, points_patial, _, _, _ = get_children_np(points2, starting=points2.shape[-1]-1, kmin = points2.shape[-1]//2, kmax = points2.shape[-1]) 
        # points_patial = out_patial.permute(0, 2, 1)
        
        # out = F.relu(net)
        # out = self.manifold.expmap0(out)
        # out = self.emb(out)


        # pout = out.permute(0, 2, 1)
        # pout = F.adaptive_max_pool1d(pout, 1).view(pout.shape[0], -1)
        # pout = self.manifold.expmap0(pout)
        # pout = self.emb(pout)

        # out = self.relu(net)
        # out = self.manifold.expmap0(out)
        # out = self.emb(out)

        # print(fea['anchors_xyz'].size(),fea['enc_feats'].size(),fea['global_feats'].size(),fea['anchors_feats'].size())
        # print(net.shape,out.shape)
        # assert 1==2

        # out = self.proj(out)
        # out = self.manifold.expmap0(out)
        # mu = self.emb(out)


        #print(x.shape)
        # xp = x.clone().permute(0, 2, 1)
        # idk = knn(xp,kn)
        # pos_x = xp[:,:,:kn]

        # for id in range(batch_size):
        #        starting_point = random.randint(0,1023)
        #        pos_x[id,:,:] = xp[id,:,idk[id, starting_point ,:]]

        # print(pos_x.shape)
        # pos_x = pos_x.permute(0, 2, 1)

        # pos_x, x = F.adaptive_max_pool1d(pos_x, 1).squeeze(dim=-1), F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        # pos_x, x = self.proj(pos_x), self.proj(x)
        # pos_x, x = self.manifold.expmap0(pos_x), self.manifold.expmap0(x)
        # child_mu, mu = self.emb(pos_x), self.emb(x)


        return out, fea['anchors_xyz'], l1, l2


# net = HcPCRPP()
# model_key = net.state_dict().keys()
# for key in model_key:
#     print(key)
