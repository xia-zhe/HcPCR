
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""


import numpy as np
import torch
import torch.nn.functional as F
from geoopt.manifolds.stereographic.math import mobius_add, dist
from models.manifolds import PoincareBall
import random

def knn(x, k):
    # [batch, 3, 1024] k
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)    # [1024, 1024]
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_children_np(data, query_rgb_gt=None, labels=None, starting=256, kmin =100, kmax=500):
        # torch.seed(10)
        # data: [batch, 3, 1024]
        # out: [B, 550, 769]
        k = random.randint(kmin,kmax)
        # data[:,:3,:]: [B, 3, 769], k: 499, idknn: [B, 769, 499]
        idknn = knn(data[:,:3,:],k)
        #print(idknn.shape)
        pos_child = data[:,:,:k]
        query_rgb_gt2 = None
        labels1 = None
        labels2 = None
        if query_rgb_gt is not None:
               query_rgb_gt2 = query_rgb_gt[:,:k,:]
               labels1 = torch.squeeze(labels[0][:,:k])
               labels2 = torch.squeeze(labels[1][:,:k])
        for id in range(data.shape[0]):
               starting_point = random.randint(0,starting)  # 随机选一个中心
               pos_child[id,:,:] = data[id,:,idknn[id, starting_point ,:]] # top k 
               if query_rgb_gt is not None:
                      query_rgb_gt2[id,:,:] = query_rgb_gt[id,idknn[id, starting_point ,:],:] # top k 
                      labels1[id,:] = labels[0][id, idknn[id, starting_point ,:]] # top k 
                      labels2[id,:] = labels[1][id, idknn[id, starting_point ,:]] # top k 
                     #  out_child[id,:,:] = out[id,idknn[id, starting_point ,:],:] # top k 
       
        mar = 1000./k
       #  return mar,pos_child, out_child, k-1
        return mar,pos_child, query_rgb_gt2, [labels1, labels2], k-1


def get_children_np2(data, out_f=None, starting=256, kmin =100, kmax=500):
        # torch.seed(10)
        # data: [batch, 3, 1024]
        # out: [B, 550, 769]
        k = random.randint(kmin,kmax)
        # data[:,:3,:]: [B, 3, 769], k: 499, idknn: [B, 769, 499]
        idknn = knn(data[:,:3,:],k)
        #print(idknn.shape)
        pos_child = data[:,:,:k]
        out_f_partial = None
        if out_f is not None:
               out_f_partial = out_f[:,:k,:]
        for id in range(data.shape[0]):
               starting_point = random.randint(0,starting)  # 随机选一个中心
               pos_child[id,:,:] = data[id,:,idknn[id, starting_point ,:]] # top k 
               if out_f is not None:
                      out_f_partial[id,:,:] = out_f[id,idknn[id, starting_point ,:],:] # top k 
       
        mar = 1000./k
       #  return mar,pos_child, out_child, k-1
        return mar,pos_child, out_f_partial

def hype_triplet_losses(parent_mu , pos_child_mu, hier_margin=0.2, contr_margin=4, ball_dim = 256, one_child = False, opposite_hier=False):
       #  parent_mu = parent_mu2.permute(0, 2, 1)
       #  parent_mu = F.adaptive_max_pool1d(parent_mu, 1).view(parent_mu.shape[0], -1)

       #  pos_child_mu = pos_child_mu2.permute(0, 2, 1)
       #  pos_child_mu = F.adaptive_max_pool1d(pos_child_mu, 1).view(pos_child_mu.shape[0], -1)

       #  B = parent_mu.shape[0]
       #  parent_mu = parent_mu.view(-1, parent_mu.shape[-1])
       #  pos_child_mu = pos_child_mu.view(-1, pos_child_mu.shape[-1])
        if one_child:
               neg_child_mu = torch.flip(parent_mu,[0])
        else:
               neg_child_mu = torch.flip(pos_child_mu,[0])


        ball = PoincareBall(c=1.0,dim=ball_dim)
        parent_norm = torch.norm(parent_mu, dim=1)       #dist(parent_mu, parent_mu*0, keepdim=True).pow(2).sum(1)
        pos_norm = torch.norm(pos_child_mu ,dim=1)

        par_norm = ball.dist0(parent_mu)  # [16, 518]   # [16, 518, 769]
        p_norm = ball.dist0(pos_child_mu) # [16, 191]   # [16, 191, 769]   #.pow(2).sum(1)

        distance_positive = ball.dist(parent_mu, pos_child_mu)  #, keepdim=True).pow(2).sum(1)
        distance_negative = ball.dist(parent_mu, neg_child_mu)  #, keepdim=True).pow(2).sum(1)


        distance = distance_positive - distance_negative + contr_margin
        #distance = -distance_positive + distance_negative + mar_dist
        if opposite_hier:
               norm = par_norm-p_norm + hier_margin
        else:
               norm = -par_norm+p_norm + hier_margin            #Standard version wich improves the baseline
        #norm = par_norm-p_norm + margin                        #Possible change: parent becomes the mean of its children --EXperiment 07--

        triplet = torch.mean(torch.max(distance, torch.zeros_like(distance)))                   #F.relu(distance_positive - distance_negative + 0.5 )
        hierarch =torch.mean(torch.max(norm, torch.zeros_like(norm)))                   #F.relu(-par_norm+p_norm + margin)

        return parent_norm.mean(), pos_norm.mean()  ,distance_positive.mean(),distance_negative.mean(),triplet, hierarch

def euc_triplet_losses(parent_mu , pos_child_mu, hier_margin=2, contr_margin=4, one_child = False, opposite_hier=False):

        if one_child:
               neg_child_mu = torch.flip(parent_mu,[0])
        else:
               neg_child_mu = torch.flip(pos_child_mu,[0])

        parent_norm = torch.norm(parent_mu,dim=1)       #dist(parent_mu, parent_mu*0, keepdim=True).pow(2).sum(1)
        pos_norm = torch.norm(pos_child_mu ,dim=1)

        distance_positive = F.pairwise_distance(parent_mu, pos_child_mu)  #, keepdim=True).pow(2).sum(1)
        distance_negative = F.pairwise_distance(parent_mu, neg_child_mu)  #, keepdim=True).pow(2).sum(1)

        distance = distance_positive - distance_negative + contr_margin

        if opposite_hier:
               norm = parent_norm-pos_norm + hier_margin
        else:
               norm = -parent_norm+pos_norm + hier_margin


        triplet = torch.mean(torch.max(distance, torch.zeros_like(distance)))                   #F.relu(distance_positive - distance_negative + 0.5 )
        hierarch =torch.mean(torch.max(norm, torch.zeros_like(norm)))                   #F.relu(-par_norm+p_norm + margin)

        #triplet = triplet_losses.mean()
        #hierarch = hierarch_losses.mean()

        return parent_norm.mean(), pos_norm.mean()  ,distance_positive.mean(),distance_negative.mean(),triplet, hierarch



def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


