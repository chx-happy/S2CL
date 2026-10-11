"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
import dgl.function as fn


class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        
        self.num_scales = 3


        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        for l in range(1, num_layers):
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, False, self.activation))


        self.layer_scorer = nn.Linear(num_hidden, 1, bias=False)   
        self.layer_gate = nn.Parameter(torch.ones(self.num_scales)) 


    def _single_step_smooth(self, h):
        """
        一步简单 GCN-style 平滑:
            h_new(v) = mean_{u in N(v)} h(u)
        """
        with self.g.local_scope():
            self.g.ndata['h'] = h
            self.g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'))
            return self.g.ndata['h']

    def forward(self, inputs):
        """
        inputs: [N, d_in]
        return: [N, num_hidden]
        """


        if self.num_layers == 1:
            
            h0 = inputs
            multi_scale_inputs = [h0]
            h_curr = h0
            for k in range(1, self.num_scales):
                h_curr = self._single_step_smooth(h_curr)  # [N, d_in]
                multi_scale_inputs.append(h_curr)

           
            gat = self.gat_layers[0]
            layer_reps = []
            for k in range(self.num_scales):
                h_k = gat(self.g, multi_scale_inputs[k])   
                G_k = h_k.mean(dim=1)                      
                layer_reps.append(G_k)

            H_all = torch.stack(layer_reps, dim=1)          


            layer_scores = self.layer_scorer(H_all).squeeze(-1)     
            layer_scores = layer_scores * self.layer_gate[:self.num_scales]  
            layer_alpha = torch.softmax(layer_scores, dim=1)        
            G = (layer_alpha.unsqueeze(-1) * H_all).sum(dim=1)      
            return G


        h = inputs
        for l in range(self.num_layers):
            temp = h.flatten(1)
            h = self.gat_layers[l](self.g, temp)  

        G = h.mean(dim=1)  # head 平均
        return G
