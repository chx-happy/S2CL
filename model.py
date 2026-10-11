import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import scipy.sparse as sp
import networkx as nx
from GAT4 import GAT


class S2CL(nn.Module):
    def __init__(self, n_u, n_i, d, u_mul_s, v_mul_s, ut, vt, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout,
                 num_hidden, num_heads, num_layers, in_drop, attn_drop, negative_slope,
                 Graph, device):
        super(S2CL, self).__init__()
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u, d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i, d)))
        self.n_u = n_u
        self.n_i = n_i
        self.train_csr = train_csr
        self.adj_norm = adj_norm
        self.l = l
        self.d = d
        self.E_u_list = [None] * (l + 1)
        self.E_i_list = [None] * (l + 1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        self.Z_u_list = [None] * (l + 1)
        self.Z_i_list = [None] * (l + 1)
        self.G_u_list = [None] * (l + 1)
        self.G_i_list = [None] * (l + 1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0
        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.act = nn.LeakyReLU(0.5)
        self.num_hidden = num_hidden
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.in_drop = in_drop
        self.attn_drop = attn_drop
        self.negative_slope = negative_slope
        self.Graph = Graph
        self.E_u = None
        self.E_i = None
        self.device = device
        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt
        self.F_u_list = [None] * (l + 1)
        self.F_i_list = [None] * (l + 1)
        self.F_u_list[0] = self.E_u_0
        self.F_i_list[0] = self.E_i_0
        self.dropout = dropout
        self.eps = 0.1
        self.noise_coeff = nn.Parameter(torch.ones(1))



    def forward(self, uids, iids, pos, neg, test=False):
        if test == True:  # testing phase
            preds = self.E_u[uids] @ self.E_i.T
            mask = self.train_csr[uids.cpu().numpy()].toarray()
            mask = torch.Tensor(mask).cuda(torch.device(self.device))
            preds = preds * (1 - mask) - 1e8 * mask
            predictions = preds.argsort(descending=True)
            return predictions

        else:  # training phase
            
            heads = ([self.num_heads] * self.num_layers)
            g = dgl.from_scipy(self.Graph)
            if torch.cuda.is_available():
                cuda = True
                g = g.int().to(self.device)
            else:
                cuda = False
           
            g = dgl.remove_self_loop(g)
            g = dgl.add_self_loop(g)

            ma = GAT(g,
                     self.num_layers,
                     self.d,
                     self.num_hidden,
                     heads,
                     F.elu,
                     self.in_drop,
                     self.attn_drop,
                     self.negative_slope)
            for layer in range(1, self.l + 1):
                
                self.Z_u_list[layer] = torch.spmm(self.adj_norm, self.E_i_list[layer - 1])
                self.Z_i_list[layer] = torch.spmm(self.adj_norm.transpose(0, 1), self.E_u_list[layer - 1])

               

                vt_ei = self.vt @ self.E_i_list[layer - 1]
                self.F_u_list[layer] = (self.u_mul_s @ vt_ei)
                ut_eu = self.ut @ self.E_u_list[layer - 1]
                self.F_i_list[layer] = (self.v_mul_s @ ut_eu)
                
                if cuda:
                    ma.cuda()
                    self.E_u_list[layer - 1] = self.E_u_list[layer - 1].cuda()
                    self.E_i_list[layer - 1] = self.E_i_list[layer - 1].cuda()


                ls = torch.cat((self.E_u_list[layer - 1], self.E_i_list[layer - 1]), dim=0)
                ls1 = ma(ls)

                self.G_u_list[layer], self.G_i_list[layer] = torch.split(ls1, [self.n_u, self.n_i], dim=0)

                self.E_u_list[layer] = self.Z_u_list[layer] + self.E_u_list[layer - 1]
                self.E_i_list[layer] = self.Z_i_list[layer] + self.E_i_list[layer - 1]
    

           
            self.G_u = sum(self.G_u_list)
            self.G_i = sum(self.G_i_list)
            self.E_u = sum(self.E_u_list)
            self.E_i = sum(self.E_i_list)
            self.F_u = sum(self.F_u_list)
            self.F_i = sum(self.F_i_list)

            # cl loss
            G_u_norm = self.G_u
            E_u_norm = self.E_u
            G_i_norm = self.G_i
            E_i_norm = self.E_i
            F_u_norm = self.F_u
            F_i_norm = self.F_i

            neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
            neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
            pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp, -5.0, 5.0)).mean() + \
                        (torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp, -5.0, 5.0)).mean()

            neg_score1 = torch.log(torch.exp(F_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
            neg_score1 += torch.log(torch.exp(F_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
            pos_score1 = (torch.clamp((F_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp, -5.0, 5.0)).mean() + \
                         (torch.clamp((F_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp, -5.0, 5.0)).mean()

            loss_s = -pos_score + neg_score - pos_score1 + neg_score1
            

           
            u_emb = self.E_u[uids]
            pos_emb = self.E_i[pos]
            neg_emb = self.E_i[neg]
            pos_scores = (u_emb * pos_emb).sum(-1)
            neg_scores = (u_emb * neg_emb).sum(-1)
            loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()

         
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()
            loss_reg *= self.lambda_2

           
            loss = loss_r + self.lambda_1 * loss_s + loss_reg
            return loss, loss_r, self.lambda_1 * loss_s
    1
    def get_embeddings(self):
            """
           
            """
            def safe(x):
                return x.detach().cpu() if x is not None else None

            out = {
                "E_u": safe(getattr(self, "E_u", None)),
                "E_i": safe(getattr(self, "E_i", None)),
                "G_u": safe(getattr(self, "G_u", None)),
                "G_i": safe(getattr(self, "G_i", None)),
                "F_u": safe(getattr(self, "F_u", None)),
                "F_i": safe(getattr(self, "F_i", None)),
               
                # "E_u_list": [safe(x) for x in getattr(self, "E_u_list", [])],
                # "E_i_list": [safe(x) for x in getattr(self, "E_i_list", [])],
               
                "E0_u": safe(self.E_u_0),
                "E0_i": safe(self.E_i_0),
                }
            return out    
