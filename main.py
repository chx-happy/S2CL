
import numpy as np
import torch
import pickle
from model import S2CL
from utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor
import pandas as pd
from parsers import args
from tqdm import tqdm
import os
from scipy import sparse
import torch.utils.data as data
from utils import TrnData
import time
from datetime import datetime
import networkx as nx
from scipy.sparse.linalg import eigsh  
from mpl_toolkits.axes_grid1 import make_axes_locatable



import matplotlib
matplotlib.use("Agg")
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes
from scipy.stats import gaussian_kde
import matplotlib as mpl

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["axes.unicode_minus"] = False








os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


d = args.d
l = args.gnn_layer
temp = args.temp
batch_user = args.batch
epoch_no = args.epoch
lambda_1 = args.lambda1
lambda_2 = args.lambda2
lr = args.lr
num_hidden = args.num_hidden
num_heads = args.num_heads
num_layers = args.num_layers
in_drop = args.in_drop
attn_drop = args.attn_drop
negative_slope = args.negative_slope
decay = args.decay
svd_q = args.q
dropout = args.dropout


path = 'data/' + args.data + '/'
with open(path + 'train.pkl', 'rb') as f:
    train = pickle.load(f)
train_csr = (train != 0).astype(np.float32)
with open(path + 'test.pkl', 'rb') as f:
    test = pickle.load(f)
test_np = test.toarray()
print('Data loaded.')
print('user_num:', train.shape[0], 'item_num:', train.shape[1], 'lr:', lr, 'num_heads:', num_heads, 'temp:', temp, 'l:',
      l, 'epoch:', epoch_no)

epoch_user = min(train.shape[0], 30000)


adj_mat = sparse.dok_matrix((train.shape[0] + train.shape[1],
                             train.shape[0] + train.shape[1]), dtype=np.float32)
adj_mat = adj_mat.tolil()
R = sparse.csr_matrix(train).tolil()
adj_mat[:train.shape[0], train.shape[0]:] = R
adj_mat[train.shape[0]:, :train.shape[0]] = R.T
Graph = adj_mat.todok()


rowD = np.array(train.sum(1)).squeeze()
colD = np.array(train.sum(0)).squeeze()
for i in range(len(train.data)):
    train.data[i] = train.data[i] / pow(rowD[train.row[i]] * colD[train.col[i]], 0.5)

train = train.tocoo()
train_data = TrnData(train)
train_loader = data.DataLoader(train_data, batch_size=args.inter_batch, shuffle=True, num_workers=0)

adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
adj_norm = adj_norm.coalesce().to(device)
print('Adj matrix normalized.')


print('Computing structure-aware multi-scale spectral encodings...')
G = nx.Graph()
n_users, n_items = train.shape[0], train.shape[1]
G.add_nodes_from(range(n_users), bipartite='user')
G.add_nodes_from(range(n_users, n_users + n_items), bipartite='item')
for u, v in zip(train.row, train.col):
    G.add_edge(u, n_users + v)
L = nx.normalized_laplacian_matrix(G).astype(np.float32)

m = svd_q + 1
ok = False; last_err = None
while not ok and m >= 2:
    try:
        evals, evecs = eigsh(L, k=m, which='SM')
        ok = True
    except Exception as e:
        last_err = e; m -= 1
if not ok:
    raise RuntimeError(f"eigsh failed for k up to {svd_q+1}: {last_err}")

evals = evals[1:]; evecs = evecs[:, 1:]; m = m - 1
lambda_max = evals.max()
evals = evals / (lambda_max + 1e-6) * 1.0

scale_mode = getattr(args, "scale_mode", "stat")
num_scales = getattr(args, "num_scales", 3)
if scale_mode == "log":
    scales = np.logspace(np.log10(0.1), np.log10(1.0), num=num_scales, base=10.0)
elif scale_mode == "stat":
    mean_lambda = np.mean(evals); std_lambda  = np.std(evals)
    scales = [1.0 / (mean_lambda + 1e-6),
              1.0 / (mean_lambda + std_lambda + 1e-6),
              1.0 / (mean_lambda + 2 * std_lambda + 1e-6)]
    scales = np.clip(scales, 0.05, 10.0)
else:
    scales = [0.25, 1.0, 4.0]
print("Normalized evals range: [%.4f, %.4f]" % (evals.min(), evals.max()))
print(f"Adaptive scales (mode={scale_mode}):", scales)

scaled_feats = []
for t in scales:
    w = np.exp(-t * evals)
    Phi_t = evecs * w[None, :]
    scaled_feats.append(Phi_t)
Phi = np.concatenate(scaled_feats, axis=1)


eps = 1e-6
deg_all = (np.array(G.degree())[:, 1].astype(np.float32) + eps)
deg_u = deg_all[:n_users]; deg_v = deg_all[n_users:]
u_phi = Phi[:n_users, :] / np.sqrt(deg_u)[:, None]
v_phi = Phi[n_users:, :] / np.sqrt(deg_v)[:, None]

def rms_feature_energy(X: np.ndarray) -> float:
    norms_sq = np.sum(X * X, axis=1)
    return float(np.sqrt(np.mean(norms_sq) + eps))

rms_u = rms_feature_energy(u_phi)
rms_v = rms_feature_energy(v_phi)
target_rms = 0.5 * (rms_u + rms_v)
scale_u = (target_rms / (rms_u + eps))
scale_v = (target_rms / (rms_v + eps))
u_features = u_phi * scale_u
v_features = v_phi * scale_v


rms_fig_dir = f"saved_model/vis_{args.data}"
os.makedirs(rms_fig_dir, exist_ok=True)





user_degree = torch.tensor(deg_u, dtype=torch.float32, device=device)
item_degree = torch.tensor(deg_v, dtype=torch.float32, device=device)


u_mul_s = torch.tensor(u_features, dtype=torch.float32, device=device)
v_mul_s = torch.tensor(v_features, dtype=torch.float32, device=device)
svd_u   = torch.tensor(u_features.T, dtype=torch.float32, device=device)
svd_v   = torch.tensor(v_features.T, dtype=torch.float32, device=device)
print('Spectral encodings ready.')


test_labels = [[] for _ in range(test.shape[0])]
for i in range(len(test.data)):
    row = test.row[i]; col = test.col[i]
    test_labels[row].append(col)
print('Test data processed.')


loss_list = []; loss_r_list = []; loss_s_list = []
recall_20_x = []; recall_20_y = []
ndcg_20_y = []; recall_40_y = []; ndcg_40_y = []

model = S2CL(adj_norm.shape[0], adj_norm.shape[1], d, u_mul_s, v_mul_s, svd_u, svd_v,
              train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout,
              num_hidden, num_heads, num_layers, in_drop, attn_drop, negative_slope, Graph, device)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0, lr=lr)

def learning_rate_decay(optimizer):
    for pg in optimizer.param_groups:
        new_lr = pg['lr'] * 0.96
        if new_lr > 0.0005:
            pg['lr'] = new_lr
    return optimizer.param_groups[0]['lr']

current_lr = lr
recall20Max = 0.0; ndcg20Max = 0.0
recall40Max = 0.0; ndcg40Max = 0.0
bestEpoch = 0


cka_hist_E0G, cka_hist_E0F = [], []
knn_hist_E0G, knn_hist_E0F = [], []
VIS_METHOD = getattr(args, "vis_method", "pca")
KNN_K = getattr(args, "vis_knn_k", 20)
KNN_SAMPLE = getattr(args, "vis_knn_sample", 5000)
VIS_EVERY = getattr(args, "vis_every", 0)  


for epoch in range(epoch_no):
    if (epoch + 1) % 50 == 0:
        os.makedirs('saved_model', exist_ok=True)
        torch.save(model.state_dict(), f'saved_model/saved_model_epoch_{epoch}.pt')
        torch.save(optimizer.state_dict(), f'saved_model/saved_optim_epoch_{epoch}.pt')

    current_lr = learning_rate_decay(optimizer)
    epoch_loss = 0.0; epoch_loss_r = 0.0; epoch_loss_s = 0.0
    train_loader.dataset.neg_sampling()

    for _, batch in enumerate(tqdm(train_loader)):
        uids, pos, neg = batch
        uids = uids.long().to(device); pos = pos.long().to(device)
        neg = neg.long().to(device)
        iids = torch.concat([pos, neg], dim=0).long().to(device)

        optimizer.zero_grad()
        loss, loss_r, loss_s = model(uids, iids, pos, neg)
        loss.backward(); optimizer.step()

        epoch_loss   += float(loss.detach().cpu().item())
        epoch_loss_r += float(loss_r.detach().cpu().item())
        epoch_loss_s += float(loss_s.detach().cpu().item())
        torch.cuda.empty_cache()

    batch_no = len(train_loader)
    epoch_loss   /= batch_no
    epoch_loss_r /= batch_no
    epoch_loss_s /= batch_no
    loss_list.append(epoch_loss); loss_r_list.append(epoch_loss_r); loss_s_list.append(epoch_loss_s)
    print('Epoch:', epoch, 'Loss:', epoch_loss, 'Loss_r:', epoch_loss_r, 'Loss_s:', epoch_loss_s)


    test_uids = np.arange(adj_norm.shape[0])
    batch_no = int(np.ceil(len(test_uids) / batch_user))
    all_recall_20 = 0.0; all_ndcg_20 = 0.0
    all_recall_40 = 0.0; all_ndcg_40 = 0.0
    for b in tqdm(range(batch_no)):
        start = b * batch_user; end = min((b + 1) * batch_user, len(test_uids))
        test_uids_input = torch.LongTensor(test_uids[start:end]).to(device)
        predictions = model(test_uids_input, None, None, None, test=True).cpu().numpy()
        recall_20, ndcg_20 = metrics(test_uids[start:end], predictions, 20, test_labels)
        recall_40, ndcg_40 = metrics(test_uids[start:end], predictions, 40, test_labels)
        all_recall_20 += recall_20; all_ndcg_20 += ndcg_20
        all_recall_40 += recall_40; all_ndcg_40 += ndcg_40

    print('-------------------------------------------')
    print('Test of epoch', epoch, ':', 'Recall@20:', all_recall_20 / batch_no, 'Ndcg@20:', all_ndcg_20 / batch_no,
          'Recall@40:', all_recall_40 / batch_no, 'Ndcg@40:', all_ndcg_40 / batch_no)

    if all_recall_20 / batch_no > recall20Max:
        recall20Max = all_recall_20 / batch_no
        ndcg20Max   = all_ndcg_20 / batch_no
        recall40Max = all_recall_40 / batch_no
        ndcg40Max   = all_ndcg_40 / batch_no
        bestEpoch   = epoch

    recall_20_x.append(epoch)
    recall_20_y.append(all_recall_20 / batch_no)
    ndcg_20_y.append(all_ndcg_20 / batch_no)
    recall_40_y.append(all_recall_40 / batch_no)
    ndcg_40_y.append(all_ndcg_40 / batch_no)

    

print('Best epoch : ', bestEpoch, ' , Recall20 : ', recall20Max, ' , NDCG20 : ', ndcg20Max, ' , Recall40 : ',
      recall40Max, ' , NDCG40 : ', ndcg40Max)

metric = pd.DataFrame({
    'epoch': recall_20_x,
    'recall@20': recall_20_y,
    'ndcg@20': ndcg_20_y,
    'recall@40': recall_40_y,
    'ndcg@40': ndcg_40_y
})
best_row = {'epoch': bestEpoch, 'recall@20': recall20Max, 'ndcg@20': ndcg20Max,
            'recall@40': recall40Max, 'ndcg@40': ndcg40Max}
metric = pd.concat([metric, pd.DataFrame([best_row])], ignore_index=True)

os.makedirs('log', exist_ok=True)
os.makedirs('saved_model', exist_ok=True)
current_t = time.gmtime()
metric.to_csv(f'log/result_{args.data}_{time.strftime("%Y-%m-%d-%H-%M-%S", current_t)}.csv', index=False)

torch.save(model.state_dict(), f'saved_model/saved_model_{args.data}_{time.strftime("%Y-%m-%d-%H", current_t)}.pt')
torch.save(optimizer.state_dict(), f'saved_model/saved_optim_{args.data}_{time.strftime("%Y-%m-%d-%H", current_t)}.pt')


print("Saved visualizations to:", save_vis_dir)

