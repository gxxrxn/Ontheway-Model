import numpy as np
import pandas as pd
import math
import json
import random
import os

import torch
from torch import optim
from scipy import sparse

from recvae import preprocess
# from recvae.utils import get_test_data
from recvae.model import VAE

import argparse

os.environ['KMP_DUPLICATE_LIB_OK']='True'
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='"recvae/datasets/test"')
parser.add_argument('--output_dir', type=str, default='recvae/results')
parser.add_argument('--hidden-dim', type=int, default=600) #600
parser.add_argument('--latent-dim', type=int, default=200) #200
parser.add_argument('--batch-size', type=int, default=500) #500
parser.add_argument('--beta', type=float, default=None)
parser.add_argument('--gamma', type=float, default=0.005)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--n-epochs', type=int, default=50)
parser.add_argument('--n-enc_epochs', type=int, default=3)
parser.add_argument('--n-dec_epochs', type=int, default=1)
parser.add_argument('--not-alternating', type=bool, default=False)
parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--topk', type=int, default=100)
args = parser.parse_args()

seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

#device = torch.device("cuda:0")
device = torch.device("cpu")
global model
model = None

class Batch:
    def __init__(self, device, idx, data_in, data_out=None):
        self._device = device
        self._idx = idx
        self._data_in = data_in
        self._data_out = data_out

    def get_idx(self):
        return self._idx

    def get_idx_to_dev(self):
        return torch.LongTensor(self.get_idx()).to(self._device)

    def get_ratings(self, is_out=False):
        data = self._data_out if is_out else self._data_in
        return data[self._idx]

    def get_ratings_to_dev(self, is_out=False):
        return torch.Tensor(
            self.get_ratings(is_out).toarray()
        ).to(self._device)

def generate(batch_size, device, data_in, data_out=None, shuffle=False, samples_perc_per_epoch=1):
    assert 0 < samples_perc_per_epoch <= 1

    total_samples = data_in.shape[0]
    samples_per_epoch = int(total_samples * samples_perc_per_epoch)

    if shuffle:
        idxlist = np.arange(total_samples)
        np.random.shuffle(idxlist)
        idxlist = idxlist[:samples_per_epoch]
    else:
        idxlist = np.arange(samples_per_epoch)

    for st_idx in range(0, samples_per_epoch, batch_size):
        end_idx = min(st_idx + batch_size, samples_per_epoch)
        idx = idxlist[st_idx:end_idx]

        yield Batch(device, idx, data_in, data_out)

def predict(model, data_in):
    model.eval()

    batch = generate(batch_size=data_in.shape[0], device=device, data_in=data_in)
    batch = next(batch)

    ratings_in = batch.get_ratings_to_dev()
    ratings_pred = model(ratings_in, calculate_loss=False).cpu().detach().numpy()
    ratings_pred[batch.get_ratings().nonzero()] = -np.inf

    return ratings_pred

def recommend(data, unique_sid, k=20):
    recs = list()
    for i in data:
        topk = np.argsort(i)[::-1][k:] # top k개의  index를 가져옴
        recommend_list = np.zeros(k)
        for tk in range(k):
            recommend_list[tk] = unique_sid[topk[tk]] # sid로 변환하여 할당
        recs.append(recommend_list)
    return recs

def pos_in(df):
    res = 6371 * math.acos(
            math.cos(math.radians(35.1643694)) 
            * math.cos(math.radians(df.lat)) * math.cos(math.radians(df.lon) - math.radians(128.9317153)) 
            + math.sin(math.radians(35.1643694)) * math.sin(math.radians(df.lat))
        )
    if res > 25:
        return False
    return True

def get_recommend_pid(unique_sid, unique_uid, data, k=100):
     # initialize
    model_kwargs = {
        'hidden_dim': args.hidden_dim,
        'latent_dim': args.latent_dim,
        'input_dim': data.shape[1]
    }

    PATH = os.path.join('recvae/model',os.listdir('recvae/model')[-1])
    global model
    if model == None:
        print('❤️❤️❤️❤️')
        model = VAE(**model_kwargs)
        model.load_state_dict(torch.load(PATH, map_location=device))

    # 데이터 넣기
    pred = predict(model, data)

    uid = dict()
    cnt=0
    for u in unique_uid:
        uid[cnt] = int(u)
        cnt += 1
    
    sid = dict()
    cnt=0
    for s in unique_sid:
        sid[cnt] = int(s)
        cnt += 1

    recs = pd.DataFrame(recommend(pred, sid, k),index=uid)
    id2place = json.loads(open(os.path.join('recvae/datasets/pre_data', 'id2place.json')).read())
    posdata = pd.read_csv(open(os.path.join('db/data', 'pos_info.csv')))

    recs = recs.T

    drop_idx = []
    pname_list = []
    lat_list = []
    lon_list = []

    for i in recs.values:
        pname = id2place[str(int(i[0]))]
        ppos = pd.DataFrame(posdata.loc[posdata['place'] == str(pname)]['pos'])
        if ppos.empty:
            drop_idx.append(recs.loc[np.float64(recs['0']) == i].index)
            continue
        
        ppos = eval(posdata.loc[posdata['place'] == str(pname)].iloc[0]['pos'])
        pname_list.append(pname)
        lat_list.append(ppos[0])
        lon_list.append(ppos[1])

    for i in drop_idx:
        recs.drop(i, inplace=True)

    recs['name'] = pname_list
    recs['lat'] = lat_list
    recs['lon'] = lon_list

    recs.drop(0, axis=1, inplace=True)

    print(recs)

    recs['pos_in'] = recs[['lat', 'lon']].apply(pos_in, axis=1)

    fin_result = pd.DataFrame(recs.loc[recs['pos_in'] == True])
    fin_result.drop('pos_in', axis=1, inplace=True)
    fin_result.reset_index(drop=True, inplace=True)
    fin_result = fin_result.T

    return fin_result.to_json()

def run(data=None, mode='test'):
    unique_sid, unique_uid, tp = preprocess.go(data, mode='test')

    n_items = len(unique_sid)
    n_users = len(unique_uid)
    global_indexing = False

    n_users = n_users if global_indexing else tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    practice_data = sparse.csr_matrix((np.ones_like(rows),
                                (rows, cols)), dtype='float64',
                                shape=(n_users, n_items))
    # practice_data = get_test_data(unique_sid, unique_uid, args.dataset)

    return get_recommend_pid(unique_sid, unique_uid, practice_data, 100)

# if __name__ == "__main__":
#     if args.mode == 'train':
#         train()
#     elif args.mode == 'test':
#         # preprocess
    