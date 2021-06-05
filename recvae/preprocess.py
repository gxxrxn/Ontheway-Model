import pandas as pd
import numpy as np
import json
import os

def load_test_data(data):
    data = list(data.values())
    places = np.array([x['name'] for x in list(data)])
    feature = ['place','uid','rating']
    data = dict()
    data[0] = [x for x in places]
    data = pd.DataFrame(data)
    data['uid']=0
    data['rating']=1
    data.columns = feature
    place2id_ = json.loads(open('recvae/datasets/pre_data/place2id.json').read())[0]
    for i in range(len(data)):
        data.place[i] = place2id_[data.place[i]]
    return data

def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

def filter_triplets(tp, min_uc=5, min_sc=0):
    # Only keep the triplets for items which were clicked on by at least min_sc users.
    if min_sc > 0:
        itemcount = get_count(tp, 'place')
        tp = tp[tp['place'].isin(itemcount.index[itemcount >= min_sc])]

    # Only keep the triplets for users who clicked on at least min_uc items
    # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
    if min_uc > 0:
        usercount = get_count(tp, 'uid')
        tp = tp[tp['uid'].isin(usercount.index[usercount['size'] >= min_uc])]

    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(tp, 'uid'), get_count(tp, 'place')
    return tp, usercount, itemcount

def pre_unique_sid(data_dir):
    sid = dict()
    cnt=0
    with open(os.path.join(data_dir, 'unique_sid.txt'), 'r') as f:
        for line in f:
            sid[int(line.strip())]=cnt
            cnt+=1
    return sid

def numerize(tp, show2id, profile2id):
    sid = list(map(lambda x: show2id[x], tp['place']))
    uid = list(map(lambda x: profile2id[x], tp['uid']))
    value = list(map(lambda x: x, tp['rating']))
    return pd.DataFrame(data={'uid': uid, 'sid': sid, 'value': value}, columns=['uid', 'sid', 'value'])

def go(data=None,data_dir="recvae/datasets/raw_data",mode='test'):
    use_table = 'tripadvisor'    # 'tripadvisor' or 'review'
    min_user_count = 5

    pro_dir = 'recvae/datasets/test'#'./datasets/pre_data'
    raw_data = load_test_data(data)

    raw_data, user_activity, item_popularity = filter_triplets(raw_data, min_uc=min_user_count, min_sc=0)
    sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])

    print("필터링 후, 방문기록 : %d개 | 사용자: %d | 관광지 : %d (sparsity: %.3f%%)" %
          (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))

    unique_uid = user_activity.index
    n_heldout_users = 0

    n_users = unique_uid.size

    tr_users = unique_uid[:(n_users - n_heldout_users)]
    vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
    te_users = unique_uid[(n_users - n_heldout_users):]

    train_plays = raw_data.loc[raw_data['uid'].isin(tr_users)]

    unique_sid = pd.unique(train_plays['place'])

    show2id = pre_unique_sid('recvae/datasets/pre_data')
    unique_sid = show2id.values()
    id2place = json.loads(open('recvae/datasets/pre_data/id2place.json').read())

    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
    
    train_plays = train_plays[train_plays['place'].isin(show2id.keys())]
    print(train_plays)
    
    train_data = numerize(train_plays,show2id, profile2id)

    return unique_sid, unique_uid, train_data

    # if not os.path.exists(pro_dir):
    #     os.makedirs(pro_dir)

    # # unique_sid, unique_uid 리스트 형태로 반환하도록 리팩토링
    # with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
    #     for sid in unique_sid:
    #         f.write('%s\n' % sid)

    # with open(os.path.join(pro_dir, 'unique_uid.txt'), 'w') as f:
    #     for uid in unique_uid:
    #         f.write('%s\n' % uid)

    # train_plays = train_plays[train_plays['place'].isin(show2id.keys())]
    # print(train_plays)
    
    # train_data = numerize(train_plays,show2id, profile2id)
    # train_data.to_csv(os.path.join('recvae/datasets/test', 'train.csv'), index=False)
