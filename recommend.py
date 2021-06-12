import time
import json
import os
from recvae import runs

def result(data, province_info):
    res = runs.run(data, province_info, mode='test')
    # test = 'python recvae/run.py --dataset "recvae/datasets/test" --mode="test" --topk=100'
    # time.sleep(3)
    # os.system(test)
    # path = 'recvae/results/result_re.json'
    return json.loads(res)

def train():
    rp.preprocessing(data, 'train')
    train = 'python run.py --dataset "datasets/pre_data"'
    time.sleep(30)
    os.system(train)