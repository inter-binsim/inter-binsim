#!/usr/bin/python3
import sys
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
#from gputracker import MemTracker
sys.path.append('%s/s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
from embedding import EmbedMeanField, EmbedLoopyBP
from pytorch_util import to_scalar
from mlp import MLPClassifier
import copy

from util import cmd_args, S2VGraph
import data
from finetune_data import load_paired_data,FunctionDataset
from utils.function_normalizer import FunctionNormalizer
from utils.instructions_converter import InstructionsConverter
from utils.capstone_disassembler import disassemble
from safetorch.safe_network import SAFE
from safetorch.parameters import Config
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
if torch.cuda.is_available():
    dev=torch.device('cuda')
else:
    dev=torch.device('cpu')
config = Config()
safe = SAFE(config)
state=torch.load('model/safe_model.pth')['state_dict']
safe.load_state_dict(state)
dev=torch.device('cuda')
safe.eval()
safe.to(dev)
converter = InstructionsConverter("model/word2id.json")
normalizer = FunctionNormalizer(max_instruction=150)

class GraphEbd(nn.Module):
    def __init__(self):
        super(GraphEbd, self).__init__()
        if cmd_args.gm == 'mean_field':
            model = EmbedMeanField
        elif cmd_args.gm == 'loopy_bp':
            model = EmbedLoopyBP
        else:
            print('unknown gm %s' % cmd_args.gm)
            sys.exit()
        self.s2v = model(latent_dim=cmd_args.latent_dim,
                        output_dim=cmd_args.out_dim,
                        num_node_feats=cmd_args.feat_dim,
                        num_edge_feats=0,
                        max_lv=cmd_args.max_lv)
        out_dim = cmd_args.out_dim

    def PrepareFeatureLabel(self, batch_graph):
        n_nodes = 0
        concat_feat = []
        for i in range(len(batch_graph)):
            n_nodes += batch_graph[i].num_nodes
            concat_feat.append(batch_graph[i].feat)
        node_feat=torch.cat(concat_feat,0)
        if cmd_args.mode == 'gpu':
            node_feat = node_feat.cuda()

        return node_feat

    def forward(self, batch_graph):
        node_feat = self.PrepareFeatureLabel(batch_graph)
        embed = self.s2v(batch_graph, node_feat, None)

        return embed


import networkx as nx
def create_data(test_datas):   #transfer
    THRESH=800
    for i in tqdm(range(len(test_datas))):
        for o in test_datas[i].keys():
            tt,cfg=test_datas[i][o]
            with torch.no_grad():
                if tt.shape[0]>THRESH:
                    insts=[]
                    tt.cuda()
                    arr=torch.split(tt,THRESH,dim=0)
                    for t in arr:
                        inst1=safe(t.cuda()).cpu()
                        insts.append(inst1)
                    inst=torch.cat(insts,dim=0)
                else:
                    inst=safe(tt.cuda()).cpu()
            g=nx.Graph()
            n=len(cfg.nodes)
            mp={}
            cnt=0
            for j in cfg.nodes:
                g.add_node(cnt)
                mp[j]=cnt
                cnt+=1
            for edge in cfg.edges:
                fro,to=edge
                if fro!=to:
                    g.add_edge(mp[fro],mp[to])  #undirected graph
            test_datas[i][o]=g,inst
            del tt,cfg
    return test_datas
def get_embedding(batch_graph,graphencoder,BATCH=64):
    arr=[]
    for j in range(0,len(batch_graph),BATCH):
        ed=min(len(batch_graph),j+BATCH)
        bg=batch_graph[j:ed]
        with torch.no_grad():
            emb=graphencoder(bg).cpu()
            arr.append(emb)
    embs=torch.cat(arr,0)
    nor=torch.norm(embs,2,dim=1)
    nor=nor.view(-1,1)
    return embs/nor  #normalize
from sklearn.metrics import auc, roc_curve
def check_result():
    batch_size = 1
    with open(f'./finetune_acc/test.pkl','rb') as f: # [IO]
        ebds = pickle.load(f)
    tri_acc = []
    label = []
    score = []
    for func in tqdm(range(len(ebds))):
        if len(ebds[func]) < 2:
            continue
        if 'O1' not in ebds[func].keys():
            continue
        if 'O3' not in ebds[func].keys():
            continue
        for i in range(batch_size):
            #pos
            # opt_rd = random.sample(ebds[func].keys(), 2)
            opt_rd = ['O1', 'O3']
            feature1 = ebds[func][opt_rd[0]]
            feature2 = ebds[func][opt_rd[1]]
            sim1 = F.cosine_similarity(feature1,feature2).cpu().numpy().tolist()[0]
            score.append(sim1)
            label.append(1)
            #neg
            func_rd = func
            while func_rd == func:
                func_rd = random.randint(0, len(ebds)-1)
            opt_neg = random.choice(list(ebds[func_rd].keys()))
            feature3 = ebds[func_rd][opt_neg]
            sim2 = F.cosine_similarity(feature1,feature3).cpu().numpy().tolist()[0]
            score.append(sim2)
            label.append(0)
            tri_acc.append(float(sim1 > sim2))
    fpr, tpr, thresholds = roc_curve(label, score)
    pair_auc = auc(fpr, tpr)
    print(pair_auc)
    print(np.array(tri_acc).mean())
    exit()
if __name__ == '__main__':
    t_dir = 'finetune_acc'
    TEST_BATCH=32
    TRAIN_BATCH=16
    lr=0.001
    test_datas=load_paired_data(converter,normalizer,datapath='/workspace/small_test',filter=None,all_data=True)

    
    # f=open('bardata','rb')
    # train_datas,test_datas=pickle.load(f)
    # f.close()
    print('LOAD BIG ')

    train_list,test_list=[],[]
    test_datas=create_data(test_datas)
    train_graphs,test_graphs=[],[]


    for i in tqdm(range(len(test_datas))):
        for o in test_datas[i].keys():
            g,feat=test_datas[i][o]
            test_datas[i][o]=len(test_list)
            test_list.append(S2VGraph(g,feat))

    for i in range(len(test_datas)):
        if len(test_datas[i])>=2:
            test_graphs.append(test_datas[i])

    # test_loader=DataLoader(FunctionDataset(test_graphs), batch_size=TEST_BATCH, num_workers=10, shuffle=True)
    print('LOAD TRAIN TEST DATA')
    
    with open(f'{t_dir}/test_graphs.pkl', 'wb') as f:
        pickle.dump(test_graphs, f)
    with open(f'{t_dir}/test_list.pkl', 'wb') as f:
        pickle.dump(test_list, f)
    with open(f'{t_dir}/test_datas.pkl', 'wb') as f:
        pickle.dump(test_datas, f)
    with open(f'{t_dir}/test_graphs.pkl', 'rb') as f:
        test_graphs = pickle.load(f)
    with open(f'{t_dir}/test_list.pkl', 'rb') as f:
        test_list = pickle.load(f)
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    graphencoder = GraphEbd()
    if cmd_args.mode == 'gpu':
        graphencoder = graphencoder.cuda()
    # state=torch.load('model/model10.pth')
    state=torch.load('finetune/model5.pth')
    graphencoder.load_state_dict(state['state_dict'])
    ebds = []
    _num = 0
    for i in tqdm(test_graphs):
        proj = ''
        ebd = {}
        for _func, _ in i.items():
            ebd[_func] = get_embedding([test_list[_num]], graphencoder)
            _num += 1
        ebds.append(ebd)

    with open(f'./{t_dir}/test.pkl','wb') as f: # [IO]
        pickle.dump(ebds, f)
    
    check_result()
