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
#from gputracker import MemTracker
sys.path.append('%s/s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
from embedding import EmbedMeanField, EmbedLoopyBP
from pytorch_util import to_scalar
from mlp import MLPClassifier

from util import cmd_args, S2VGraph
import data
from data import load_paired_data,FunctionDataset
from utils.function_normalizer import FunctionNormalizer
from utils.instructions_converter import InstructionsConverter
from utils.capstone_disassembler import disassemble
from safetorch.safe_network import SAFE
from safetorch.parameters import Config
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader

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
if __name__ == '__main__':
    train_datas=load_paired_data(converter,normalizer,datapath='/workspace/small_train',filter=None,all_data=True)
    test_datas=load_paired_data(converter,normalizer,datapath='/workspace/small_test',filter=None,all_data=True)

    TEST_BATCH=32
    TRAIN_BATCH=16
    lr=0.001
    f=open('bardata','rb')
    train_datas,test_datas=pickle.load(f)
    f.close()
    print('LOAD BIG ')

    train_list,test_list=[],[]

    train_datas=create_data(train_datas)
    test_datas=create_data(test_datas)
    train_graphs,test_graphs=[],[]

    for i in tqdm(range(len(train_datas))):
        for o in train_datas[i].keys():
            g,feat=train_datas[i][o]
            train_datas[i][o]=len(train_list)   #id in train_list
            train_list.append(S2VGraph(g,feat))
    for i in range(len(train_datas)):
        if len(train_datas[i])>=2:
            train_graphs.append(train_datas[i])


    for i in tqdm(range(len(test_datas))):
        for o in test_datas[i].keys():
            g,feat=test_datas[i][o]
            test_datas[i][o]=len(test_list)
            test_list.append(S2VGraph(g,feat))

    for i in range(len(test_datas)):
        if len(test_datas[i])>=2:
            test_graphs.append(test_datas[i])


    train_loader=DataLoader(FunctionDataset(train_graphs),batch_size=TRAIN_BATCH, num_workers=10, shuffle=True)
    test_loader=DataLoader(FunctionDataset(test_graphs), batch_size=TEST_BATCH, num_workers=10, shuffle=True)
    print('LOAD TRAIN TEST DATA')

    f=open('bar_datas','wb')
    pickle.dump((train_graphs, test_graphs,train_list,test_list),f)
    f.close()

    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)


    graphencoder = GraphEbd()
    if cmd_args.mode == 'gpu':
        graphencoder = graphencoder.cuda()

    optimizer = optim.Adam(graphencoder.parameters(), lr=lr)
    EPOCH=100
    for e in range(EPOCH):
        for i, (id1,id2,id3) in enumerate(tqdm(train_loader)):
            batch_graph_anchor = [train_list[id1[idx]] for idx in range(len(id1))]
            anchor=graphencoder(batch_graph_anchor)
            batch_graph_pos = [train_list[id2[idx]] for idx in range(len(id2))]
            pos=graphencoder(batch_graph_pos)
            batch_graph_neg = [train_list[id3[idx]] for idx in range(len(id3))]
            neg=graphencoder(batch_graph_neg)
            sim1=F.cosine_similarity(anchor,pos)
            sim2=F.cosine_similarity(anchor,neg)
            loss1=(sim1-1)**2
            loss2=(sim2+1)**2

            loss=loss1.sum()+loss2.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        now_ans=[]
        for i, (id1,id2,id3) in enumerate(tqdm(test_loader)):
            batch_graph_anchor = [test_list[id1[idx]] for idx in range(len(id1))]
            emb1=graphencoder(batch_graph_anchor)
            batch_graph_pos = [test_list[id2[idx]] for idx in range(len(id2))]
            emb2=graphencoder(batch_graph_pos)
            if emb1.shape[0]!=TEST_BATCH:
                continue

            #keep batch size
            for j in range(emb1.shape[0]):
                sim=[]
                for k in range(emb2.shape[0]):
                    sim.append(F.cosine_similarity(emb1[j:j+1],emb2[k:k+1]))
                sim=np.array(sim)
                rk=np.argsort(-sim)
                for k in range(len(rk)):
                    if rk[k]==j:
                        now_ans.append(1/(k+1))
                        break

        mean_mrr=np.array(now_ans).mean()
        print(e,'mean_mrr: ',mean_mrr)
        if e%5==0:
            state = {'state_dict':graphencoder.state_dict(),  'epoch':e}
            torch.save(state,'model/model{}.pth'.format(e))
