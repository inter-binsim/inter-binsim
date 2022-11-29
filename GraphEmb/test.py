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
from finetune_data import load_paired_data,FunctionDataset
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
if __name__ == '__main__':
    test_datas=load_paired_data(converter,normalizer,datapath='/workspace/small_test',filter=None,all_data=True)
    import sys
    TEST_BATCH=int(sys.argv[1])
    TRAIN_BATCH=16
    # f=open('bar_datas','rb')
    # train_graphs, test_graphs,train_list,test_list=pickle.load(f)
    # f.close()
    train_list,test_list=[],[]
    test_datas=create_data(test_datas)
    train_graphs,test_graphs=[],[]

    for i in tqdm(range(len(test_datas))):
        for o in test_datas[i].keys():
            g,feat=test_datas[i][o]
            test_datas[i][o]=len(test_list)
            test_list.append(S2VGraph(g,feat))

    for i in range(len(test_datas)):
        test_graphs.append(test_datas[i])
    def test_mrr(opt1,opt2):
        ds=FunctionDataset(test_graphs,opt1,opt2)
        ind=[]
        for i in range(len(ds)):
            if test_graphs[i].get(opt1) and test_graphs[i].get(opt2):
                ind.append(i)
        subds=torch.utils.data.Subset(ds,ind)
        test_loader=DataLoader(subds, batch_size=TEST_BATCH, num_workers=10, shuffle=True)
        print(opt1,opt2)


        random.seed(cmd_args.seed)
        np.random.seed(cmd_args.seed)
        torch.manual_seed(cmd_args.seed)


        graphencoder = GraphEbd()
        if cmd_args.mode == 'gpu':
            graphencoder = graphencoder.cuda()

        state=torch.load('model/model10.pth')
        graphencoder.load_state_dict(state['state_dict'])

        now_ans=[]
        now_ap=[]
        for i, (id1,id2) in enumerate(tqdm(test_loader)):
            if id1.shape[0]!=TEST_BATCH:
                continue
            batch_graph_anchor = [test_list[id1[idx]] for idx in range(len(id1))]
            anchor=get_embedding(batch_graph_anchor,graphencoder)#graphencoder(batch_graph_anchor)
            batch_graph_pos = [test_list[id2[idx]] for idx in range(len(id2))]
            pos=get_embedding(batch_graph_pos,graphencoder)#graphencoder(batch_graph_pos)
            for j in range(len(anchor)):
                vA=anchor[j:j+1]
                sim = np.array(torch.mm(vA, pos.T).cpu().squeeze())
                y=np.argsort(-sim)
                for k in range(len(pos)):
                    if y[k]==j:
                        now_ans.append(1/(k+1))
                        if k+1<=1: #AP1
                            now_ap.append(1)
                        else:
                            now_ap.append(0)
                        break
        mean_mrr=np.array(now_ans).mean()
        mean_ap=np.array(now_ap).mean()
        print('mrr{}:'.format(TEST_BATCH),mean_mrr,'AP1:',mean_ap)
                #keep batch size

    test_mrr('O0','O3')
    test_mrr('O0','Os')
    test_mrr('O1','O3')
    test_mrr('O1','Os')
    test_mrr('O2','O3')
    test_mrr('O2','Os')
