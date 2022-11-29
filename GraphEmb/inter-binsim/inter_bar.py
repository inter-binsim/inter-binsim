#!/usr/bin/python3
import sys
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
import torch.autograd as autograd
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
import time
from safetorch.safe_network import SAFE
from inter_data import load_paired_data,FunctionDataset
from utils.function_normalizer import FunctionNormalizer
from utils.instructions_converter import InstructionsConverter
from utils.capstone_disassembler import disassemble
from safetorch.parameters import Config
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
config = Config()
safe = SAFE(config)
state=torch.load('model/safe_model.pth')['state_dict']
safe.load_state_dict(state)
dev=torch.device('cuda')
safe.eval()
safe.train()
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
        #x_grad = autograd.grad(embed.mean(), node_feat, retain_graph=True)[0]
    
        return embed


import networkx as nx

inst_dict = {}
def create_data(test_datas, att_pair):   #transfer
    THRESH=800
    with open(att_pair,'rb') as f:
        func_list = pickle.load(f)
    b_list = []#source attack func list
    a_list = []# target func list
    for items  in func_list:#only shuffle ins and blocks of attack source 
        b_list.append(items[0]) 
        a_list.append(items[1])
        a_list.append(items[2])
    for i in tqdm(range(len(test_datas))):
        for o in test_datas[i].keys():# o is the function name
            
            if o in b_list:# only shuffle the attack func
                flag = True
            else:
                flag = False
                if not o in a_list:#shrink the func pair to be represent to save memory
                    tt,cfg=test_datas[i][o]
                    with torch.no_grad():
                        inst, ins=safe(tt.cuda().long(), flag)#.cpu()#inst denotes blocks,  ins denotes the embedding of instructions
                        inst = inst.cpu()# avoid memory out
                        ins = ins.cpu()
                    g=nx.Graph()# create undirected graph
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
                    if i in inst_dict:
                        inst_dict[i].update({o: ins})
                    else:
                        inst_dict.update({i : {o: ins}})
                    continue
            
            tt,cfg=test_datas[i][o]
            #flag = False
            #with torch.no_grad():# requiring grads might induce memory out
            if tt.shape[0]>THRESH:
                insts=[]
                tt.cuda()
                arr=torch.split(tt,THRESH,dim=0)
                for t in arr:
                    inst1=safe(t.cuda().long(), flag).cpu()
                    insts.append(inst1)
                inst=torch.cat(insts,dim=0)
            else:
                inst, ins=safe(tt.cuda().long(), flag)#.cpu()#inst denotes blocks,  ins denotes the embedding of instructions
                # inst_grad = autograd.grad(inst.norm(1), ins, retain_graph=True)[0]#.norm(n_num, 1)
                # print("inst_grad:", inst_grad)
                inst = inst.cpu()# avoid memory out
                #ins = ins.cpu()
                #time.sleep(5)
            # inst denotes basic blocks
            g=nx.Graph()# create undirected graph
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
            flag = False
            if flag:# modify cfg
                '''#random delete blocks
                node_list = random.sample (range(0, n), 2)
                node_list = sorted(node_list)
                i_ = 0
                for item in node_list:# remove nodes in graph to acquire new cfg
                    g.remove_node(item) 
                    inst = torch.cat((inst[:item-i_], inst[item+1-i_:]))
                    ins = torch.cat((ins[:item-i_], ins[item+1-i_:]))
                    i_+=1
                c_ = 0
                for item in  g.nodes:# rename nodes
                    mapping = {item: c_}
                    g = nx.relabel_nodes(g, mapping)   
                    c_ += 1
                '''

                #'''
                # random delete edges
                e_list=random.sample(g.edges, 3)
                #print('e l:', e_list)
                for item in e_list:
                    #print('tem:', item)
                    g.remove_edge(item[0], item[1])
                #'''

            test_datas[i][o]=g,inst


            
            # safe ins for calculating gradients
            if i in inst_dict:
                inst_dict[i].update({o: ins})
            else:
                inst_dict.update({i : {o: ins}})
                #thedict.update({key_a:{key_b: val}})
            

            del tt,cfg
    return test_datas
def get_embedding(batch_graph,graphencoder,BATCH=64):
    arr=[]
    for j in range(0,len(batch_graph),BATCH):
        ed=min(len(batch_graph),j+BATCH)
        bg=batch_graph[j:ed]
        #with torch.no_grad():
        emb=graphencoder(bg).cpu()
        arr.append(emb)
    embs=torch.cat(arr,0)
    nor=torch.norm(embs,2,dim=1)
    nor=nor.view(-1,1)
    return embs/nor  #normalize

def cmp(x):
    return x[-1]
def label_local_ins(name_, emb_a, cos2):
    #name = 'blockdev-wipe@die@mofpp@getLineEnding'
    name = 'feature/'+name_
    print('name:', name)
    with open(name,'rb') as f:
        l_ins = pickle.load(f)
        num = 0
        c_ = 0
        for item in l_ins:
            #print('item:', item)
            emb = item[-1]
            cos1 = F.cosine_similarity(emb, emb_a)
            flag = 0
            if cos1>cos2:
                c_+=1
                flag =1
            l_ins[num].insert(-1, flag)
            num+=1
        print('succ rate:', c_/num)
        with open('feature-train/'+name_,'wb') as k: # [IO]
            pickle.dump(l_ins, k)
    return 
        

if __name__ == '__main__':
    graphencoder = GraphEbd()
    if cmd_args.mode == 'gpu':
        graphencoder = graphencoder.cuda()
    state=torch.load('model/model10.pth')
    graphencoder.load_state_dict(state['state_dict'])
    
    
    att_pair = './list3/success_change_01_614.pkl'# record all attack successful pair, to list in arg
    #att_pair = './success_change_01.pkl'
    test_datas=load_paired_data(converter,normalizer,att_pair,safe, graphencoder,datapath='/workspace/full_test',filter=None,all_data=True)
    
    TEST_BATCH=32
    TRAIN_BATCH=16
    lr=0.001
    f=open('bardata_full','wb')
    pickle.dump(test_datas,f)
    f.close()
    # with open('bardata_full','rb') as f:
    #     test_datas = pickle.load(f)
    print('LOAD BIG ')

    
    train_list,test_list=[],[]
    test_datas=create_data(test_datas, att_pair)
    train_graphs,test_graphs=[],[]

    for i in tqdm(range(len(test_datas))):
        for o in test_datas[i].keys():
            g,feat=test_datas[i][o]
            test_datas[i][o]=len(test_list)
            test_list.append(S2VGraph(g,feat))
            #inst_dict[i][o]
    
    for i in range(len(test_datas)):
        test_graphs.append(test_datas[i])
    test_loader=DataLoader(FunctionDataset(test_graphs), batch_size=TEST_BATCH, num_workers=10, shuffle=True)
    print('LOAD TRAIN TEST DATA')

    test_loader=DataLoader(FunctionDataset(test_graphs), batch_size=TEST_BATCH, num_workers=10, shuffle=True)
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    
    
    
    ebds = {}
    _num = 0
    ebd_grad = {}
    ebd_basic = {}
    ebd_ins = {}
    c_ = 0 
    for i in test_graphs:
        proj = ''
        ebd = {}
        for _proj, _ in i.items():
            proj = '@'.join(_proj.split('@')[:2])
            break
        ebd['proj'] = proj
        ebd['funcname'] = proj
        #print("i:", i)
        #k_, v_=inst_dict[_num].items()
        #print("key:", inst_dict[_num].keys())
        
        for _func, _ in i.items():
            ebd[_func] = get_embedding([test_list[_num]], graphencoder)
            ebd_grad[_func] = get_embedding([test_list[_num]], graphencoder)
            ebd_basic[_func] = test_list[_num].feat
            ebd_ins[_func] = inst_dict[c_][_func]
            _num += 1
        c_ +=1
        ebds[proj] = ebd
    
    with open(att_pair,'rb') as f:
        func_list = pickle.load(f)

        
    num = 0
    crr = 0
    n_num = 1
    #s_f = open('no_shuffle.txt', 'w+')
    s_f = open('shuffle.txt', 'w+')
    #s_f = open('modify_cfg.txt', 'w+')
    for items  in func_list:
        a_fea = ebd_grad[items[0]]
        t_fea = ebd_grad[items[1]]
        t2_fea = ebd_grad[items[2]]
        cos1 = F.cosine_similarity(a_fea,t_fea)
        cos2 = F.cosine_similarity(t_fea,t2_fea)
        
        label_local_ins(items[0], t_fea, cos2)
        
        print('func name:', items[0], items[1], items[2], file=s_f, flush=True)
        print('cos1 & cos2:', cos1, cos2, file=s_f, flush=True)

        #'''
        b_grad = autograd.grad(cos1, ebd_basic[items[0]], retain_graph=True)[0].norm(n_num, 1)
        a0_grad = autograd.grad(cos1, ebd_basic[items[1]], retain_graph=True)[0].norm(n_num, 1)

        # cal gradients for ins
        # b_inst = autograd.grad(cos1, ebd_ins[items[0]], retain_graph=True)[0].norm(n_num, 2)
        # a0_inst = autograd.grad(cos1, ebd_ins[items[1]], retain_graph=True)[0].norm(n_num, 2)
        
        #dif_a = (b_grad - a0_grad).norm(1, 0)
        print('b_grad:', b_grad, file=s_f, flush=True)
        print('a0_grad:', a0_grad, file=s_f, flush=True)
        print('---------------', file=s_f, flush=True)

        
        num +=1
        if cos1 > cos2:
            crr +=1
    print('rate:', crr / num, num)

    



    cos1 = F.cosine_similarity(b_grad, a0_grad)
    cos2 = F.cosine_similarity(a0_grad, a1_grad)

    print("cos1 cos2:", cos1, cos2, a1_grad.shape)
