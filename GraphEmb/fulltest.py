#!/usr/bin/python3
from cmath import nan
import sys
import os
import torch
import math
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
from statistics import mean
from torch.utils.data import DataLoader
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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

def cmp(x):
    return x[-1]
def check_result():
    ta_opt = 'o0'
    t_dir = 'full_test'
    with open(f'./{t_dir}/result_sorted.pkl', 'rb') as f: # [IO]
        result = pickle.load(f)
    with open(f'./{t_dir}/cfg_nodes.pkl', 'rb') as f: # [IO]
        cfg_nodes = pickle.load(f)
    with open(f'./{t_dir}/pre_cfg_nodes.pkl', 'rb') as f: # [IO]
        pre_cfg_nodes = pickle.load(f)
    with open('./cfg_sim.pkl', 'rb') as f:
        cfg_sim = pickle.load(f)
    t_bbs = []
    tot_func = 0
    len_ = 0
    s_bbs = []
    for k in pre_cfg_nodes.keys():
        for i in pre_cfg_nodes[k].keys():
            if i.split('@')[-1] == ta_opt:
                len_ += 1
            if len(i.split('@')[-1]) == 2 and i.split('@')[-1][-2] == 'o':
                t_bbs.append(pre_cfg_nodes[k][i])
            else:
                tot_func += 1
                s_bbs.append(pre_cfg_nodes[k][i])
    print(len(result))
    print("len:", len_)
    print("avg:", tot_func/len_)
    print("t_bbs:", mean(t_bbs))
    print("s_bbs:", mean(s_bbs))
    total_opt = ['o0', 'o1', 'o2', 'o3', 'os', 'of']
    for t_opt in total_opt:
        with open(f'./{t_dir}/result_sorted.pkl', 'rb') as f: # [IO]
            result = pickle.load(f)
        print(t_opt)
        if t_opt == ta_opt:
            continue
        sum = [0, 0]
        for target in result.keys():
            for line in result[target]:
                if line[0] == target + f'@{t_opt}' and not math.isnan(line[1]):
                    sum[0] += line[1]
                    sum[1] += 1
        sum[0] = sum[0]/sum[1]
        ans = []
        tot_target = len(result)
        print("Before:", tot_target)
        with open(f'./{t_dir}/pre_test.pkl','rb') as f: # [IO]
            ebds = pickle.load(f)
        o_name = ['o0', 'o1', 'o2', 'o3', 'os', 'of', 'origin_gcc', 'sort_gcc']
        for target in result.keys():
            tot = 0
            
            if t_opt + '_' + ta_opt in cfg_sim[target].keys():
                if  cfg_sim[target][t_opt + '_' + ta_opt] == 1:
                    result[target] = []
                    tot_target -= 1
                    # print(target)
                    continue
            if ta_opt + '_' + t_opt in cfg_sim[target].keys():
                if  cfg_sim[target][ta_opt + '_' + t_opt] == 1:
                    result[target] = []
                    tot_target -= 1
                    # print(target)
                    continue
            for i, source in enumerate(result[target]):
                if source[0].split('@')[-1] not in o_name:
                    proj1 = '@'.join(source[0].split('@')[:2])
                    proj2 = target
                    try:
                        feature1 = ebds[proj1][source[0]]
                        feature2 = ebds[proj2][target + f'@{ta_opt}']
                        feature3 = ebds[proj2][target + f'@{t_opt}']
                        if F.cosine_similarity(feature1,feature2).cpu().numpy().tolist()[0] < F.cosine_similarity(feature2,feature3).cpu().numpy().tolist()[0]:
                            tot += 1
                        else:
                            result[target][i] = []
                    except:
                        result[target][i] = []
                        continue
            if tot == 0:
                result[target] = []
                tot_target -= 1
        print("After:", tot_target)
        # print("Not sim_cfg", not_sim_cfg)
        # print("Sim_cfg", sim_cfg)
        at_num = 0
        for target in result.keys():
            tot = []
            if result[target] == []:
                continue
            for source in result[target]:
                if source == []:
                    continue
                if source[0] == target + f'@{t_opt}' and float(source[1]) > 0:
                    # if cfg_nodes[target][target + '@o0'] == cfg_nodes[target][haha[0]]:
                    if len(tot) == 0:
                        continue
                    for i in tot:
                        ans.append([i, target + f'@{ta_opt}', target+f'@{t_opt}'])
                    at_num += 1
                    break
                if source[0].split('@')[-1] not in o_name:
                    tot.append(source[0])
        if tot_target!=0:
            print(at_num, tot_target)
            print("success_rate", at_num/tot_target)
        # print(len(result))
        # print("succ/tot_func:", len(success_change)/tot_func)
    exit()
if __name__ == '__main__':
    # check_result()
    ta_opt = 'o0'
    t_dir = "full_test"
    test_datas=load_paired_data(t_dir, converter,normalizer,datapath=f'/workspace/{t_dir}_extract',filter=None,all_data=True)

    TEST_BATCH=32
    TRAIN_BATCH=16
    lr=0.001
    f=open(f'./{t_dir}/bardata_full','wb')
    pickle.dump(test_datas,f)
    f.close()
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
        test_graphs.append(test_datas[i])
    test_loader=DataLoader(FunctionDataset(test_graphs), batch_size=TEST_BATCH, num_workers=10, shuffle=True)
    print('LOAD TRAIN TEST DATA')

    f=open(f'./{t_dir}/bar_datas_full','wb')
    pickle.dump((test_graphs,test_list),f)
    f.close()
    f=open(f'./{t_dir}/bardata_full','rb')
    test_datas = pickle.load(f)
    f.close()
    f=open(f'./{t_dir}/bar_datas_full','rb')
    test_graphs,test_list=pickle.load(f)
    f.close()
    test_loader=DataLoader(FunctionDataset(test_graphs), batch_size=TEST_BATCH, num_workers=10, shuffle=True)
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    graphencoder = GraphEbd()
    if cmd_args.mode == 'gpu':
        graphencoder = graphencoder.cuda()
    # state=torch.load('model/model10.pth')
    state=torch.load('finetune/model5.pth')
    graphencoder.load_state_dict(state['state_dict'])
    ebds = {}
    _num = 0
    for i in tqdm(test_graphs):
        proj = ''
        ebd = {}
        for _proj, _ in i.items():
            proj = '@'.join(_proj.split('@')[:2])
            break
        ebd['proj'] = proj
        ebd['funcname'] = proj
        for _func, _ in i.items():
            ebd[_func] = get_embedding([test_list[_num]], graphencoder)
            _num += 1
        ebds[proj] = ebd

    with open(f'./{t_dir}/test.pkl','wb') as f: # [IO]
        pickle.dump(ebds, f)
    with open(f'./{t_dir}/test.pkl','rb') as f: # [IO]
        ebds = pickle.load(f)
    # with open(f'cfg_01_sim.pkl', 'rb') as f: # [IO]
    #     ans = pickle.load(f)
    # success_change = []
    # for i in ans:
    #     proj1 = '@'.join(i[0].split('@')[:2])
    #     proj2 = '@'.join(i[1].split('@')[:2])
    #     feature1 = ebds[proj1][i[0]]
    #     feature2 = ebds[proj2][i[1]]
    #     feature3 = ebds[proj2][i[2]]
    #     if F.cosine_similarity(feature1,feature2).cpu().numpy().tolist()[0] < F.cosine_similarity(feature2,feature3).cpu().numpy().tolist()[0]:
    #         success_change.append(i)
    # with open('success_change_01.pkl','wb') as f: # [IO]
    #     pickle.dump(success_change, f)
    # print(success_change)

    result = {}
    o_name = ['o0', 'o1', 'o2', 'o3', 'os', 'of', 'sort_gcc', 'origin_gcc']
    for _, proj in ebds.items():
        for filename, feature1 in proj.items():
            if filename == 'proj' or filename == 'funcname':
                continue
            target = filename.split('@')[-1]
            if target == ta_opt:
                target = '@'.join(filename.split('@')[:-1])
                for o_ in o_name:
                    if o_ == ta_opt:
                        continue
                    if target + '@' + o_ in ebds[target]:
                        feature2 = ebds[target][target + '@' + o_]
                        if result.get(target):
                            result[target].append([f'{target}@{o_}',
                                                   F.cosine_similarity(feature1,feature2).cpu().numpy().tolist()[0]])
                        else:
                            result[target] = [[f'{target}@{o_}',
                                               F.cosine_similarity(feature1,feature2).cpu().numpy().tolist()[0]]]
            else:
                if target in o_name:
                    continue
                target = '@'.join(filename.split('@')[-2:])
                if target not in ebds.keys():
                    continue
                if target + f'@{ta_opt}' in ebds[target]:
                    feature2 = ebds[target][target + f'@{ta_opt}']
                    if result.get(target):
                        result[target].append([filename, F.cosine_similarity(feature1,feature2).cpu().numpy().tolist()[0]])
                    else:
                        result[target] = [[filename, F.cosine_similarity(feature1,feature2).cpu().numpy().tolist()[0]]]

    with open(f'./{t_dir}/result.pkl','wb') as f: # [IO]
        pickle.dump(result, f)
    with open(f'./{t_dir}/result.pkl','rb') as f: # [IO]
        result = pickle.load(f)
    for i in result.keys():
        result[i] = sorted(result[i], key=cmp, reverse=True)
    with open(f'./{t_dir}/result_sorted.pkl', 'wb') as f: # [IO]
        pickle.dump(result, f)


    