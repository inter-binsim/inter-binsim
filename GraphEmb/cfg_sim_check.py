#!/usr/bin/python3
from cmath import nan
import sys
import os
import torch
import math
import random
import numpy as np
from tqdm import tqdm
import pickle
from grakel.utils import graph_from_networkx
from grakel.kernels import ShortestPath,RandomWalk, WeisfeilerLehmanOptimalAssignment
from grakel import Graph
def Calc(G1, G2):
    edges1=G1.edges()
    edges2=G2.edges()
    node1=G1.nodes()
    node2=G2.nodes()
    gs=list()
    y=list()
    node_label={}
    for x in node1:
        node_label[x]=1
    edge_label={}
    for x in edges1:
        edge_label[x]=1
    gra=Graph(edges1, node_labels=node_label, edge_labels=edge_label)
    gs.append(gra)
    node_label={}
    for x in node2:
        node_label[x]=1
    edge_label={}
    for x in edges2:
        edge_label[x]=1
    gra=Graph(edges2, node_labels=node_label, edge_labels=edge_label)
    gs.append(gra)

    y.append(1)
    y.append(2)
    gk=WeisfeilerLehmanOptimalAssignment(normalize=True)
    t=gk.fit_transform(gs)
    ret=t[0][1]
    return ret
if __name__ == '__main__':
    cfg_sim = {}
    t_dir = "full_test"
    f=open(f'./{t_dir}/bardata_full','rb')
    test_datas = pickle.load(f)
    f.close()
    o_name = ['o0', 'o1', 'o2', 'o3', 'os', 'of']
    for i in tqdm(test_datas):
        for j in i.keys():
            if j[-2:] in o_name:
                funcname = '@'.join(j.split('@')[:2])
                if funcname in cfg_sim.keys():
                    continue
                res = {}
                for o1 in range(len(o_name)):
                    for o2 in range(o1+1, len(o_name)):
                        if funcname + '@' + o_name[o1] in i.keys() and funcname + '@' + o_name[o2] in i.keys():
                            res[o_name[o1] + '_' + o_name[o2]] = Calc(i[funcname+'@'+o_name[o1]][1], i[funcname+'@'+o_name[o2]][1])
                cfg_sim[funcname] = res
                break
    with open('./cfg_sim.pkl', 'wb') as f:
        pickle.dump(cfg_sim, f)
        
        
    