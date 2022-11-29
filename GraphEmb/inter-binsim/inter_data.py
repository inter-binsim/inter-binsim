from random import randint
import sys
sys.path.append('../')
from playdata import DatasetBase as  DatasetBase
import networkx
import os
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
import pickle
import argparse
import capstone
import re
import torch
import random
import numpy as np
from util import S2VGraph
from copy import deepcopy
import math
from grakel.utils import graph_from_networkx
from grakel.kernels import ShortestPath,RandomWalk, WeisfeilerLehmanOptimalAssignment
from grakel import Graph
def normalize(opcode):
    opcode = opcode.replace(' - ', ' + ')
    opcode = re.sub(r'0x[0-9a-f]+', 'CONST', opcode)
    opcode = re.sub(r'\*[0-9]', '*CONST', opcode)
    opcode = re.sub(r' [0-9]', ' CONST', opcode)
    return opcode
def filter_memory_references(i):
    inst = "" + i.mnemonic
    for op in i.operands:
        if op.type == 1:
            #print(i.op_str,i.reg_name(op.reg))
            inst = inst + " " + i.reg_name(op.reg)
        elif op.type == 2:
            imm = int(op.imm)
            if -int(5000) <= imm <= int(5000):
                inst = inst + " " + str(hex(op.imm))
            else:
                inst = inst + " " + str("HIMM")
        elif op.type == 3:
            mem = op.mem
            if mem.base == 0:
                r = "[" + "MEM" + "]"
                #print(i.op_str,r)
            else:
                r = (
                    "["
                    + str(i.reg_name(mem.base))
                    + "*"
                    + str(mem.scale)
                    + "+"
                    + str(mem.disp)
                    + "]"
                )
               # print(i.op_str,r)
                inst = inst + " " + r
        if len(i.operands) > 1:
            inst = inst + ","
    if "," in inst:
        inst = inst[:-1]
    inst = inst.replace(" ", "_")
    return str(inst)

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
    #gs.append(Graph(edges1,node1))
    #gs.append(Graph(edges2,node2))
    #gs.append(graph_from_networkx(G1))
    #gs.append(graph_from_networkx(G2))
    y.append(1)
    y.append(2)
    gk=WeisfeilerLehmanOptimalAssignment(normalize=True)
    t=gk.fit_transform(gs)
    ret=t[0][1]
    return ret


import time
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.count = 0
        
    def MergeSort(self, arr):
        if len(arr) <= 1:
            return arr
        mid = int(len(arr)/2)
        left = self.MergeSort(arr[:mid])
        right = self.MergeSort(arr[mid:])
        l,r = 0,0
        res = []
        while l<len(left) and r<len(right):
            if left[l] <= right[r]:
                res.append(left[l])
                l += 1
            elif right[r] < left[l]:
                res.append(right[r])
                self.count += (len(left) - l)
                r += 1
        res += left[l:]
        res += right[r:]
        return res
    
    def InversePairs(self, data):
        # write code here
        self.MergeSort(data)
        return self.count%1000000007

def features_stas(ins, cfg, ins_v):
    n_call=0# count number of call
    calls = ['call', 'jal', 'jalr']

    n_trans=0# count number of transfer
    trans = ['jmp', 'jz', 'jnz', 'js', 'je', 'jne', 'jg', 'jle', 'jge', 'ja', 'jnc', 'call']
    
    n_logic = 0 # count number of logic
    logic = {'add', 'sub', 'div', 'imul', 'idiv', 'mul', 'shl', 'dec', 'inc'}

    n_others = 0
    for i in ins:
        t = i[2:]
        pos = t.find('_')
        opc = t[:pos]
        if opc in calls:
            n_call+=1
        elif opc in trans:
            n_trans+=1
        elif opc in logic:
            n_logic+=1
        else:
            n_others+=1
    s = Solution() #cal the num of reverse pairs 
    n_r = s.InversePairs(ins_v)
    return [n_call, n_trans, n_logic, n_others, n_r]
    

def add_ins(i, o, ins_num, ins_corpus, cap, converter, normalizer, per = 0.1):#generate local instances for the given input
    
    n_ins=0# count number of ins
    n_call=0# count number of call
    n_trans=0# count number of transfer
    n_logic = 0 # count number of logic
    
    fea_list = [0]*5
    tt=[]
    # func_addr, asm_list, rawbytes_list, cfg, bai_featrue=ret_func_data[opt]
    func_addr, asm_list, rawbytes_list, cfg, bai_featrue=i[2][o]
    cfg =deepcopy(cfg)
    # percentage ins to be added
    maximum =round(ins_num * per) #if int(ins_num * per) > len(cfg.nodes) else len(cfg.nodes)#the number of ins to add
    num= random.choice(range(1, maximum+1))
    l_n = []# record the num of ins to insert in each block 
    for j in range(len(cfg.nodes)):
        if num<=0:
            tmp = 0 
        else:
            tmp = random.choice(range(0, num))
            if j == (len(cfg.nodes)-1):# make sure all ins has been added
                tmp = num
        num = num - tmp
        l_n.append(tmp)
    c_ = 0
    for bb in cfg.nodes:
        instructions = []
        ra=cfg.nodes[bb]['raw']
        for code in cap.disasm(ra,bb):
            instructions.append("X_" + filter_memory_references(code))
        num_ins = len(instructions)
        l_a = 0
        ins_c = random.sample(ins_corpus, l_n[c_]) 
        if l_n[c_] > num_ins:# if ins length is higher than num ins
            l_a = l_n[c_] - num_ins
            l_n[c_] = num_ins
        index_ = random.sample(range(0, num_ins+1), l_n[c_])# select some positions to insert ins
        index_.sort()
        k = 0
        for j in index_:
            instructions.insert(j+k, ins_c[k])
            k+=1
        c_+=1
        instructions = instructions + ins_c[ k : k+l_a ]
        converted_instructions = converter.convert_to_ids(instructions)
        
        instructions_, length = normalizer.normalize_functions([converted_instructions])
        tt.append(instructions_[0])
        list_ = features_stas(instructions, cfg, converted_instructions)#acquire feature vector
        for k in range(len(list_)):
            fea_list[k] += list_[k]
    tt=torch.IntTensor(tt)
    return [tt, cfg, fea_list]

def delete_ins(i, o, ins_num, ins_corpus, cap, converter, normalizer, per = 0.1):#generate local instances for the given input
    tt=[]
    fea_list = [0]*5
    # func_addr, asm_list, rawbytes_list, cfg, bai_featrue=ret_func_data[opt]
    func_addr, asm_list, rawbytes_list, cfg, bai_featrue=i[2][o]
    cfg =deepcopy(cfg) 
    maximum =round(ins_num * per) #if int(ins_num * per) > len(cfg.nodes) else len(cfg.nodes)#the number of ins to add
    num= random.choice(range(1, maximum+1))
    l_n = []# record the num of ins to insert in each block 
    
    index_del = []
    for bb in cfg.nodes:# the block has little ins cannot be selected to delete ins
        ins = cfg.nodes[bb]['asm']
        if len(ins) <= maximum:
            continue 
        index_del.append(bb)
    if len(index_del) == 0:
        for bb in cfg.nodes:
            instructions = []
            ra=cfg.nodes[bb]['raw']
            for code in cap.disasm(ra,bb):
                instructions.append("X_" + filter_memory_references(code))
            converted_instructions = converter.convert_to_ids(instructions)
            instructions_, length = normalizer.normalize_functions([converted_instructions])
            tt.append(instructions_[0])
            list_ = features_stas(instructions, cfg, converted_instructions)#acquire feature vector
            for k in range(len(list_)):
                fea_list[k] += list_[k]
        tt=torch.IntTensor(tt)
        return [tt, cfg, fea_list]
    
    l_n = []# record the num of ins to insert in each block 
    for j in range(len(index_del)):
        if num<=0:
            tmp = 0 
        else:
            tmp = random.choice(range(0, num))
            if j == (len(index_del)-1):# make sure all ins has been added
                tmp = num
        num = num - tmp
        l_n.append(tmp)
    c_ = 0
    for bb in cfg.nodes:
        instructions = []
        ra=cfg.nodes[bb]['raw']
        for code in cap.disasm(ra,bb):
            instructions.append("X_" + filter_memory_references(code))
        num_ins = len(instructions)
        if bb in index_del:
            if num_ins < l_n[c_]:
                l_n[c_] = num_ins-1
            index_ = random.sample(range(0, num_ins), l_n[c_])# select some positions to insert ins
            index_.sort()
            k = 0
            for j in index_:
                if j-k<0:
                    continue
                del instructions[j-k]
                k+=1
            c_+=1
        converted_instructions = converter.convert_to_ids(instructions)
        instructions_, length = normalizer.normalize_functions([converted_instructions])
        
        tt.append(instructions_[0])
        list_ = features_stas(instructions, cfg, converted_instructions)#acquire feature vector
        for k in range(len(list_)):
            fea_list[k] += list_[k]
    tt=torch.IntTensor(tt)
    return [tt, cfg, fea_list]

def shuffle_ins(i, o, ins_num, ins_corpus, cap, converter, normalizer, per = 0.1):#generate local instances for the given input
    func_addr, asm_list, rawbytes_list, cfg, bai_featrue=i[2][o]
    cfg =deepcopy(cfg)
    tt = []   
    fea_list = [0]*5 
    dict_ra = {}
    index_list = []


    for bb in cfg.nodes:
        for i in range(len(cfg.nodes[bb]['asm'])):
            index_list.append((bb, i))
        
        ins = []
        ra=cfg.nodes[bb]['raw']
        for code in cap.disasm(ra, bb):
            ins.append("X_" + filter_memory_references(code))
        dict_ra[bb]=ins
    
    maximum = math.ceil(ins_num * per) #if int(ins_num * per) > len(cfg.nodes) else len(cfg.nodes)#the number of ins to add
    if maximum <2:
        maximum+=1
    r_n= random.choice(range(2, maximum+1))

    shuffle_ins = random.sample (index_list, r_n)
    s_cp = shuffle_ins.copy()
    np.random.shuffle(s_cp)
    cfg_ins = deepcopy(cfg) # create this cfg for deliver ins
    dict_ra_cp = deepcopy(dict_ra)
    for i in range( len(shuffle_ins) ):
        bb = shuffle_ins[i][0]
        index = shuffle_ins[i][1]

        bb_cp = s_cp[i][0]
        index_cp = s_cp[i][1]
        cfg.nodes[bb]['asm'][index] = cfg_ins.nodes[bb_cp]['asm'][index_cp] 
        dict_ra[bb][index] = dict_ra_cp[bb_cp][index_cp]


    for bb in cfg.nodes:
        #instructions = []
        ins = dict_ra[bb]
        converted_instructions = converter.convert_to_ids(ins)
        list_ = features_stas(ins, cfg, converted_instructions)#acquire feature vector
        for k in range(len(list_)):
            fea_list[k] += list_[k]
        instructions_, length = normalizer.normalize_functions([converted_instructions])
        #print('instructions_:', instructions_[0])
        tt.append(instructions_[0])

    #tt = ins
    tt=torch.IntTensor(tt)
    #print('fea:', fea_list)
    return [tt, cfg, fea_list]


def delete_block(i, o, ins_num, ins_corpus, cap, converter, normalizer, per = 0.1):#generate local instances for the given input
    func_addr, asm_list, rawbytes_list, cfg, bai_featrue=i[2][o]
    cfg_ori = cfg
    cfg =deepcopy(cfg)
    fea_list = [0]*5
    tt = []   
    if len(cfg.nodes) <= 2:
        for bb in cfg.nodes:
            instructions = []
            ra=cfg.nodes[bb]['raw']
            for code in cap.disasm(ra,bb):
                instructions.append("X_" + filter_memory_references(code))
            converted_instructions = converter.convert_to_ids(instructions)
            instructions_, length = normalizer.normalize_functions([converted_instructions])
            ins_num += len(instructions)
            tt.append(instructions_[0])
            list_ = features_stas(instructions, cfg, converted_instructions)#acquire feature vector
            for k in range(len(list_)):
                fea_list[k] += list_[k]
        tt=torch.IntTensor(tt)
        return [tt, cfg, fea_list]


    maximum =math.ceil(len(cfg.nodes) * per) #if int(ins_num * per) > len(cfg.nodes) else len(cfg.nodes)#the number of ins to add
    r_n = random.choice(range(1, maximum+1))
    r_n = r_n if r_n < len(cfg.nodes) else len(cfg.nodes)-1
    block_to_del = random.sample(cfg.nodes, r_n)
    for bb in block_to_del:
        cfg.remove_node(bb)
    if len(cfg.edges) == 0:# avoid the number of edge to be ZERO.
        list = cfg.nodes
        two_nodes = random.sample(list, 2)
        cfg.add_edge(two_nodes[0], two_nodes[1])
    

    for bb in cfg.nodes:
        instructions = []
        ra=cfg.nodes[bb]['raw']
        for code in cap.disasm(ra,bb):
            instructions.append("X_" + filter_memory_references(code))
        converted_instructions = converter.convert_to_ids(instructions)
        instructions_, length = normalizer.normalize_functions([converted_instructions])
        ins_num += len(instructions)
        tt.append(instructions_[0])
        list_ = features_stas(instructions, cfg, converted_instructions)#acquire feature vector
        for k in range(len(list_)):
            fea_list[k] += list_[k]
    #print('fea_list:', fea_list)
    tt=torch.IntTensor(tt)
    return [tt, cfg, fea_list]
def delete_edge (i, o, ins_num, ins_corpus, cap, converter, normalizer, per = 3):#generate local instances for the given input
    func_addr, asm_list, rawbytes_list, cfg, bai_featrue=i[2][o]
    cfg_ori=cfg
    cfg =deepcopy(cfg)
    fea_list = [0]*5
    tt = []   
    
    maximum =math.ceil(len(cfg.edges) * per) #if int(ins_num * per) > len(cfg.nodes) else len(cfg.nodes)#the number of ins to add
    r_n = random.choice(range(1, maximum+1))
    r_n = r_n if r_n < len(cfg.edges) else len(cfg.edges)-1
    
    e_list=random.sample(cfg.edges, r_n)


    for item in e_list:
        #print('tem:', item)
        cfg.remove_edge(item[0], item[1])
    for bb in cfg.nodes:
        instructions = []
        ra=cfg.nodes[bb]['raw']
        for code in cap.disasm(ra,bb):
            instructions.append("X_" + filter_memory_references(code))
        converted_instructions = converter.convert_to_ids(instructions)
        instructions_, length = normalizer.normalize_functions([converted_instructions])
        ins_num += len(instructions)
        tt.append(instructions_[0])
        list_ = features_stas(instructions, cfg, converted_instructions)#acquire feature vector
        for k in range(len(list_)):
            fea_list[k] += list_[k]
    tt=torch.IntTensor(tt)
    # print('start search:')
    # edit_distance = nx.graph_edit_distance(cfg, cfg_ori)
    # print('edit_di:', edit_distance)
    return [tt, cfg, fea_list]
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


def Get_Emb(tt, cfg, S2VGraph, safe, graphencoder):
    THRESH=800
    
    with torch.no_grad():
        if tt.shape[0]>THRESH:
            insts=[]
            tt.cuda()
            arr=torch.split(tt,THRESH,dim=0)
            for t in arr:
                inst1, _=safe(t.cuda().long(), False)
                insts.append(inst1)
            inst=torch.cat(insts,dim=0)
        else:
            inst, _=safe(tt.cuda().long(), False)
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
    ebd = get_embedding([S2VGraph(g, inst)], graphencoder)
    return ebd
    
def generate_local_ins(cfg, i, o, ins_num, ins_corpus, cap, converter, normalizer, safe, graphencoder, max_iter = 1000):#generate local instances for the given input
    # delete_ins(i, o, ins_num, ins_corpus, cap, converter, normalizer, per = 0.1)
    # shuffle_ins(i, o, ins_num, ins_corpus, cap, converter, normalizer)
    # shuffle_block(i, o, ins_num, ins_corpus, cap, converter, normalizer)
    # delete_block(i, o, ins_num, ins_corpus, cap, converter, normalizer)#generate local instances for the given input
    # delete_edge (i, o, ins_num, ins_corpus, cap, converter, normalizer)#generate local instances for the given input
    loc_list = []
    for j in range (max_iter):# the number of local instances
        choice = random.randint(0, 4)
        #choice = 4
        if choice == 0:
            t = add_ins(i, o, ins_num, ins_corpus, cap, converter, normalizer, per = 0.1)
        elif choice == 1:
            t = delete_ins(i, o, ins_num, ins_corpus, cap, converter, normalizer, per = 0.1)
        elif choice == 2:
            t = shuffle_ins(i, o, ins_num, ins_corpus, cap, converter, normalizer, per = 0.1)
        elif choice == 3:
            t = delete_block(i, o, ins_num, ins_corpus, cap, converter, normalizer, per = 0.1)#generate local instances for the given input
        elif choice == 4:
            t = delete_edge (i, o, ins_num, ins_corpus, cap, converter, normalizer, per = 0.1)#generate local instances for the given input
        ebd = Get_Emb(t[0], t[1], S2VGraph, safe, graphencoder)
        node_len = len(t[1].nodes)
        edge_len = len(t[1].edges)
        graph_sim = Calc(t[1], cfg)
        t[2].append(node_len)
        t[2].append(edge_len)
        t[2].append(graph_sim)
        t.append(ebd)
        loc_list.append(t)
        #print('lc list:', loc_list)
    with open('feature/'+o,'wb') as f: # [IO]
        pickle.dump(loc_list, f)
    #sys.exit()
    # to do: add create data here.

def establish_inscorpus(dataset, cap):
    ins_corpus = []
    for i in dataset.get_paired_data_iter():  #proj, func_name, func_addr, asm_list, rawbytes_list, cfg, bai_featrue
        
        if i[2] == 0:
            continue

        for o in i[2].keys():
            ins_num = 0 # count the number of ins for current func
            tt=[]
            # func_addr, asm_list, rawbytes_list, cfg, bai_featrue=ret_func_data[opt]
            func_addr, asm_list, rawbytes_list, cfg, bai_featrue=i[2][o]
            
            for bb in cfg.nodes:
                
                instructions = []
                ra=cfg.nodes[bb]['raw']
                for code in cap.disasm(ra,bb):
                    instructions.append("X_" + filter_memory_references(code))
                ins_corpus += instructions
                if len(ins_corpus) > 2000:
                    return ins_corpus
    return ins_corpus

def load_paired_data(converter,normalizer, att_pair, safe, graphencoder, datapath='../binsimdata/extract',filter=None,all_data=True):
    dataset = DatasetBase(datapath, att_pair, filter, all_data)
    cap=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64)
    cap.detail = True
    functions=[]

    ins_corpus = establish_inscorpus(dataset, cap)# create an ins corpus to save ins
    with open(att_pair,'rb') as f:
        func_list = pickle.load(f)
        b_list = []#source attack func list
        a_list = []# target func list
    for items  in func_list:#only shuffle ins and blocks of attack source 
        b_list.append(items[0]) 
        a_list.append(items[1])
        a_list.append(items[2])
    # for proj, func_name, ret_func_data in tqdm(dataset.get_paired_data_iter()):
    for i in dataset.get_paired_data_iter():  #proj, func_name, func_addr, asm_list, rawbytes_list, cfg, bai_featrue
        
        if i[2] == 0:
            continue
        functions.append({})
        for o in i[2].keys():
            ins_num = 0 # count the number of ins for current func
            tt=[]
            # func_addr, asm_list, rawbytes_list, cfg, bai_featrue=ret_func_data[opt]
            func_addr, asm_list, rawbytes_list, cfg, bai_featrue=i[2][o]
            
            for bb in cfg.nodes:
                instructions = []
                ra=cfg.nodes[bb]['raw']
                for code in cap.disasm(ra,bb):
                    instructions.append("X_" + filter_memory_references(code))
                converted_instructions = converter.convert_to_ids(instructions)
                instructions_, length = normalizer.normalize_functions([converted_instructions])
                ins_num += len(instructions)
                tt.append(instructions_[0])
            tt=torch.IntTensor(tt)
            # ebd[opt] = (tt,cfg)
            # functions[proj] = ebd
            functions[-1][o]=(tt,cfg)

             
            
            if o in b_list:# generate local ins
                #pass
                generate_local_ins(cfg, i, o, ins_num, ins_corpus, cap, converter, normalizer, safe, graphencoder)
            #sys.exit()

    return functions

class FunctionDataset(torch.utils.data.Dataset): #binary version dataset
    def __init__(self,datas,opt1=None,opt2=None):  #random visit
        self.datas=datas
        self.opt1=opt1
        self.opt2=opt2
    def __getitem__(self, idx):             #also return bad pair
        pairs=self.datas[idx]
        if self.opt1 is not None:
            f1=pairs[self.opt1]
            f2=pairs[self.opt2]
            return f1,f2
        else:
            pos=random.randint(0,len(pairs)-1)
            pos2=random.randint(0,len(pairs)-1)
            while pos2==pos:
                pos2=random.randint(0,len(pairs)-1)
            f1=pairs[sorted(pairs.keys())[pos]]
            f2=pairs[sorted(pairs.keys())[pos2]]
            ftype=random.randint(0,len(self.datas)-1)
            while ftype==idx:
                ftype=random.randint(0,len(self.datas)-1)
            pair_opp=self.datas[ftype]
            pos3=random.randint(0,len(pair_opp)-1)
            f3=pair_opp[sorted(pair_opp.keys())[pos3]]
            return f1,f2,f3
    def __len__(self):
        return len(self.datas)


