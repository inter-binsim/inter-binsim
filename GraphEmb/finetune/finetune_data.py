from random import randint
import sys
sys.path.append('../')
from finetune_playdata import DatasetBase as  DatasetBase
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

import time
def load_paired_data(converter,normalizer,datapath='../binsimdata/extract',filter=None,all_data=True):
    dataset = DatasetBase(datapath, filter, all_data)
    cap=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64)
    cap.detail = True
    functions=[]
    for i in tqdm(dataset.get_paired_data_iter()):  #proj, func_name, func_addr, asm_list, rawbytes_list, cfg, bai_featrue
        functions.append({})
    # for proj, func_name, ret_func_data in tqdm(dataset.get_paired_data_iter()):
    #     ebd = {}
    #     ebd['proj'] = proj
    #     ebd['funcname'] = func_name
    #     for opt in ret_func_data:
        for o in i[2].keys():
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
                tt.append(instructions_[0])
            tt=torch.IntTensor(tt)
            # ebd[opt] = (tt,cfg)
            # functions[proj] = ebd
            functions[-1][o]=(tt,cfg)
            i[2][o] = len(cfg.nodes(data=True))
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


