#!/usr/bin/env python3

import os
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
import pickle
import argparse
from functools import reduce
import numpy as np

class DatasetBase(object):
    def __init__(self, path, prefixfilter=None, all_data=True, opt=None):
        self.path = path
        self.prefixfilter = prefixfilter
        self.all_data = all_data
        self.unpaired = defaultdict(list)
        self.opt = opt
        if self.opt is not None:
            # assert len(self.opt) == 2, "set len(opt) != 2"
            self.paired = defaultdict(defaultdict)
        else:
            self.paired = defaultdict(list)
        assert os.path.exists(self.path), "Dataset Path Not Exists"
        assert (self.prefixfilter is not None) != self.all_data, "You should set prefixfilter with all_data = False"

    def traverse_file(self):
        for root, dirs, _ in os.walk(self.path):
            for dir in dirs:
                if self.all_data:
                    for file in os.listdir(os.path.join(root, dir)):
                        yield dir, file, os.path.join(root, dir, file)
                else:
                    for filter in self.prefixfilter:
                        if dir.startswith(filter):
                            for file in os.listdir(os.path.join(root, dir)):
                                yield dir, file, os.path.join(root, dir, file)          

    def load_pickle(self, file):
        with open(file, 'rb') as f:
            return pickle.load(f)

    def load_unpair_data(self):
        for proj, filename, pkl_path in self.traverse_file():
            if filename != 'saved_index.pkl':
                pickle_data = self.load_pickle(pkl_path)
                self.unpaired[proj].append(pickle_data)
    
    def load_pair_data(self):
        if self.opt is None:
            for proj, filename, pkl_path in self.traverse_file():
                if filename == 'saved_index.pkl':
                    pickle_data = self.load_pickle(pkl_path)
                    self.paired[proj].append(pickle_data)
        else:
            for proj, filename, pkl_path in self.traverse_file():
                if filename == 'saved_index.pkl':
                    continue
                opt = filename.split('-')[-2]
                if opt in self.opt:
                    print(filename)
                    pickle_data = self.load_pickle(pkl_path)
                    self.paired[proj][opt] = pickle_data
                        
    def get_unpaird_data_iter(self):
        for proj, filename, pkl_path in self.traverse_file():
            if filename != 'saved_index.pkl':
                pickle_data = self.load_pickle(pkl_path)
                for func_name, func_data in pickle_data.items():
                    func_addr, asm_list, rawbytes_list, cfg, biai_featrue = func_data
                    yield proj, func_name, func_addr, asm_list, rawbytes_list, cfg, biai_featrue


    def get_paired_data_iter(self):
        proj2pickle = defaultdict(defaultdict)
        for proj, filename, pkl_path in self.traverse_file():
            if filename == 'saved_index.pkl':
                continue
            opt = filename.split('-')[-2]
            proj2pickle[proj][opt] = pkl_path

        for proj, pickle_path_dict in proj2pickle.items():
            if len(pickle_path_dict) < 2:
                continue
            function_list = []
            tmp_pickle_dict = {}
            for opt, pkl_path in pickle_path_dict.items():
                pkl = pickle.load(open(pkl_path, 'rb'))
                function_list.append(list(pkl.keys()))
                tmp_pickle_dict[opt] = pkl
            function_set = reduce(lambda x,y : set(x) & set(y), function_list)
            for func_name in function_set:
                ret_func_data = defaultdict()
                for opt, pkl in tmp_pickle_dict.items():
                    ret_func_data[opt] = pkl[func_name]
                yield proj, func_name, ret_func_data

    def get_unpaird_data(self):
        for proj, pkl_list in self.unpaired.items():
            for pkl in pkl_list:
                for func_name, func_data in pkl.items():
                    func_addr, asm_list, rawbytes_list, cfg, biai_featrue = func_data
                    yield proj, func_name, func_addr, asm_list, rawbytes_list, cfg, biai_featrue

    def get_paired_data(self):
        if self.opt is None:
            for proj, pkl_list in self.paired.items():
                for pkl in pkl_list:
                    for func_name, func_data_list in pkl.items():
                        yield proj, func_name, func_data_list
                         # for func_data in func_data_list:
                         #       func_addr, asm_list, rawbytes_list, cfg, biai_featrue = func_data
        else:
            for proj, pkl_dict in self.paired.items():
                if len(pkl_dict) < 2:
                    continue
                function_list = []
                for opt, pkl in pkl_dict.items():
                    function_list.append(list(pkl.keys()))
                function_set = reduce(lambda x,y : set(x) & set(y), function_list)
                for func_name in function_set:
                    ret_func_data = defaultdict()
                    for opt, pkl in pkl_dict.items():
                        ret_func_data[opt] = pkl[func_name]
                    yield proj, func_name, ret_func_data
#                     for opt in self.opt:
#                         func_addr, asm_list, rawbytes_list, cfg, biai_featrue = ret_func_data[opt]
    def traverse_cfg_node(self, cfg):
        for node in cfg.nodes():
            yield cfg.nodes[node]['asm'], cfg.nodes[node]['raw']

class DataBaseCrossCompiler(DatasetBase):
    def __init__(self, path, prefixfilter=None, all_data=True, opt=None):
        super(DataBaseCrossCompiler, self).__init__(path, prefixfilter, all_data, opt)

    def load_pair_data(self):
        if self.opt is not None:
            for proj, filename, pkl_path in self.traverse_file():
                if filename == 'saved_index.pkl':
                    continue
                opt = filename.split('-')[-2]
                compiler = filename.split('-')[-3]
                final_opt = compiler+opt
                if opt in self.opt:
                    print(filename)
                    pickle_data = self.load_pickle(pkl_path)
                    self.paired[proj][final_opt] = pickle_data
        else:
            print("opt is None")
            exit(1)

    def get_paired_data(self): 
        # return proj, func_name, ret_func_data 
        # ret_func_data = {
        #                   opt: {
        #                           compiler : (func_addr, asm_list, rawbytes_list, cfg, biai_featrue) 
        #                        } 
        #                  }
        if self.opt is not None:
            for proj, pkl_dict in self.paired.items():
                if len(pkl_dict) < 2:
                    continue
                function_list = []
                for opt, pkl in pkl_dict.items():
                    function_list.append(list(pkl.keys()))
                function_set = reduce(lambda x,y : set(x) & set(y), function_list)
                for func_name in function_set:
                    ret_func_data = defaultdict()
                    for opt, pkl in pkl_dict.items():
                        ret_func_data[opt] = pkl[func_name]
                    yield proj, func_name, ret_func_data
        else:
            print("opt is None")
            exit(1)