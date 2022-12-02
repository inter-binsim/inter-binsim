import os
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import lief
import capstone
import keystone
from tqdm import tqdm
target_path = './func'

ratio = []

file_detail = []
class Assembler(object):
    def __init__(self):
        self.cs = capstone.Cs(self.csmode[0], self.csmode[1])
        self.ks = keystone.Ks(self.ksmode[0], self.ksmode[1])
        return

    @staticmethod
    def _flatten(x):
        temp_zz=''.join(map(chr, x))
        return bytes(x)
        #return ''.join(map(chr, x))

    def asm(self, *args, **kwargs):
        #print("self.ks.asm(*args, **kwargs)[0]:",(self.ks.asm(*args)[0]))
        temp=(self.ks.asm(*args, **kwargs)[0])
        return self._flatten(temp)

    def disasm(self, *args, **kwargs):
        return self.cs.disasm(*args, **kwargs)

    def jmp(self, target, *args, **kwargs):
        raise NotImplementedError("Abstract class")

    def call(self, target, *args, **kwargs):
        raise NotImplementedError("Abstract class")

class X86Assembler(Assembler):
    csmode = (capstone.CS_ARCH_X86, capstone.CS_MODE_32)
    ksmode = (keystone.KS_ARCH_X86, keystone.KS_MODE_32)
    def jmp(self, target, *args, **kwargs):
        return self.asm("jmp {}".format(target), *args, **kwargs)

    def call(self, target, *args, **kwargs):
        return self.asm("call {}".format(target), *args, **kwargs)

class X64Assembler(X86Assembler):
    csmode = (capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    ksmode = (keystone.KS_ARCH_X86, keystone.KS_MODE_64)
    #ksmode=(4,8)
    print("ksmode:",ksmode)

class ARMAssembler(Assembler):
    csmode = (capstone.CS_ARCH_ARM, capstone.CS_MODE_ARM)
    ksmode = (keystone.KS_ARCH_ARM, keystone.KS_MODE_ARM)

def disasm(text):
  asmdict = {
      'x86': X86Assembler,
      'amd64': X64Assembler,
      'arm': ARMAssembler
      }
  arch= 'x86'
  arch_class=asmdict[arch]()
  csmode=arch_class.csmode
  ksmode=arch_class.ksmode
  cs = capstone.Cs(csmode[0], csmode[1])
  data_2=cs.disasm(text, 0)
  list_=[]
  for i in data_2:
    if "call" == i.mnemonic:
      list_.append((i.address, i.size))

  return list_
    #print("0x%x:\t%s\t%s" %(i.address, i.mnemonic, i.op_str), flush=True)


if __name__ == '__main__':
    count = 0
    bbnum_dic = {}
    bbnum_filename_dic = {}
    for root, dirs, files in os.walk(target_path):
      for file in files:
        #print("file name:", file)
        if file.endswith(".txt") and (not file.endswith("_suc.txt"))and (not file.startswith("0_")):
          fn = os.path.join(root, file)
          few = False
          with open(fn) as f:
            lines = f.readlines()
            if int(lines[0]) <= 1:
              few = True
            
            if( few == False ):

              detail_dic = {}
              file_detail.append(detail_dic)
              dic = {}
              i = 1
              while(i < len(lines) ):
                #print("lines:", lines)
                lines[i] = lines[i][:-1]
                #print("lines [i]:", lines[i])
                a, b = lines[i].split(" ")
                dic[a] = 1
                dic[b] = 1
                i+=1
              if len(dic) not in bbnum_dic: 
                bbnum_dic[ len(dic) ] = 1
                bbnum_filename_dic[ len(dic) ] = []
              else:
                bbnum_dic[ len(dic) ] += 1
              bbnum_filename_dic[len(dic)].append(fn)
              ratio.append( int(lines[0]) / len(dic) )

              detail_dic["filename"] = fn.split("/")[-1].strip()
              detail_dic["bbs"] = len(dic)
              detail_dic["edges"] = int(lines[0])
              detail_dic["ratio"] = int(lines[0]) / len(dic)
            
          if few == True:
            os.system("rm "+fn)
          else:
            count+=1

    max_bbnum = 0
    x_bbnum = []
    y_bbnum = []
    for i in sorted(bbnum_dic):
      max_bbnum = i
      x_bbnum.append(i)
      y_bbnum.append(bbnum_dic[i])

    
    
    testtxt = " ./test.txt"
    targettxt = "./target.txt"
    c_b=0

    attack_name="attack.txt" #record attack list
    attack_l=open(attack_name, "w+")

    for b in tqdm(file_detail):
      for a in file_detail:
        print(a["bbs"],b["bbs"],a["edges"],b["edges"])
        if a["bbs"] <= b["bbs"] and a["edges"] <= b["edges"] :
          #print("a filename:", a["filename"], b["filename"])
          #sys.exit()
          ansfile = "./result/result_" + a["filename"][:-4]+"____"+b["filename"][:-4]+ ".txt"
          ansfile = ansfile.strip()
          # if os.path.exists(ansfile) :
          #   #print("continue:"+ansfile.split("/")[-1])
          #   continue

          #print(ansfile.split("/")[-1])

          # p1 = "cat ./" + a["filename"] + ">" + testtxt
          # p2 = "cat ./" + b["filename"] + ">" + targettxt
          # p3 = "/root/rongyi/attack/attack >> " + ansfile
          try:
            os.system("rm " + testtxt)
          except:
            pass
          #print("test:", a["filename"])
          os.system("cat ./func/" + a["filename"] + ">" + testtxt)
          try:
            os.system("rm "+ targettxt)
          except:
            pass
          os.system("cat ./func/" + b["filename"] + ">" + targettxt)

          with open(ansfile, "w") as f:
            print(a, file = f, flush=True)
            print(b, file = f, flush=True)

          
          time.sleep(0.05)
          #time.sleep(2)
          print("norandom start")
          os.system("./norandom >> " + ansfile )
          print("norandom finish")
          #time.sleep(2)
          with open(ansfile, "r") as f:
            #print("ans file:", ansfile)
            tmp_all=f.read()
            if ("no-solution" in tmp_all):
              pass
            else:# success to transfer
              with open("source_target.txt", "w+") as tmp:
                pos=a["filename"].find(".")
                s_name=a["filename"][:pos]
                pos=b["filename"].find(".")
                t_name=b["filename"][:pos]
                print(s_name, t_name, file=tmp, flush=True)
                
                
                with open("flag_label.txt", "w+") as fl:
                  print(t_name, file=fl, flush=True)#print target func name to the txt, for flag label
                
                #apply flag labels, i.e., insert flag to unused basicblocks for inserting junk instructions
                opt_func_name="./func/"+t_name+"_suc.txt"
                with open(opt_func_name, "r") as tmp:
                  opt_file=tmp.readline()
                  opt_file=opt_file.split(" ")[1].strip() 
                opt_file="./" + opt_file
                opt_bc="./opt/" + t_name + "_flag" + ".bc" #transfer pair
                os.system("opt -load  /llvm-project-10.0.1/build/lib/Transforms/Flag_blocks/build/libFlag_blocks.so -flag_blocks " + opt_file + " -o " + opt_bc )
                print("flag has been inserted")
                flag_t=t_name+"_flag"
                os.system("clang " + opt_bc + " -lm " + " -o " + flag_t) #get the flagged opt file
                print("calculate flag pos", flag_t)
                b_f=lief.parse(flag_t)
                try:# b_f may be none
                    sym=b_f.symbols
                    #print("flag t:", flag_t, b_f)
                except AttributeError as e:
                    print("attribute error happen")
                    continue
                sym_dict=dict()
                b_text=b_f.get_section(".text")
                text_vir=b_text.virtual_address
                text_off=b_text.file_offset
                for sy in sym:
                  #print("sy.name:", sy.name)
                  sym_dict[sy.name]=sy.value 

                v_dict=dict()  
                with open("flag_tar.txt", "w+") as f_t:             
                  with open("flag.txt", "r") as ff: # calculate block size, flag.txt only contains label name, i.e., s_name, e_name
                    for line in ff.readlines():
                      ls=line.split(" ")[0].strip()
                      le=line.split(" ")[1].strip()
                      v_s=sym_dict[ls[:-1]]-text_vir+text_off
                      v_e=sym_dict[le[:-1]]-text_vir+text_off
                      v_dict[ls[:-3]]=v_e-v_s# to do
                      #print("ls:", ls, le)
                      print(ls[:-3], v_s, v_e-v_s, flush=True, file=f_t)#label name, start point, length




                #apply cfg transfer
                opt_func_name="./func/"+s_name+"_suc.txt"
                with open(opt_func_name, "r") as tmp:
                  opt_file=tmp.readline()
                  opt_file=opt_file.split(" ")[1].strip()
                opt_file="./" + opt_file
                
                opt_bc="./opt/" + s_name + "_" + t_name + ".bc"#transfer pair
                print("opt file :", opt_file, opt_bc)

                status = os.system("opt -load  /llvm-project-10.0.1/build/lib/Transforms/llvm-pass-skeleton/build/libSkeletonPass.so -hello " + opt_file + " -o " + opt_bc )
                # if status == 34304:
                #   print("status:", status)
                #   sys.exit()
                
                
                rev=os.system("clang -O0 -lm -o binary/"+ s_name+"_"+t_name + " " + opt_bc)
                #sys.exit()
                if rev==0:# some .c fail to compile, here only save successful pairs
                  print(s_name, t_name, file=attack_l, flush=True)# record pair to test binary similarity tools
                else:
                  continue

                #apply flag labels to source binary
                flag_s="binary/" + s_name + "_" + t_name
                b_f=lief.parse(flag_s)
                sym=b_f.symbols
                sym_dict=dict()
                b_text=b_f.get_section(".text")
                text_vir=b_text.virtual_address
                text_off=b_text.file_offset
                for sy in sym:
                  sym_dict[sy.name]=sy.value 

                v_dict=dict() 

                b_t=open(flag_t,"rb+")# open target flagged binary
                text_bt=b_t.read()
                text_bt=bytearray(text_bt)

                b_s=open(flag_s,"rb+")# open ori flagged binary
                text_bs=b_s.read()
                text_bs=bytearray(text_bs)
                
                with open("flag_ori.txt", "r+") as ff:             
                  #with open("flag.txt", "r") as ff: # calculate block size, flag.txt only contains label name, i.e., s_name, e_name
                  for line in ff.readlines():
                    ls=line.split(" ")[0].strip()
                    le=line.split(" ")[1].strip()

                    pos_s=int(line.split(" ")[2].strip())
                    num_=int(line.split(" ")[3].strip())
                    v_s=sym_dict[ls[:-1]]-text_vir+text_off
                    v_e=sym_dict[le[:-1]]-text_vir+text_off
                    #print("vs:ve:", text_bs[v_s : v_e], text_bt[pos_s : pos_s + num_])
                    if num_>0:
                      list_=disasm(text_bt[pos_s : pos_s + num_])
                      text_bs[v_s : v_e]=text_bt[pos_s : pos_s + num_]
                      for tuple_ in list_:# remove call ins, for ida to well extract cfg
                        text_bs[v_s + tuple_[0] : v_s+tuple_[0] + tuple_[1] ] = tuple_[1] * b'\x90'
                    # else:
                    #   text_bs[v_s : v_e-5]=text_bt[pos_s : pos_s + num_-5]
                    
                    
                
                b_s.close()
                b_fakeins=open(flag_s, "w+")
                b_fakeins.buffer.write(text_bs)
                b_fakeins.close()
                print("flag_s:", flag_s)

                b_s=open(flag_s,"rb+")# open ori flagged binary
                text_bs=b_s.read()
                text_bs=bytearray(text_bs)
                #sys.exit()
        
                
                tar_func_name="./func/" + t_name + "_suc.txt"
                print("t_name:", tar_func_name)
                with open(tar_func_name, "r") as tmp:
                  tar_file=tmp.readline()
                  tar_file=tar_file.split(" ")[1].strip()
                tar_file="./" + tar_file[:-2] + "c"
                print("tar:", tar_file)
                
                
                os.system("clang " + tar_file + " -O1 -lm -o binary/"+ t_name + "_o1")
                os.system("clang " + tar_file + " -O2 -lm -o binary/"+ t_name + "_o2")
                os.system("clang " + tar_file + " -O3 -lm -o binary/"+ t_name + "_o3")
                os.system("clang " + tar_file + " -Os -lm -o binary/"+ t_name + "_os")
                os.system("clang " + tar_file + " -O0 -lm -o binary/"+ t_name + "_o0")
                os.system("clang " + tar_file + " -Ofast -lm -o binary/"+ t_name + "_of")
                print(" success")
                c_b+=1
          time.sleep(0.05)
          if c_b>50:
            sys.exit()




    
    
    
    
          
          


