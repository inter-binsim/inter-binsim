import os

target_path = 'proj'
dot_path = './'

all_file=''

if __name__ == '__main__':
  #transform all .c to .bc using llvm
  for root, dirs, files in os.walk(target_path): 
    for file in files:
      f = os.path.join(root,file)
      if file.endswith(".c") :
        #os.system("clang -c -emit-llvm -fno-discard-value-names " + f + " -S -o " + f[:-2] + ".ll")
        os.system("clang -c -emit-llvm  " + f + " -o " + f[:-2] + ".bc")
        all_file=all_file+" "+f[:-2]+".bc"+" "
    

  os.system("llvm-link "+all_file+" -o "+ "proj/jointed.bc")
  print(all_file)
  #remove all old file
  for root, dirs, files in os.walk(target_path):
    for file in files:
      f = os.path.join(root,file)
      if file.endswith("_changed.bc") :
        os.system("rm " + f)
  
  #extract cfg for each func from .bc.
  for root, dirs, files in os.walk(target_path):
    for file in files:
      f = os.path.join(root,file)
      if file.endswith(".bc") :
        os.system("opt -load /llvm-project-10.0.1/build/lib/Transforms/GetCFG/getcfg/libGetCFG.so -getcfg " + f + " -o " + f[:-3] + "_changed.bc" )
        

