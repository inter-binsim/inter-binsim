import os
from tqdm import tqdm
import pickle
c_path = '/workspace/code'
bc_path = '/workspace/test_full'
result_file = '/workspace/GraphEmb/result/test_full'
file_list = []
file_root = []
if not os.path.exists(bc_path):
    os.mkdir(bc_path)
for root, dirs, files in os.walk(result_file):
    if root == result_file:
        continue
    file_list.append(files[0])
    file_root.append(f'{root}/{files[0]}')
for y in tqdm(range(len(file_root))):
    with open(file_root[y], 'r') as f:
        result = f.readlines()
    target = file_list[y][:-4]
    target_file = target.split('@')[0]
    t_bc_path = bc_path + '/' + target
    target_func = target.split('@')[1]
    t_c_path = c_path + '/' + target_file + '.c'
    if not os.path.exists(t_bc_path):
        os.mkdir(t_bc_path)
    # print(f'clang -c -O2 -emit-llvm {t_c_path} -o {t_bc_path}/{target_file}@O2.bc')
    # print(f'opt -dot-cfg {t_bc_path}/{target_file}@O2.bc')
    # print(f'dot -Tpng -o {t_bc_path}/{target}@O2.png .{target_func}.dot')
    #
    # print(f'clang -c -O3 -emit-llvm {t_c_path} -o {t_bc_path}/{target_file}@O3.bc')
    # print(f'opt -dot-cfg {t_bc_path}/{target_file}@O3.bc')
    # print(f'dot -Tpng -o {t_bc_path}/{target}@O3.png .{target_func}.dot')
    os.system(f'clang -c -O2 -emit-llvm {t_c_path} -o {t_bc_path}/{target_file}@O2.bc')
    os.system(f'opt -dot-cfg {t_bc_path}/{target_file}@O2.bc')
    os.system(f'dot -Tpng -o {t_bc_path}/{target}@O2.png .{target_func}.dot')

    os.system(f'clang -c -O3 -emit-llvm {t_c_path} -o {t_bc_path}/{target_file}@O3.bc')
    os.system(f'opt -dot-cfg {t_bc_path}/{target_file}@O3.bc')
    os.system(f'dot -Tpng -o {t_bc_path}/{target}@O3.png .{target_func}.dot')

    num = 0
    for x in result:
        # target = x.split(',')[0]
        num += 1
        if num >= 10:
            break
        yuan = x.split(',')[0]
        yuan_file = yuan.split('@')[0]
        yuan_func = yuan.split('@')[1]
        y_c_path = c_path + '/' + yuan_file + '.c'
        os.system(f'clang -c -emit-llvm {y_c_path} -o {t_bc_path}/{yuan_file}.bc')
        os.system(f'opt -dot-cfg {t_bc_path}/{yuan_file}.bc')
        os.system(f'dot -Tpng -o {t_bc_path}/{num}@{yuan}.png .{yuan_func}.dot')

        # os.system(f'clang -c -emit-llvm {t_c_path} -o {t_bc_path}/{target_file}.bc')
        # os.system(f'opt -dot-cfg {t_bc_path}/{target_file}.bc')
        # os.system(f'dot -Tpng -o {t_bc_path}/{target}.png .{target_func}.dot')
        #
        # os.system(f'clang -c -O1 -emit-llvm {t_c_path} -o {t_bc_path}/{target_file}@O1.bc')
        # os.system(f'opt -dot-cfg {t_bc_path}/{target_file}@O1.bc')
        # os.system(f'dot -Tpng -o {t_bc_path}/{target}@O1.png .{target_func}.dot')
        #
        # os.system(f'clang -c -emit-llvm {y_c_path} -o {t_bc_path}/{yuan_file}.bc')
        # os.system(f'opt -dot-cfg {t_bc_path}/{yuan_file}.bc')
        # os.system(f'dot -Tpng -o {t_bc_path}/{yuan}.png .{yuan_func}.dot')

        # os.system(f'clang -c -emit-llvm {t_c_path} -o {t_bc_path}/{target_file}.bc')



