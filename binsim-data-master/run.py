import os
import subprocess
import multiprocessing
import time
import shutil
from util.pairdata import pairdata
from tqdm import tqdm

ida_path = "/workspace/ida/ida_linux/idapro-7.6/idat64"
work_dir = os.path.abspath('.')
dataset_dir = './binary'
strip_path = "./dataset_strip/"
idb_path = "./idb"
script_path = "./process.py"
SAVE_ROOT = "./result"

def getTarget(path, prefixfilter=None):
    target = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if prefixfilter is None:
                target.append(os.path.join(root, file))
            else:
                for prefix in prefixfilter:
                    if file.startswith(prefix):
                        target.append(os.path.join(root, file))
    return target
def setDir(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)
if __name__ == '__main__':
    # prefixfilter = ['libcap-git-setcap']

    setDir(SAVE_ROOT)
    setDir(idb_path)
    setDir(strip_path)
    start = time.time()
    target_list = getTarget(dataset_dir)

    pool = multiprocessing.Pool(processes=8)
    for target in tqdm(target_list):
        filename = target.split('/')[-1]
        filename_strip = filename + '.strip'
        ida_input = os.path.join(strip_path, filename_strip)
        os.system(f"strip -s {target} -o {ida_input}")
        print(f"strip -s {target} -o {ida_input}")

        # cmd_str = f'{ida_path} -Llog/{filename}.log -c -A -S{script_path} -oidb/{filename}.idb {ida_input}'
        cmd_str = f'{ida_path} -Llog/{filename}.log -c -A -S{script_path} -o{idb_path}/{filename}.idb {ida_input}'

        print(cmd_str)
        cmd = [ida_path, f'-Llog/{filename}.log', '-c', '-A', f'-S{script_path}', f'-o{idb_path}/{filename}.idb', f'{ida_input}']
        pool.apply_async(subprocess.call, args=(cmd,))
    pool.close()
    pool.join()
    print('[*] Features Extracting Done')
    pairdata(SAVE_ROOT)
    end = time.time()
    print(f"[*] Time Cost: {end - start} seconds")
