import os
import pandas as pd
import numpy as np
from tqdm import tqdm

gen_dir = '/viscam/u/ying1123/language2pose/src/save/model/exp_42_cpk_jl2p_model_Seq2SeqConditioned9_time_16_chunks_1'
gt_dir = '/viscam/u/ying1123/dataset/kit-mocap/kit-mocap-lang2pose'

joints = ['root', 'BP', 'BT', 'BLN', 'BUN', 'LS', 'LE', 'LW', 'RS', 'RE',\
          'RW', 'LH', 'LK', 'LA', 'LMrot', 'LF', 'RH', 'RK', 'RA', 'RMrot',\
          'RF']
suffix = '.fke'

def l2dis(p1, p2):
    assert len(p1) == 3 and len(p2) == 3
    dis = 0
    for i in range(3):
        dis += (p1[i] - p2[i]) ** 2
    return dis ** 0.5

def main():
    ape = np.zeros((3, len(joints)))
    for dataset, desc in enumerate(['train', 'dev', 'test']):
        count = 0

        cur_path = os.path.join(gen_dir, desc)
        for filename in tqdm(os.listdir(cur_path)):
            if not filename.endswith(suffix):
                continue
            mat_path = os.path.join(cur_path, filename)
            gt_path = os.path.join(gt_dir, filename)
            mat = pd.read_csv(mat_path)
            gt = pd.read_csv(gt_path)
            
            gen_len = len(mat.values)
            assert len(mat.values) <= len(gt.values), f'{filename}'
            count += gen_len
 
            for j, joint in enumerate(joints):
                mat_joint = mat[[joint+sub for sub in ['_tx', '_ty', '_tz']]].values
                gt_joint = gt[[joint+sub for sub in ['_tx', '_ty', '_tz']]].values
                for i in range(gen_len):
                    ape[dataset][j] += l2dis(mat_joint[i], gt_joint[i])
#                    print(i, l2dis(mat_joint[i], gt_joint[i]))
        
#        print(f'count: {count}')
#        print(f'ape[{desc}]: {ape[dataset]}')
        ape[dataset] /= count
        print(f'ape[{desc}]: {ape[dataset]}')


    ## write to file
    filename = os.path.join(gen_dir, 'ape.csv')
    with open(filename, 'w') as f:
        pd.DataFrame(data=ape, index=['train', 'dev', 'test'], columns=joints).to_csv(f)

if __name__ == '__main__':
    main()

