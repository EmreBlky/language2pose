import os
import sys
import matplotlib.pyplot as plt
import numpy as np

base_dir = '/viscam/u/ying1123/language2pose/src/save/model'
template = 'exp_{}_cpk_jl2p_model_Seq2SeqConditioned9_time_16_chunks_1'

dic = {259: 'slurm-3837685.out',
       260: 'slurm-3837697.out',
       261: 'slurm-3837702.out',
       262: 'slurm-3837705.out',
       263: 'slurm-3837707.out',
       42: 'slurm-3840800.out',
       46: 'slurm-3846541.out'}

model_id = 46
if len(sys.argv) > 1:
    model_id = int(sys.argv[1])
log_file = dic[model_id]
print(model_id, log_file)

with open(log_file, 'r') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]

epoch = 0
train = []
dev = []
test = []
refined = []
for line in lines:
    if 'epch: ' in line:
        if 'None' in line:
            break
        refined.append(line)
        start = line.find('epch: ')
        end = line.find(', lr:')
        number = line[start + 6 : end]
        assert int(number) == epoch
        epoch += 1
    if '| loss |' in line:
        refined.append(line)
        tokens = line.split('|')
        train.append(float(tokens[2]))
        dev.append(float(tokens[3]))
        test.append(float(tokens[4]))

#for line in refined:
#    print(line)

#print(train[-5:])
#print(dev[-5:])
#print(test[-5:])
#print(len(train), len(dev), len(test))
#print(epoch)

x = np.array(range(epoch))
train = np.array(train)
dev = np.array(dev)
test = np.array(test)

plt.plot(x, train, color='r', label='train')
plt.plot(x, dev, color='g', label='dev')
plt.plot(x, test, color='b', label='test')

plt.xlabel('epochs')
plt.ylabel('loss')

plt.legend()

dirname = os.path.join(base_dir, template.format(model_id))
filename = os.path.join(dirname, 'loss_vs_epoch.pdf')
print('write plot to', filename)

plt.savefig(filename)

