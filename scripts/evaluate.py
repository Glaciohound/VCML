import os
import numpy as np
import multiprocessing as mp
import sys

concepts = ['blue', 'brown', 'purple', 'cyan',
            'gray', 'green', 'red', 'yellow',
            'large', 'small',
            'cube', 'sphere', 'cylinder',
            'metal', 'rubber']

if sys.argv[1] == 'filter_isinstance_pt':
    def command(concept, i=0):
        return 'jac-crun {0} train.py --task clevr_pt --subtask filter_isinstance\
            --name pfi_{1}_{2} --val_concepts {1}\
            --epochs {3}\
            --model h_embedding_add2 --non_bool_weight 0.01 --lr 0.01 --init_variance 0.001'\
            .format(np.random.randint(sys.argv[4]), concept, i, sys.argv[3])

elif sys.argv[1] == 'filter_isinstance_dt':
    def command(concept, i=0):
        return 'jac-crun {0} train.py --task clevr_dt --subtask filter_isinstance\
            --name dfi_{1}_{2} --val_concepts {1}\
            --epochs {3}\
            --model h_embedding_add2 --non_bool_weight 0.01 --lr 0.01 --init_variance 0.001'\
            .format(np.random.randint(sys.argv[4]), concept, i, sys.argv[3])

i = int(sys.argv[2])

processes = []
nohup = ''

for concept in concepts:
    command_ = command(concept, i)
    print(command_)
    p = mp.Process(target=os.system, args=(nohup+command_,))
    nohup = 'nohup '
    p.start()
    processes.append((p, command_))

for p in processes:
    p[0].join()
    print('recycle:', p[1])
