import os
import numpy as np
import multiprocessing as mp
import sys
from pprint import pprint

concepts = ['blue', 'brown', 'purple', 'cyan',
            'gray', 'green', 'red', 'yellow',
            'large', 'small',
            'cube', 'sphere', 'cylinder',
            'metal', 'rubber']


commands = []

if 'filter_isinstance' in sys.argv[1]:

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
                --model h_embedding_add2 --non_bool_weight 0.01 --lr 0.01 --init_variance 0.001\
                --max_sizeDataset 20000'\
                .format(np.random.randint(sys.argv[4]), concept, i, sys.argv[3])

    i = int(sys.argv[2])

    for concept in concepts:
        commands.append(command(concept, i))

elif sys.argv[1] == 'visual_bias':

    def command(random_seed, aid):
        return 'jac-crun {0} train.py --task clevr_pt --subtask visual_bias\
            --name bias_{1}{2} --random_seed {1}\
            --epochs {3}\
            --train_config sphere:blue blue:sphere'\
            .format(np.random.randint(sys.argv[4]), i,
                    '' if aid else '_noaid --no_aid', sys.argv[3])

    n = int(sys.argv[2])

    for i in range(n):
        commands.append(command(i, True))
        commands.append(command(i, False))


processes = []
nohup = ''

for command_ in commands:
    p = mp.Process(target=os.system, args=(nohup+command_,))
    nohup = 'nohup '
    p.start()
    processes.append((p, command_))
pprint(commands)

for p in processes:
    p[0].join()
    print('recycle:', p[1])
