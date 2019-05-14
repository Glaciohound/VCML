## to use Jacinle python library, 

```
export PATH=<path_to_jacinle>/bin:$PATH
jac-run xxx.py to replace python3 xxx.py. 
```

You can also use the `jac-crun <gpu_ids> xxx.py` to set the gpus you want to
use. Here,` <gpu_ids>` is a comma-separated list of gpu ids, following the
convension of `CUDA_VISIBLE_DEVICES`

## notes

This code requires a `data` symbolic link sibling the root directory of this
codebase

## commonly used scripts:

### classification:

```
jac-run train.py --task {clevr_pt, clevr_dt} --subtask classification
--classification [color, shape, ...] --name classification --max_sizeDataset 20000
--silent
```

### exist, filter, query:
`{*, *, ...}` stands for any one inside the brace
`[*, *, ...]` stands for one or more inside the brackets, (without the brackets
        themselves)

```
jac-run train.py --task {clevr_pt, clevr_dt, toy} --subtasks {exist,
    filter, query} --name any_name --silent
```

### isinstance
```
jac-run train.py --task {clevr_pt, clevr_dt} --subtasks {exist, filter, query,
    classification} isinstance --visualize_relation isinstance
    --name any_name --silent
```

### isinstance for de-biasing
```
jac-run train.py --task {clevr_pt, clevr_dt} --subtasks {exist, filter, query,
    classification} isinstance --visual_bias red:large large:red,blue
    --generalization_ratio 0 --visualize_relation isinstance --name any_name --silent
```

### synonym
```
jac-run train.py --task {clevr_pt, clevr_dt} --subtasks {exist, filter, query,
    classification} synonym --synonym blue red --name any_name --silent
```

### common options

`--task` for selecting VQA task data. `clevr_pt` stands for `clevr pretrained feature`,
    `clevr_dt` stands for `clevr detected bounding boxes`
    (i.e. jointly training attribute network),
    and `toy` stands for using a toy dataset with ground-truth attributes for objects

`--random_seed 0` for setting manual random seeds, otherwise no manual random
seed is set.

`--init_variance 0.001` for specifying initial variance of parameters, with defualt value 0.001.

`--lr 0.001` for setting learning rate

`--silent` for turning off visualization, checkpoint saving and result logging

`--name trial` is a name for this experiment used for visualization, logging,
    saving checkpoints, etc

`--visual_bias red:large large:red,blue` for specifying visual-bias configurations,
    e.g. `--visual_bias red:sphere,large blue:small` will be tidied into
    `args.visual_bias == {'red': ['sphere', 'large'], 'blue': ['small']}`,
    which may enforce all 'red' objects to be 'large', while all 'large'
    objects to be 'red' or 'blue'

`--synonym red blue` for adding synonyms. In this case, synonyms of 'red' and
'blue' will be added into the VQA corpus

### deprecated options

`--model` arguments other than `h_embedding_add2` are deprecated
some options in `config.py` are deprecated and list at the end of
`parse_args()` method()
