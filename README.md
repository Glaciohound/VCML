## to use Jacinle python library, 

```
export PATH=<path_to_jacinle>/bin:$PATH
jac-run xxx.py to replace python3 xxx.py. 
```

You can also use the `jac-crun <gpu_ids> xxx.py` to set the gpus you want to
use. Here,` <gpu_ids>` is a comma-separated list of gpu ids, following the
convension of `CUDA_VISIBLE_DEVICES`

## notes

This code requires a `data` symbolic link beside the root directory of this
codebase

## commonly used scripts:
Here,
`{*, *, ...}` stands for any one inside the brace
`[*, *, ...]` stands for one or more inside the brackets, (without the brackets
        themselves)

### classification:

```
jac-run train.py --task clevr_pt
    --subtask classification
    --classification color shape
    --name classification --max_sizeDataset 20000
--silent
```

### exist, filter, query:
```
jac-run train.py --task clevr_pt
    --subtasks filter exist
    --name any_name --silent
```

### isinstance
```
jac-run train.py --task clevr_pt
    --subtasks exist isinstance
    --visualize_relation isinstance
    --val_concepts red blue cube
    --name any_name --silent
```

### isinstance for de-biasing
setting A (with assistance of 'isinstance' questions)
```
jac-run train.py --task clevr_pt
    --subtasks classification isinstance
    --visual_bias red:large large:red,blue
    --generalization_ratio 0
    --visualize_relation isinstance --name any_name --silent
```
setting B (without assistance of 'isinstance' questions) by adding `--no_aid`
option
```
jac-run train.py --task clevr_pt
    --subtasks classification isinstance
    --visual_bias red:large large:red,blue
    --no_aid
    --visualize_relation isinstance --name any_name --silent
```

### synonym
```
jac-run train.py --task clevr_pt
    --subtasks exist synonym
    --synonym blue red
    --name any_name --silent
```

### common options

`--task` for selecting VQA task data. `clevr_pt` stands for `clevr pretrained feature`,
    `clevr_dt` stands for `clevr detected bounding boxes`
    (i.e. jointly training attribute network),
    and `toy` stands for using a toy dataset with ground-truth attributes for objects

`--subtasks` for selecting (one or more) subtasks, from visual subtasks such as 
`filter`, `exist`, `query`, and conceptual subtasks such as `isinstance`, `synonym`.

`--random_seed 0` for setting manual random seed to `0`, otherwise no manual random
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
