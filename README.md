## to use Jacinle python library, 

```
export PATH=<path_to_jacinle>/bin:$PATH
jac-run xxx.py to replace python3 xxx.py. 
```

You can also use the `jac-crun <gpu_ids> xxx.py` to set the gpus you want to
use. Here,` <gpu_ids>` is a comma-separated list of gpu ids, following the
convension of `CUDA_VISIBLE_DEVICES`

## notes

Please set `--clevr_data_dir` to a directory containing CLEVR dataset, e.g.
`--clevr_data_dir /data/vision/billf/scratch/chihan/clevr` which is the
defualt value for this option

Please specify your own `--visualize_dir`, `--log_dir`, `--ckpt_dir` to avoid
permission issues, or set `--silent` to turn off file output

## commonly used scripts:

### classification:

```
jac-run scripts/train.py --task clevr_pt
    --subtask classification
    --name classification --max_sizeDataset 20000
    --silent
```

### exist, filter, query:
```
jac-run scripts/train.py --task clevr_pt
    --subtasks filter exist
    --name any_name --silent
```

### isinstance
pretrained feature version
```
jac-run scripts/train.py --task clevr_pt
    --subtasks exist isinstance
    --visualize_relation isinstance
    --val_concepts red blue cube
    --name any_name --silent
```

detection version
```
jac-run scripts/train.py --task clevr_dt
    --subtask filter isinstance
    --visualize_relation isinstance
    --val_concepts blue large
    --feature_dim 256
    --max_sizeDataset 20000 
    --name any_name 
```

### isinstance for de-biasing
setting A (with assistance of 'isinstance' questions)
```
jac-run scripts/train.py --task clevr_pt
    --subtasks classification isinstance
    --visual_bias red:large large:red,blue
    --generalization_ratio 0
    --visualize_relation isinstance --name any_name --silent
```
setting B (without assistance of 'isinstance' questions) by adding `--no_aid`
option
```
jac-run scripts/train.py --task clevr_pt
    --subtasks classification isinstance
    --visual_bias red:large large:red,blue
    --no_aid
    --visualize_relation isinstance --name any_name --silent
```

### synonym
```
jac-run scripts/train.py --task clevr_pt
    --subtasks exist synonym
    --synonym blue red
    --name any_name --silent
```

### common options

`--task` for selecting VQA task data. `clevr_pt` stands for `clevr pretrained feature`,
    `clevr_dt` stands for `clevr detected bounding boxes`
    (i.e. jointly training attribute network),
    and `toy` stands for using a toy dataset with ground-truth attributes for objects

When in `--task clevr_dt` setting, please specify `--feature_dim 256`.
When in `--task toy` setting, please specify `--feature_dim 50` or
`--embed_dim 512`


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


`--incremental_training partial full` for specifying trianing scheme, which in This
case will first train with a `partial` dataset, where concepts specified by
`--removed_concepts` will not be trained or tested, and then, when accuracy achieves
threshold specified by `--perfect_th`, change to a `full` dataset to train and
test all concepts


`--visualize_relation isinstance` visualize information related to
`isinstance` metaconcept


`--no_validation` no validation or generalization test


`--val_by_classification red blue` when in validation, test classification
performance on `red` and `blue` rather vqa performance


`--max_sizeDataset 20000` for generating 20000 questions in the dataset


### deprecated options

`--model` arguments other than `h_embedding_add2` are deprecated.

some options in `config.py` are deprecated and listed at the end of
`Config.parse_args()` method()
