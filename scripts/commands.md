Shown below are commands used in experiments in VCML paper:

## Commands used in *Concepts Help Metaconcept Generalization*

### *Synonym* generalization on CLEVR dataset
```
jac-crun <GPU-id> scripts/main.py --mode run-experiment --task CLEVR --model VCML --experiment synonym_generalization --num_parallel {num} --log_dir {log_dir} --data_dir {data_dir}
```
For instance:
```
jac-crun 0,1,2,3 scripts/main.py --mode run-experiment --task CLEVR --model VCML --experiment synonym_generalization --num_parallel 4 --log_dir ../data/log --data_dir ../data/vcml
```

### *Same-kind* generalization on CLEVR dataset
```
jac-crun <GPU-id> scripts/main.py --mode run-experiment --task CLEVR --model VCML --experiment samekind_generalization --num_parallel {num} --log_dir {log_dir} --data_dir {data_dir}
```

### *Synonym* generalization on GQA dataset
```
jac-crun <GPU-id> scripts/main.py --mode run-experiment --task GQA --model VCML --experiment synonym_generalization --num_parallel {num} --log_dir {log_dir} --data_dir {data_dir} --sample_size 10000
```

### *Same-kind* generalization on GQA dataset
```
jac-crun <GPU-id> scripts/main.py --mode run-experiment --task GQA --model VCML --experiment samekind_generalization --num_parallel {num} --log_dir {log_dir} --data_dir {data_dir} --sample_size 10000
```

### *Hypernym* generalization on CUB dataset
```
jac-crun <GPU-id> scripts/main.py --mode run-experiment --task CUB --model VCML --experiment hypernym_generalization --num_parallel {num} --log_dir {log_dir} --data_dir {data_dir} --sample_size 50000
```

### *Meronym* generalization on CUB dataset
```
jac-crun <GPU-id> scripts/main.py --mode run-experiment --task CUB --model VCML --experiment meronym_generalization --num_parallel {num} --log_dir {log_dir} --data_dir {data_dir} --sample_size 50000
```

## Commands used in *Metaconcepts Help Concept Learning*

### *Synonym* Supports Zero-Shot Learning of Novel Concepts
```
jac-crun <GPU-id> scripts/main.py --mode run-experiment --task CLEVR --model VCML --experiment zeroshot --num_parallel {num} --log_dir {log_dir} --data_dir {data_dir} --penalty 1e-4 --length_penalty 1e-3
```
```
jac-crun <GPU-id> scripts/main.py --mode run-experiment --task GQA --model VCML --experiment zeroshot --num_parallel {num} --log_dir {log_dir} --data_dir {data_dir} --sample_size 10000 --penalty 1e-4 --length_penalty 1e-3
```

### *Same-kind* Supports Learning from Biased Data

```
jac-crun <GPU-id> scripts/main.py --mode run-experiment --task CLEVR --model VCML --experiment debiasing_leaked --name debiasing_leaked_200 --num_parallel {num} --log_dir {log_dir} --data_dir {data_dir} --penalty 0.1
```

```
jac-crun <GPU-id> scripts/main.py --mode run-experiment --task CLEVR --model VCML --experiment debiasing_leaked --name debiasing_leaked_20 --num_parallel {num} --log_dir {log_dir} --data_dir {data_dir} --penalty 0.1
```

### *Hypernym* Supports Few-Shot Learning Concepts

```
jac-crun <GPU-id> scripts/main.py --mode run-experiment --task CUB --model VCML --experiment fewshot --num_parallel {num} --log_dir {log_dir} --data_dir {data_dir} --sample_size 50000 --dropout 0.9 --conceptual_weight 0.1
```

## Options

If you want to test the performance of pre-trained checkpoints, or to load and fine-tune it during training, you should add `--pretrained` option.

If you want to train other models, you are free to change the argument of option `--model` to any one of `NSCL`, `GRUCNN`, `GRU`, `BERT`, `BERTvariant`.

If you would like to only evaluate the model's performance on test data (e.g.
with pretrained checkpoints), you can add argument `--in_epoch test`, which changes the option from the default value `train val test` to only `test`.

If you would like to mute the output or logging files, you may add argument `--silent`.

After each epoch, the code will save a checkpoint. You may use `--resume`
option to resume from a previous checkpoint.

For CUB dataset, always set the argument `--sample_size 50000`. For GQA dataset, always set the argument `--sample_size 10000`.
