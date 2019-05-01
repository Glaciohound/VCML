PYTHONPATH=./ python tools/run_test.py\
    --run_dir ../../data/attr_net/results \
    --dataset clevr \
    --load_checkpoint_path ../../../data/gqa/checkpoints/pretrained/nsvqa/attribute_net.pt \
    --clevr_val_ann_path ../../../data/clevr/attr_net/objects/clevr_train_objs_pretrained.json \
    --output_path ../../../data/clevr/attr_net/results/clevr_train_scenes_parsed_pretrained.json  --clevr_val_img_dir ../../../data/clevr/raw/CLEVR_v1.0/images/train --split train
