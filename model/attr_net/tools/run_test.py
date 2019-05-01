import os
import json
import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Plain',
                                     color_scheme='Linux', call_pdb=1)

from options import get_options
from datasets import get_dataloader
from model import get_model
import utils
from tqdm import tqdm

COMP_CAT_DICT_PATH = 'tools/clevr_comp_cat_dict.json'

opt = get_options('test')
test_loader = get_dataloader(opt, 'test')
model = get_model(opt)

if opt.use_cat_label:
    with open(COMP_CAT_DICT_PATH) as f:
        cat_dict = utils.invert_dict(json.load(f))

image_idxs = sorted(list(set(test_loader.dataset.img_ids)))
if opt.dataset == 'clevr':
    scenes = [{
        'image_index': i,
        'image_filename': 'CLEVR_%s_%06d.png' % (opt.split, i),
        'objects': []
    } for i in image_idxs]

count = 0
for data, _, idxs, cat_idxs in tqdm(test_loader):
    model.set_input(data)
    model.forward()
    pred, features = model.get_pred(get_feature=True)
    from IPython import embed; embed()
    for i in range(pred.shape[0]):
        if opt.dataset == 'clevr':
            img_id = idxs[i]
            obj = utils.get_attrs_clevr(pred[i])
            if opt.use_cat_label:
                cid = cat_idxs[i] if isinstance(cat_idxs[i], int) else cat_idxs[i].item()
                obj['color'], obj['material'], obj['shape'] = cat_dict[cid].split(' ')
            obj['features'] = features[i].tolist()
        scenes[img_id]['objects'].append(obj)
    count += idxs.size(0)

output = {
    'info': '%s derendered scene' % opt.dataset,
    'scenes': scenes,
}
print('| saving annotation file to %s' % opt.output_path)
utils.mkdirs(os.path.dirname(opt.output_path))
with open(opt.output_path, 'w') as fout:
    json.dump(output, fout)
