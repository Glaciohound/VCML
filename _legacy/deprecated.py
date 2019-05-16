def load_gqa_sceneGraphs():
    pass

    '''
    SG_h5 = h5.File(args.sceneGraph_h5, 'r')
    splits = SG_h5['split'][:]
    image_ids = SG_h5['img_ids'][:].astype('U').tolist()
    filenames = [os.path.join(args.image_dir, filename + '.jpg')
                    for filename in image_ids]
    index = np.arange(splits.shape[0])
    split_indexes = {
        k: np.array([i for i in range(splits.shape[0])
                        if splits[i] == v])
        for k, v in {'train': 0, 'val': 1, 'test': 2}.items()
    }

    obj_ranges = SG_h5['obj_ranges'][:]
    rel_ranges = SG_h5['rel_ranges'][:]

    # loading box information
    gt_classes = SG_h5['labels'][:, 0]
    gt_boxes = SG_h5['boxes_{}'.format(args.box_scale)][:].astype(np.float32)  # will index later
    gt_attributes = SG_h5['attributes'][:]
    # convert from xc, yc, w, h to x1, y1, x2, y2
    gt_boxes[:, :2] = gt_boxes[:, :2] - gt_boxes[:, 2:] / 2
    gt_boxes[:, 2:] = gt_boxes[:, :2] + gt_boxes[:, 2:]

    # load relation labels
    gt_relations = SG_h5['relations'][:]

    SG_h5.close()
    sceneGraphs = {
        'splits': splits,
        'image_ids': image_ids,
        'filenames': filenames,
        'split_indexes': split_indexes,
        'obj_ranges': obj_ranges,
        'rel_ranges': rel_ranges,
        'gt_classes': gt_classes,
        'gt_boxes': gt_boxes,
        'gt_attributes': gt_attributes,
        'gt_relations': gt_relations,
    }
    '''

def gqa_getitem():
    pass
    '''
    image_unpadded = Image.open(self.filenames[index]).convert('RGB')

    w, h = image_unpadded.size
    img_scale_factor = args.image_scale / max(w, h)
    if h > w:
        im_size = (args.image_scale, int(w * img_scale_factor), img_scale_factor)
    elif h < w:
        im_size = (int(h * img_scale_factor), args.image_scale, img_scale_factor)
    else:
        im_size = (args.image_scale, args.image_scale, img_scale_factor)

    gt_boxes, gt_classes, gt_attributes = \
        [e[self.obj_ranges[index, 0]: self.obj_ranges[index, 1]]
            for e in [self.gt_boxes, self.gt_classes, self.gt_attributes]]
    gt_rels = self.gt_relations \
        [self.rel_ranges[index, 0]: self.rel_ranges[index, 1]]

    entry = {
        'index': index,
        'img_size': im_size,
        'img': self.transform_pipeline(image_unpadded),

        'gt_boxes': gt_boxes,
        'gt_classes': gt_classes,
        'gt_attributes': gt_attributes,
        'gt_relations': gt_rels,

        'scale': args.image_scale / args.box_scale,
    }

    return entry
    '''
