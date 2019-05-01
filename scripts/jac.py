In [12]: batch['objects_length']
Out[12]: tensor([10, 10, 10, 10, 10, 10, 10, 10, 10, 10,  9,  9,  9,  9,  9,  9])

In [13]: batch['objects'].shape
Out[13]: torch.Size([154, 4])

In [14]: batch['image'].shape
Out[14]: torch.Size([16, 3, 256, 384])

a = model.resnet(batch['image'].cuda())
b = model.scene_graph(a, batch['objects'].cuda(), batch['objects_length'].cuda())

len(b) == 16
len(b[0]) == 3
b[0][0] == None
b[0][1].shape = [10, 256]
b[0][2].shape = [10, 10, 256]
