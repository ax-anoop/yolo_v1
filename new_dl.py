from data.datasets import voc
import visualize as v 
import torch

viz = v.Viz()
train_root = "/home/server/Desktop/pascal_voc/"
trainloader = voc.VOCDatasets(train_root, train=True)

img, lbl = next(iter(trainloader))
print(img.shape)
viz.show_image(img, format="yolo", boxes=lbl, saveimg=True)
# print(img.shape)
# print(lbl)
# trainloader = make_dist_voc_loader(os.path.join(train_root,'train.txt'),
#                                 img_size=448,
#                                 batch_size=1,
#                                 train=True,
#                                 )