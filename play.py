# def func(**kwargs):
#     print("Hello World" , type(kwargs), kwargs)
#     print(add(**kwargs))

# def add(a, b, c=0):
#     return a + b

# args = {'a': 1, 'b': 2}

# print(add(**args))

from torchvision import transforms
import torch

import dataset as dataset_creator
import visualize
import model
import loss 
import numpy as np
from legacy import loss as legacy_loss
from legacy import utils as legacy_utils
from torch.utils.data import DataLoader
import utils 


# torch.manual_seed(16) # for reproducibility

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

def get_data():
    BATCH = 8
    dataset = dataset_creator.VOC("/home/server/Desktop/pascal_voc", transform=transform, train=True)
    train_loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True)
    return train_loader

train_loader = get_data()
model = model.create()
viz = visualize.Viz()
loss_fn = loss.YoloLoss()
loss_fn_legacy = legacy_loss.YoloLoss()

# Inference test #
utils.load(model, filename="my_checkpoint.pth.tar")
img, lbl = next(iter(train_loader))
out = model(img)

l1 = loss_fn(out, lbl)
l2 = loss_fn_legacy(out, lbl)

print(l1.item(), l2.item())
# # Training test #
# for i in range(100):
#     img, lbl = dataset.__getitem__(1)
#     img, lbl = img.unsqueeze(0), lbl.unsqueeze(0)
#     out = model(img)
#     l = loss(out, lbl)
#     model.zero_grad()
#     l.backward()
#     for p in model.parameters():
#         p.data -= p.grad.data * 0.00001
#     print(l, torch.abs(lbl.flatten().unsqueeze(0)-out).sum())
# torch.save(model.state_dict(), "test_model.pth")
# Random tests & stuff
# i = 0
# print(out[0][20:30])
# print(out.reshape(-1, 49, 30)[0][i][20:])
# boxes = legacy_utils.cellboxes_to_boxes(out)
# out = out.detach()
# out = utils.cell_to_boxes(out)
# mbx = utils.remove_low_conf(out).numpy()
# print(mbx.shape)
# nms_boxes = legacy_utils.non_max_suppression(list(mbx[0]), 0.5, 0.4)
# print(nms_boxes)