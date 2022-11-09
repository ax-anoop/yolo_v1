import torch
import time
from torch.utils.data import DataLoader
import model as model_creator 
import dataset as dataset_creator 
from torchvision import transforms
import numpy as np

import utils 
import visualize
from legacy import utils as legacy_utils

DEVICE = "cuda:1"
torch.manual_seed(0)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

def get_data(BATCH, train=True):
    dataset = dataset_creator.VOC("/home/server/Desktop/pascal_voc", transform=transform, train=train)
    train_loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=1, drop_last=True)
    return train_loader

def shape(x):
    print(np.array(x).shape)
    # return np.array(x).shape

def main():
    viz = visualize.Viz()
    train_loader = get_data(1, train=False)
    model = model_creator.create()
    model.load_state_dict(torch.load("my_checkpoint.pth.tar",map_location=torch.device('cpu')))
    model = model.to(DEVICE)

    pred_boxes, target_boxes = legacy_utils.get_bboxes(
        train_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE
    )

    mean_avg_prec = legacy_utils.mean_average_precision(
        pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
    )
    print(mean_avg_prec)
    # img, lbl = next(iter(train_loader))
    # targets = lbl.clone()
    # img.to(DEVICE)
    # pred = model(img)
    # boxesp = utils.cellboxes_to_boxes(pred)
    # boxest = utils.cellboxes_to_boxes(lbl)
    # #[class_pred, prob_score, x1, y1, x2, y2]
    # nms_boxes = legacy_utils.non_max_suppression(
    #             boxesp[0],
    #             iou_threshold=0.5,
    #             threshold=0.4,
    #             box_format="midpoint",
    #         )
    # rpred = pred.detach().reshape(1,7,7,-1)
    # print(lbl.sum(dim=3))
    # viz.show_image(img, format="mid", boxes=[[0.5,0.5,0.55,0.55]])

if __name__ == "__main__":
    main()
