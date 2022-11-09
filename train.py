from matplotlib import test
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.optim as optim

import model as model_creator
import dataset as dataset_creator
import loss
import utils 
import time

from legacy import loss as legacy_loss 
from legacy import utils as legacy_utils

LR = 2e-5 # learning rate
BATCH = 8 # 64 in original paper but I don't have that much vram, grad accum?
WD = 0 # weight decay
WORKERS = 10 # number of workers for dataloader
EPOCHS = 1000 # number of epochs to train for
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("USING:",DEVICE)

def get_data(trainval=0.995):
    train_set = dataset_creator.VOC("/home/server/Desktop/pascal_voc", train=True) # Random split
    valid_set = dataset_creator.VOC("/home/server/Desktop/pascal_voc", train=False) # Random split
    # train_set_size = int(len(train_set) * trainval)
    # valid_set_size = len(train_set) - train_set_size
    # train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size])
    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=WORKERS, drop_last=True)
    val_loader = DataLoader(valid_set, batch_size=BATCH, shuffle=True, num_workers=WORKERS, drop_last=True)
    return train_loader, val_loader

def map_test(val_loader, samples, model, loss_fn):
    model.eval()
    x, y = next(iter(val_loader))
    x, y = x.to(DEVICE), y.to(DEVICE)
    o = model(x)
    l, m_l = loss_fn(o,y)
    pred_boxes, target_boxes = legacy_utils.get_bboxes(val_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE, samples=samples)
    mean_avg_prec = legacy_utils.mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
    model.train()
    return mean_avg_prec.item(), l.item(), m_l

# add plot values, [["name", value]]
def tb_add_scalar(tb, names, values, i_ter, printvals=False, topic=None):
    if len(names) != len(values):
        print("invalid dims for tensorboard log")
        return
    pval = ''
    for i in range(len(names)):
        if topic:
            tb.add_scalar(topic+"/"+names[i], values[i], i_ter)
        else:
            tb.add_scalar(names[i], values[i], n)
        if printvals:
            pval += names[i] + ":" + str(round(values[i],4)) + " | "
    if printvals:
        print('\n',pval)

def init_tensorboard(tb, model, in_shape=[1,3,448,448]):
    tb.add_graph(model, torch.randn(in_shape))
    tb.close()

def train(model, optimizer, loss_fn, train_loader, val_loader, check_pt_name, save_model=True):
    tb = SummaryWriter()
    init_tensorboard(tb, model)

    model.to(DEVICE)
    loop = tqdm(train_loader, leave=True)
    avg_loss, c, meta_sum = [], 0, torch.tensor([0,0,0,0]).type(torch.float)
    for epoch in range(EPOCHS):
        for i, (x, y) in enumerate(loop):
            x,y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss, meta = loss_fn(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss.append(loss.item())
            meta_sum += meta

            if len(avg_loss)>100:
                l_en = len(avg_loss)
                m, tloss, m_l = map_test(val_loader, 128, model, loss_fn)
                l = round(sum(avg_loss)/l_en,3)
                #add to tensorboard

                tb_add_scalar(tb, ["Loss","T-loss", "mAP", "Epoch"], [l, tloss, m, epoch], c, printvals=True,topic="resnet50")
                tb_add_scalar(tb, ["box", "obj", "nobj", "class"], meta_sum/l_en, c, printvals=False,topic="train-meta")
                tb_add_scalar(tb, ["box", "obj", "nobj", "class"], m_l, c, printvals=False,topic="test-meta")
                c += 1
                avg_loss = []
                meta_sum = torch.tensor([0,0,0,0]).type(torch.float)
        if save_model:
            utils.save(model,filename= check_pt_name)

# Epoch: 95 | Batch: 326 | Loss: 4.602 | T-Loss: 126.04718017578125 | mAP:0.07954848557710648                                                         │
def main():
    check_pt_name = "my_checkpoint.pth.tar"
    load_model = False
    
    model = model_creator.create(backbone="vgg16")
    if load_model:
        utils.load(model,filename=check_pt_name)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = loss.YoloLoss()
    # loss_fn = legacy_loss.YoloLoss()
    train_loader, val_loader = get_data()
    train(model, optimizer, loss_fn, train_loader, val_loader, check_pt_name, save_model=False)

if __name__ == "__main__":
    main()