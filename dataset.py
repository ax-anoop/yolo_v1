from cProfile import label
import torch 
import pandas as pd 
from PIL import Image
import visualize as viz
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2 as cv 
import torch.nn.functional as F
import random 

class VOC(torch.utils.data.Dataset):
    def __init__(self, dir, num_clases=20, C=7, B=2, train=True):
        self.dir = dir
        self.C = C 
        self.B = B
        self.num_clases = num_clases
        self.train = train
        self.csv = dir+"/test.csv"
        if self.train:
            self.csv = dir+"/train.csv"
        self.data = self._load_csv()        
        self.to_tensor = transforms.ToTensor()
        self.size = (448,448)
        self.augmentations = [self.random_flip, self.random_scale, self.random_saturation, self.random_hue, self.random_brightness, self.random_blur]
        mean_rgb = [122.67891434, 116.66876762, 104.00698793]
        self.mean = np.array(mean_rgb, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        boxes = [] # [class, mid_x, mid_y, width, height]
        label_matrix = torch.zeros((self.C,self.C,self.num_clases+5*self.B))
        imgdir = self.data.iloc[[idx]].img.item()
        lbldir = self.data.iloc[[idx]].label.item()

        # img = Image.open(self.dir+"/images/"+imgdir)
        img = cv.imread(self.dir+"/images/"+imgdir)
        lbl = open(self.dir+"/labels/"+lbldir)
        for line in lbl.readlines():
            line = line.split()
            box = list(map(float, line))
            boxes.append(box)
            
        boxes = torch.tensor(boxes)
        if self.train:
            for aug in self.augmentations:
                img, boxes = aug(img, boxes)
        for box in boxes:
            i, j = self.C*box[1], self.C*box[2]
            c  = torch.nn.functional.one_hot(torch.tensor([int(box[0])]), self.num_clases)
            y  = torch.tensor([1,i%1,j%1,self.C*box[3],self.C*box[4]])
            lm = torch.cat((c[0],y))
            label_matrix[int(j),int(i),:25] = lm
        img = cv.resize(img, dsize=self.size, interpolation=cv.INTER_LINEAR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = (img - self.mean) / 255.0 # normalize from -1.0 to 1.0.
        img = self.to_tensor(img)
        return img, label_matrix

    def _load_csv(self):
        df = pd.read_csv(self.csv,names=["img","label"],header=None)
        return df 

    def random_flip(self, img, boxes):
        if random.random() < 0.5:
            return img, boxes  
        img = np.fliplr(img)
        boxes[:, 1] = 0.5 - (boxes[:, 1]-0.5)
        return img, boxes

    def random_scale(self, img, boxes):
        if random.random() < 0.5:
            return img, boxes
        scalex = random.uniform(0.8, 1.2)
        scaley = 1 #random.uniform(0.8, 1.2)
        h, w, _ = img.shape
        img = cv.resize(img, dsize=(int(w * scalex), int(h*scaley)), interpolation=cv.INTER_LINEAR)
        return img, boxes

    def random_blur(self, bgr, boxes):
        if random.random() < 0.5:
            return bgr, boxes
        ksize = random.choice([2, 3, 4, 5])
        bgr = cv.blur(bgr, (ksize, ksize))
        return bgr, boxes

    def random_brightness(self, bgr, boxes):
        if random.random() < 0.5:
            return bgr, boxes
        hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        adjust = random.uniform(0.5, 1.5)
        v = v * adjust
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        hsv = cv.merge((h, s, v))
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        return bgr, boxes

    def random_hue(self, bgr, boxes):
        if random.random() < 0.5:
            return bgr, boxes
        hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        adjust = random.uniform(0.8, 1.2)
        h = h * adjust
        h = np.clip(h, 0, 255).astype(hsv.dtype)
        hsv = cv.merge((h, s, v))
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        return bgr, boxes

    def random_saturation(self, bgr, boxes):
        if random.random() < 0.5:
            return bgr, boxes
        hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        adjust = random.uniform(0.5, 1.5)
        s = s * adjust
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        hsv = cv.merge((h, s, v))
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        return bgr, boxes
    
def main():
    dataset = VOC("/home/server/Desktop/pascal_voc", train=True)
    img, lbl = dataset.__getitem__(6)
    print(img.shape)
    # v = viz.Viz()
    # v.show(img, format="yolo", boxes = lbl)
    # print(lbl)

if __name__ == "__main__":
    main()
