import torch
import torch.nn as nn 
import utils 


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
    
    def forward(self, predictions, target):
        batch_size = predictions.size(0)
        box_loss, obj_loss, no_obj_loss, class_loss = 0, 0, 0, 0

        p = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)        
        obj_mask = target.sum(dim=3).clamp(0, 1) # create mask of where targets exist
        noobj_mask = 1 - obj_mask # create mask of where targets do not exist

        p_objs  = torch.mul(p,obj_mask.unsqueeze(-1)) # get predictions where targets exist
        p_nobjs = torch.mul(p,noobj_mask.unsqueeze(-1)) # get predictions where targets exist
        
        p_classes, t_classes= p_objs[:,:,:,:self.C], target[:,:,:,:self.C] # get class predictions    
        p_no_classes = p_nobjs[:,:,:,:self.C] # get class predictions


        b0 = target[:,:,:,self.C+1:self.C+5].reshape(-1, 4) # target boxes
        nz = torch.nonzero(b0.sum(dim=1)) # get indices of non-zero elements
        z = torch.nonzero(b0.sum(dim=1) == 0) # get indices of zero elements
                
        b1 = p_objs[:,:,:,self.C+1:self.C+5].reshape(-1, 4)
        b2 = p_objs[:,:,:,self.C+6::].reshape(-1, 4)

        b0 = b0[nz].reshape(-1,4)
        b1 = b1[nz].reshape(-1,4)
        b2 = b2[nz].reshape(-1,4)
        
        ious1 = utils.iou_torch(b0, b1) 
        ious2 = utils.iou_torch(b0, b2) 
        ious = torch.cat([ious1.unsqueeze(0), ious2.unsqueeze(0)], dim=0)
        iou_maxes, iou_ids = torch.max(ious, dim=0) # get max ious and their indices

        b1_iou_mask = (iou_ids == 0).float().unsqueeze(-1) # create mask for boxes with max iou
        b2_iou_mask = (iou_ids == 1).float().unsqueeze(-1) # create mask for boxes with max iou
        
        # correct boxes and targets after IOU comparison
        b0_1 = torch.mul(b0,b1_iou_mask)
        b0_2 = torch.mul(b0,b2_iou_mask)
        b1 = torch.mul(b1,b1_iou_mask)
        b2 = torch.mul(b2,b2_iou_mask)

        # calculate box loss (x, y, w, h)
        b1_bloss_xy = ((b0_1[:,0] - b1[:,0])**2) + ((b0_1[:,1] - b1[:,1])**2)
        b2_bloss_xy = ((b0_2[:,0] - b2[:,0])**2) + ((b0_2[:,1] - b2[:,1])**2)
        b1s, b2s = torch.sign(b1), torch.sign(b2)
        b1abs, b2abs  = torch.abs(b1+1e-6), torch.abs(b2+1e-6)
        b1_bloss_wh = ((b1s[:,2]*torch.sqrt(b0_1[:,2]) - b1s[:,2]*torch.sqrt(b1abs[:,2]))**2)  + (((b0_1[:,3]**.5) - b1s[:,3]*(b1abs[:,3]**.5))**2)
        b2_bloss_wh = (((b0_2[:,2]**.5) - b2s[:,2]*(b2abs[:,2]**.5))**2) + (((b0_2[:,3]**.5) - b2s[:,3]*(b2abs[:,3]**.5))**2)
        box_loss = b1_bloss_xy.sum() + b2_bloss_xy.sum()  #+ b2_bloss_wh.sum()
        box_loss += b1_bloss_wh.sum() + b2_bloss_wh.sum()
        
        # calculate has object loss
        c1 = p_objs[:,:,:,self.C:self.C+1].reshape(-1, 1)[nz].reshape(-1,1)
        c2 = p_objs[:,:,:,self.C+5:self.C+6].reshape(-1, 1)[nz].reshape(-1,1)
        yc1 = torch.mul(target[:,:,:,self.C:self.C+1].reshape(-1, 1)[nz].reshape(-1,1), b1_iou_mask)
        yc2 = torch.mul(target[:,:,:,self.C:self.C+1].reshape(-1, 1)[nz].reshape(-1,1), b2_iou_mask)
        obj_loss = (yc1-torch.mul(c1, b1_iou_mask))**2 + (yc2-torch.mul(c2, b2_iou_mask))**2
        obj_loss = obj_loss.sum()

        # calculate no object loss
        c1 = p_nobjs[:,:,:,self.C:self.C+1].reshape(-1, 1)[z].reshape(-1,1)
        c2 = p_nobjs[:,:,:,self.C+5:self.C+6].reshape(-1, 1)[z].reshape(-1,1)
        no_obj_loss = c1**2 + c2**2
        no_obj_loss = no_obj_loss.sum()

        # calculate class loss
        class_loss = ((t_classes-p_classes)**2).sum()

        total_loss = self.lambda_coord*box_loss + obj_loss + self.lambda_noobj*no_obj_loss + class_loss
        return total_loss, torch.tensor([box_loss.item(), obj_loss.item(), no_obj_loss.item(), class_loss.item()])


def main():
    b1 = torch.tensor([[0.5, 0.5, 0.1, 0.1], [0.2, 0.2, 0.1, 0.1]])
    b2 = torch.tensor([[0.55, 0.55, 0.1, 0.1], [0.29, 0.29, 0.1, 0.1]])

    print(intersection_over_union(b1, b2, box_format="midpoint"))
    print(utils.iou_torch(b1, b2))

if __name__ == "__main__":
    main()

# tmp_loss_a, tmp_loss_b = 0, 0
# for i in range(len(iou_ids)):
#     if iou_ids[i] == 0:
#         pred = b1[i]
#     else:
#         pred = b2[i]
#     yvals = b0[i]
#     tmp_loss_a += ((yvals[0] - pred[0])**2) + ((yvals[1] - pred[1])**2)

#     w, h, sw, sh = torch.abs(pred[2])+ 1e-6, torch.abs(pred[3])+ 1e-6, torch.sign(pred[2]), torch.sign(pred[3])
#     tmp_loss_b += ((yvals[2]**.5 - sw*(w**.5))**2) + ((yvals[3]**.5 - sh*(h**.5))**2)
# tml_ab = tmp_loss_a + tmp_loss_b
        