import torch 

def _corner_to_center(box):
    return [box[0], box[1], box[2]-box[0], box[3]-box[1]]

def _center_to_corners(box):
    return [box[0]-box[2]/2, box[1]-box[3]/2, box[0]+box[2]/2, box[1]+box[3]/2]

def _segment_intersection(a1, a2, a3, a4):
    ''' calculate intersection of two segments [x1,x2] & [x3,x4] '''
    if a1 > a2 or a3 > a4:
        raise ValueError("Invalid segment")
    if a2 < a3 or a4 < a1:
        return 0
    return min(a2,a4) - max(a1,a3)

def box_conversion(boxes, current_format="midpoints", target_format="corners"):
    n_boxes = []
    if current_format == target_format:
        return boxes
    if current_format == "midpoints" and target_format == "corners":
        for box in boxes:
            n_boxes.append(_center_to_corners(box))
    elif current_format == "corners" and target_format == "midpoints":
        for box in boxes:
            n_boxes.append(_corner_to_center(box))
    else:
        raise ValueError("Invalid formats")
    return n_boxes

def calc_area(box):
    ''' format assumed is corners '''
    x1, y1, x2, y2 = box
    return (x2-x1)*(y2-y1)
    
def calc_intersection(box1, box2):
    ''' format assumed is corners '''
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    y_inc = _segment_intersection(y1, y2, y3, y4)
    x_inc = _segment_intersection(x1, x2, x3, x4)
    return x_inc*y_inc

def box_iou(box1, box2, format="midpoints"):
    '''
    - Box formats: 
        > corners   = [x,y,x,y]
        > midpoints = [x,y,w,h]
    - Calculate intersection over union of two boxes
    '''
    if format == "midpoints":
        box1 = _center_to_corners(box1)
        box2 = _center_to_corners(box2)
    intersection = calc_intersection(box1, box2)
    if intersection == 0:
        return 0
    area = calc_area(box1) + calc_area(box2)
    return intersection / (area - intersection + 1e-6)

def iou_torch(box1, box2, format="midpoints", copy=True):
    if copy:
        box1 = box1.clone()
        box2 = box2.clone()
    if format == "midpoints":
        box1 = mid_to_corners_torch(box1)
        box2 = mid_to_corners_torch(box2)
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    return intersection / (box1_area + box2_area - intersection + 1e-6)

def corners_to_mid_torch(boxes, clone=False):
    b = boxes.clone()
    x1, y1, x2, y2 = b[:,0], b[:,1], b[:,2], b[:,3]
    boxes[:,0] = (x1 + x2) / 2
    boxes[:,1] = (y1 + y2) / 2
    boxes[:,2] = x2 - x1
    boxes[:,3] = y2 - y1
    if clone:
        return boxes.clone()
    return boxes

def mid_to_corners_torch(boxes, clone=False):
    b = boxes.clone()
    x1, y1, w, h = b[:,0], b[:,1], b[:,2], b[:,3]
    boxes[:,0] = x1 - w/2
    boxes[:,1] = y1 - h/2
    boxes[:,2] = x1 + w/2
    boxes[:,3] = y1 + h/2
    if clone:
        boxes = boxes.clone()
    return boxes

def remove_low_conf(preds):
    ''' remove boxes with low confidence & turn onehot class vecotrs into class labels
    - pred [-1,49,30]: (C1..CN, PC1, X1, X2, W, H, PC2, X1, X2, W, H)
    - out  [-1,49,30]: (C1..CN, PC1, X1, X2, W, H)
    '''
    classes = torch.argmax(preds[:, :, 0:20], axis=2).unsqueeze(-1)
    b1c = preds[:, :,20:25]
    b2c = preds[:, :,25:]
    confs = (b1c[:,:,0] > b2c[:,:,0]).unsqueeze(-1)
    b1c *= confs
    b2c *= torch.logical_not(confs)
    cboxes = b1c + b2c
    # torch.cat((classes, cboxes), axis=2).shape
    return torch.cat((classes, cboxes), axis=2)

def cell_to_boxes(pred, S=7, C=20, B=2):
    '''
    Transform boxes in cell space to box space.
    - pred: (C1..CN, PC1, X1, X2, W, H, PC2, X1, X2, W, H)
    - out : (C1..CN, PC1, X1, X2, W, H, PC2, X1, X2, W, H)
    '''
    batch_size = pred.size(0)
    b = pred.clone().reshape(batch_size, S, S, -1)
    for i in range(S):
        for j in range(S):
            #box 1
            b[:,i,j,C+1] = (b[:,i,j,C+1]+j)/S
            b[:,i,j,C+2] = (b[:,i,j,C+2]+i)/S
            b[:,i,j,C+3] /= S
            b[:,i,j,C+4] /= S
            #box 2
            b[:,i,j,C+6] = (b[:,i,j,C+6]+j)/S
            b[:,i,j,C+7] = (b[:,i,j,C+7]+i)/S
            b[:,i,j,C+8] /= S
            b[:,i,j,C+9] /= S
    return b.reshape(batch_size, S*S, -1)

def save(model, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(model.state_dict(), filename)

def load(model, optimizer=None, filename="my_checkpoint.pth.tar"):
    print("=> Loading checkpoint")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])

def summary(model, input=(3,448,448)):
    import torchsummary
    model = model.cuda()
    torchsummary.summary(model, input)

def main():
    # boxes = [[0.45,0.45,0.55,0.55]]
    # print("corners > midpts:",box_conversion(boxes, current_format="corners", target_format="midpoints"))
    # boxes = [[0.5,0.5,0.1,0.1]]
    # print("midpts  > corners:",box_conversion(boxes, current_format="midpoints", target_format="corners"))
   
    # a = [0.45,0.45,0.55,0.55]
    # b = [0.54,0.54,0.6,0.6]
    # print(box_iou(a,b,format="corners"))

    # a = torch.tensor([[0.45,0.45,0.55,0.55],[0.45,0.45,0.55,0.55]])
    # b = torch.tensor([[0.54,0.54,0.6,0.6],[0.54,0.54,0.6,0.6]])
    # print(iou_torch(a,b))

    m = torch.tensor([[0.5,0.5,0.1,0.1]])
    boxesC = mid_to_corners_torch(m, clone=True)
    print(boxesC)

    c = torch.tensor([[0.45,0.45,0.55,0.55]])
    boxesM = corners_to_mid_torch(c, clone=True)
    print(boxesM)

if __name__ == '__main__':
    main()














# def iou_calc(box1, box2, box_format="midpoint"):
    # box1 = (x1, y1, x2, y2)
    # box2 = (x1, y1, x2, y2)

# def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
#     """
#     Calculates intersection over union
#     Parameters:
#         boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
#         boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
#         box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
#     Returns:
#         tensor: Intersection over union for all examples
#     """
#     if box_format == "midpoint":
#         box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
#         box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
#         box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
#         box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
#         box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
#         box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
#         box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
#         box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

#     if box_format == "corners":
#         box1_x1 = boxes_preds[..., 0:1]
#         box1_y1 = boxes_preds[..., 1:2]
#         box1_x2 = boxes_preds[..., 2:3]
#         box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
#         box2_x1 = boxes_labels[..., 0:1]
#         box2_y1 = boxes_labels[..., 1:2]
#         box2_x2 = boxes_labels[..., 2:3]
#         box2_y2 = boxes_labels[..., 3:4]

#     x1 = torch.max(box1_x1, box2_x1)
#     y1 = torch.max(box1_y1, box2_y1)
#     x2 = torch.min(box1_x2, box2_x2)
#     y2 = torch.min(box1_y2, box2_y2)

#     # .clamp(0) is for the case when they do not intersect
#     intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

#     box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
#     box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

#     return intersection / (box1_area + box2_area - intersection + 1e-6)