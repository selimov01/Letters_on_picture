import torch, torchvision
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 27
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load("fasterrcnn_letters.pth", map_location='cpu'))
model.eval()

def predict_image(img_path, conf_thresh=0.5):
    img = Image.open(img_path).convert("RGB")
    t = T.ToTensor()(img)
    
    with torch.no_grad():
        preds = model([t])[0]
        
    boxes = preds['boxes'].numpy()
    scores = preds['scores'].numpy()
    labels = preds['labels'].numpy()
    keep = scores >= conf_thresh
    boxes = boxes[keep]; labels = labels[keep]; scores = scores[keep]
    count = len(boxes)
    
    return count, boxes, labels, scores

if __name__ == "__main__":
    path = "dataset/images/000001.png"
    c, boxes, labels, scores = predict_image(path, 0.5)
    print("Found", c, "letters")
