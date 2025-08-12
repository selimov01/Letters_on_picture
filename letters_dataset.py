import os
import torch
from PIL import Image
import torchvision.transforms as T

class LettersDataset(torch.utils.data.Dataset):
    
    def __init__(self, root, transforms=None):
        self.root = root
        self.imgs = sorted(os.listdir(os.path.join(root, "images")))
        self.ann_dir = os.path.join(root, "annotations")
        self.transforms = transforms

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.root, "images", img_name)
        img = Image.open(img_path).convert("RGB")
        w,h = img.size
        ann_path = os.path.join(self.ann_dir, img_name.replace(".png", ".txt"))
        boxes = []
        labels = []
        
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                
                for line in f:
                    line=line.strip()
                    if not line: continue
                    cls,x0,y0,x1,y1 = line.split()
                    labels.append(int(cls)+1)
                    boxes.append([float(x0), float(y0), float(x1), float(y1)])
                    
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:,3]-boxes[:,1]) * (boxes[:,2]-boxes[:,0]) if boxes.shape[0]>0 else torch.tensor([])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        if self.transforms:
            img = self.transforms(img)
        else:
            img = T.ToTensor()(img)
            
        return img, target

    def __len__(self):
        return len(self.imgs)