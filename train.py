import torch, torchvision
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader, random_split
from letters_dataset import LettersDataset


def collate_fn(batch):
    return tuple(zip(*batch))


root = "Letters_on_picture"
dataset = LettersDataset(root, transforms=None)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_fn)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 26 + 1 
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, targets in train_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        running_loss += losses.item()
        
    lr_scheduler.step()
    
    print(f"Epoch {epoch} loss {running_loss/len(train_loader):.4f}")

print("Done training. Save model.")
torch.save(model.state_dict(), "fasterrcnn_letters.pth")