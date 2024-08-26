import os
from torch.autograd import Variable
import sys
from dataset import LungDataset
from model import UNet
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import cv2
from coatention_model import CoattentionModel

seed=10
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def normalize(img):
    min=np.min(img)
    max=np.max(img)
    img=img-(min)
    img=img/(max-min)
    return img

class DiceScore(torch.nn.Module):
    """
    class to compute the Dice Loss
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, mask):
        # flatten label and prediction tensors
        pred = torch.flatten(pred)
        mask = torch.flatten(mask)

        counter = (pred * mask).sum()  # Counter
        denum = pred.sum() + mask.sum()  # denominator
        dice = (2 * counter) / denum

        return dice



# val_path = Path("/media/poorsoltani/C65CCF735CCF5D35/LungTumor-Segmentation/task06/preproceed/val")
# val_path = Path("../task06/preproceed/val")
val_path = Path("/media/poorsoltani/C65CCF735CCF5D35/LungTumor-Segmentation/razavi_hospital/v2_seed_small_dataset/data_resized_registered/val")
val_dataset = LungDataset(val_path, None)
# val_dataset = LungDataset(val_path, None,is_val=True,one_patient='3160145')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,  shuffle=False)

model=CoattentionModel().cuda()


model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

evaluate=open('./logs/evaluate_10_former_later.txt','a')
des_addr = './logs/seg_result/'

loss_fn = torch.nn.BCEWithLogitsLoss()
max_dice=0
min_BCE=100
THRESHOLD = 0.5
for e in range(15,28):
    cnt = 0
    BCE_log=[]
    weights = torch.load('./logs/epoch_'+str(e)+'.pth', map_location='cuda')['model']
    model.load_state_dict(weights)
    preds = []
    labels = []
    step = 0
    for img, label in tqdm(val_loader):
        step += 1
        with torch.no_grad():
            node0 = torch.tensor(img[0]).cuda()
            node1 = torch.tensor(img[1]).cuda()
            node2 = torch.tensor(img[2]).cuda()

            label0 = torch.tensor(label[0]).cuda()
            label1 = torch.tensor(label[1]).cuda()
            label2 = torch.tensor(label[2]).cuda()

            pred0, pred1, pred2 = model(node0, node1, node2)
            loss0=loss_fn(pred0,label0)

            pred = pred0
            BCE_log.append(loss0.data.cpu().numpy())
        preds.append(pred.cpu().numpy())
        pred = pred > 0.5
        pred = (pred.cpu().squeeze().numpy().astype('uint'))
        labels.append(label[0].cpu().numpy())

    preds = np.array(preds)
    labels = np.array(labels)
    dice_score = DiceScore()(torch.from_numpy(preds), torch.from_numpy(labels).unsqueeze(0).float())
    mean_BCE = np.mean(BCE_log)
    evaluate.write('epoch= {}\t dice_score= {} mean bce={}\n'.format(e, dice_score, mean_BCE))
    evaluate.flush()
    if dice_score>max_dice:
        max_dice=dice_score
    if mean_BCE < min_BCE:
        min_BCE = mean_BCE

    # print(f"The Val Dice Score is: {dice_score}")

evaluate.write('max dice is= {}\t min BCE is= {}\n'.format(max_dice,min_BCE))
evaluate.flush()


