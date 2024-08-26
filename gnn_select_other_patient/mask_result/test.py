import os
from torch.autograd import Variable
import sys

sys.path.append('/media/fumcomp/9CAA7029AA6FFDD8/fold_0/project/gnn_select_other_patient')
from dataset import LungDataset
from unet import UNet
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import cv2
from coatention_model import CoattentionModel

seed = 10
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def normalize(img):
    min = np.min(img)
    max = np.max(img)
    img = img - (min)
    img = img / (max - min)
    return img * 255


def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined


def normalize(img):
    min = np.min(img)
    max = np.max(img)
    img = img - (min)
    img = img / (max - min)
    return img


def outline(image, mask, color):
    mask = np.round(mask)
    yy, xx = np.nonzero(mask)
    for y, x in zip(yy, xx):
        if 0.0 < np.mean(mask[max(0, y - 1): y + 2, max(0, x - 1): x + 2]) < 1.0:
            image[max(0, y): y + 1, max(0, x): x + 1] = color
    return image


def gray2rgb(image):
    w, h = image.shape
    image += np.abs(np.min(image))
    image_max = np.abs(np.max(image))
    if image_max > 0:
        image /= image_max
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = image * 255
    return ret


def evaluate_metrics(gt, pred_mask):
    intersect = np.sum(pred_mask * gt)
    union = np.sum(pred_mask) + np.sum(gt) - intersect
    sum_gt = np.sum(gt)
    sum_pred = np.sum(pred_mask)
    total_sum = sum_pred + sum_gt
    xor = np.sum(gt == pred_mask)
    gt_invers = 1 - gt
    TN = np.sum(pred_mask * gt_invers)
    # dice = round(np.mean(2 * intersect / total_sum), 3)
    # iou = round(np.mean(intersect / union), 3)
    # acc = round(np.mean(xor / (union + xor - intersect)), 3)
    # recall = round(np.mean(intersect / sum_gt), 3)
    # precision = round(np.mean(intersect / sum_pred), 3)

    dice = np.mean(2 * intersect / total_sum)
    iou = np.mean(intersect / union)
    acc = np.mean(xor / (union + xor - intersect))
    recall = np.mean(intersect / sum_gt)
    precision = np.mean(intersect / sum_pred)

    return {'dice': dice, 'iou': iou, 'acc': acc, 'recall': recall, 'precision': precision}


model = CoattentionModel().cuda()

model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

evaluate = open('./logs/evaluate.txt', 'a')
des_addr = './logs/seg_result/'

max_dice = 0
min_BCE = 100
THRESHOLD = 0.5
val_patients_addr = '/media/fumcomp/9CAA7029AA6FFDD8/fold_0/project/data/val'
train_path = Path('/media/fumcomp/9CAA7029AA6FFDD8/fold_0/project/data/train')
patients = os.listdir(val_patients_addr)
dice_per_patients = []
patients_name = []
weights = torch.load('../logs/epoch_' + str(29) + '.pth', map_location='cuda')['model']
model.load_state_dict(weights)
for p in patients:
    p = '3160145'
    patients_name.append(p)

    patients_path = Path(os.path.join(val_patients_addr, p))
    val_dataset = LungDataset(patients_path, train_path, None)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

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
            pred = pred0

        pred = pred > 0.5
        pred = pred.squeeze().cpu().numpy()

        ct = node0.squeeze().cpu().numpy()[1]
        ct = normalize(ct) * 255

        if np.any(pred):
            pred = (pred * 255).astype(np.uint8)
            ct = node0.squeeze().cpu().numpy()[1]
            ct = normalize(ct) * 255
            label0 = label0.squeeze().cpu().numpy()
            ct = gray2rgb(ct)
            pred_overlay = overlay(ct, pred, (0, 0, 255), 0.5)
            gt_overlay = overlay(ct, label0, (0, 255, 0), 0.5)
            cv2.imwrite(str(step) + 'gt.jpg', gt_overlay)
            cv2.imwrite(str(step) + 'pred.jpg', pred_overlay)

        preds.append(pred)
        labels.append(label[0].squeeze().cpu().numpy())

    preds = np.array(preds)
    labels = np.array(labels)

    eval_metrics = evaluate_metrics(labels, preds)

    # mean_BCE = np.mean(BCE_log)
    # evaluate.write('################### epoch {} ###############################\n'.format(e))
    for key in eval_metrics:
        if key == 'dice':
            dice_per_patients.append(eval_metrics[key])
        evaluate.write('{}: {}\t'.format(key, eval_metrics[key]))
    evaluate.write('\n')
    evaluate.write('############################################################\n')
    evaluate.flush()

    evaluate.flush()
    if eval_metrics['dice'] > max_dice:
        max_dice = eval_metrics['dice']
# np.save('./dice_per_patient.npy',dice_per_patients)
# np.save('./patients_name.npy', patients_name)
