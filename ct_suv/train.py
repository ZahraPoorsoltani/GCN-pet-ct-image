import numpy as np
import torch
import os

from unet import UNet
from torch.autograd import Variable
import imgaug.augmenters as iaa
from pathlib import Path
from dataset import LungDataset
from tqdm import tqdm

seed = 10
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

seq = iaa.Sequential([
    iaa.Affine(rotate=[90, 180, 270]),  # rotate up to 45 degrees
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
])

train_path = Path("/media/fumcomp/9CAA7029AA6FFDD8/fold_0/project/data/train")
train_dataset = LungDataset(train_path, seq)

# target_list = []
# for _, label in tqdm(train_dataset):
#     # Check if mask contains a tumorous pixel:
#     if np.any(label):
#         target_list.append(1)
#         label_squeeze=label[0,:,:]
#     else:
#         target_list.append(0)
# np.save('./target_list.npy',target_list)

target_list = np.load('/media/fumcomp/9CAA7029AA6FFDD8/fold_0/project/target_list.npy')
uniques = np.unique(target_list, return_counts=True)
fraction = uniques[1][0] / uniques[1][1]

weight_list = []
from tqdm import tqdm

for target in tqdm(target_list):
    if target == 0:
        weight_list.append(1)
    else:
        weight_list.append(fraction)

sampler = torch.utils.data.sampler.WeightedRandomSampler(weight_list, len(weight_list))
batch_size = 8  # TODO
num_workers = 2  # TODO
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

model = UNet().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.BCEWithLogitsLoss()
logger = open('./logs/logs.txt', 'a')
resume = 0
if resume:
    state_model = torch.load('/content/drive/MyDrive/razavi_ct_suv/logs/epoch_' + str(resume - 1) + '.pth')
    model.load_state_dict(state_model['model'])
    optimizer.load_state_dict(state_model['optimizer'])
EPOCHE = 100
loss_report = 0
for i in range(resume, EPOCHE):
    batch_losses = []
    step_100_loss = []
    cnt = 1
    # progress_bar=tqdm(enumerate(train_loader), total=(len(train_loader.batch_sampler)))
    for step, (img, label) in tqdm(enumerate(train_loader), total=(len(train_loader.batch_sampler))):
        img.requires_grad_()
        img = Variable(img).cuda()

        label.requires_grad_()
        label = Variable(label).cuda()

        pred = model(img)

        loss_w = loss_fn(pred, label)
        optimizer.zero_grad()  # (reset gradients)
        loss_w.backward()  # (compute gradients)
        optimizer.step()

        loss_value = loss_w.data.cpu().numpy()
        batch_losses.append(loss_value)
        step_100_loss.append(loss_value)
        if not (cnt % 100):
            logger.write('step= {}\t mean loss={}\n'.format(step, np.mean(batch_losses)))
            logger.flush()

        cnt += 1

    logger.write('###########################################################\n')
    logger.write('epoche= {}\t mean loss={}\n'.format(i, np.mean(batch_losses)))
    logger.write('###########################################################\n')
    # torch.save(model.state_dict(),'./logs/epoch_'+str(i)+'.pth')
    torch.save({
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
    }, '/home/fumcomp/Desktop/poorsoltani/ct_suv/epoch_' + str(i) + '.pth')
    print('##########################################')
    print('################### epoch {} ############'.format(i))
    print('##########################################')
