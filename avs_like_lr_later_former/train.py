import numpy as np
import torch
import os
from coatention_model import CoattentionModel
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

def adjust_learning_rate(optimizer, i_iter, max_iter):
    new_lr = 0.00012 * ((1 - float(i_iter) / max_iter) ** (0.9))
    optimizer.param_groups[0]['lr'] = new_lr


def adjust_learning_rate(optimizer, i_iter, max_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = 0.00012 * ((1 - float(i_iter) / max_iter) ** (0.9))
    # optimizer.param_groups[0]['lr'] = lr
    if i_iter % 3 == 0:
        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = lr * 10
    else:
        optimizer.param_groups[0]['lr'] = 0.01 * lr
        optimizer.param_groups[1]['lr'] = lr * 10

    return lr


train_path = Path("/media/fumcomp/9CAA7029AA6FFDD8/fold_0/project/data/train")
train_dataset = LungDataset(train_path, seq)

target_list = np.load('./target_list.npy')
uniques = np.unique(target_list, return_counts=True)
fraction = uniques[1][0] / uniques[1][1]

weight_list = []
for target in tqdm(target_list):
    if target == 0:
        weight_list.append(1)
    else:
        weight_list.append(fraction)

sampler = torch.utils.data.sampler.WeightedRandomSampler(weight_list, len(weight_list))
batch_size = 4  # TODO
num_workers = 8  # TODO
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                           sampler=sampler)

model = CoattentionModel().cuda()
unet_pretrained = torch.load(
    "/media/fumcomp/9CAA7029AA6FFDD8/fold_0/project/ct_suv/logs/epoch_40.pth",
    map_location='cuda')
weights = unet_pretrained['model']
new_params = model.state_dict().copy()
for i in new_params:
    i_parts = i.split('.')
    if i_parts[0] == 'encoder':
        new_params[i] = weights[".".join(i_parts[1:])]
    elif i_parts[0] in ['upconv3', 'd31', 'd32', 'upconv4', 'd41', 'd42', 'outconv']:
        new_params[i] = weights[".".join(i_parts[:])]

model.load_state_dict(new_params)


def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = []
    b.append(model.encoder.e11)
    b.append(model.encoder.e12)
    b.append(model.encoder.e21)
    b.append(model.encoder.e22)
    b.append(model.encoder.e31)
    b.append(model.encoder.e32)
    b.append(model.encoder.e41)
    b.append(model.encoder.e42)
    b.append(model.encoder.e51)
    b.append(model.encoder.e52)
    b.append(model.encoder.upconv1)
    b.append(model.encoder.d11)
    b.append(model.encoder.d12)
    b.append(model.encoder.upconv2)
    b.append(model.encoder.d21)
    b.append(model.encoder.d22)


    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.linear_e.parameters())
    b.append(model.sa.parameters())
    b.append(model.gate.parameters())
    b.append(model.conv1.parameters())
    b.append(model.ConvGRU.parameters())
    b.append(model.bn1.parameters())
    b.append(model.conv_fusion.parameters())
    b.append(model.conv51.parameters())

    b.append(model.upconv3.parameters())
    b.append(model.d31.parameters())
    b.append(model.d32.parameters())
    b.append(model.upconv4.parameters())
    b.append(model.d41.parameters())
    b.append(model.d42.parameters())
    b.append(model.outconv.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i


optimizer = torch.optim.SGD([{'params': get_1x_lr_params(model), 'lr': 1 * 0.00025},  # Learn for specific layers, some layers don’t learn
                             {'params': get_10x_lr_params(model), 'lr': 10 * 0.00025}],
                            lr=0.00025, momentum=0.9, weight_decay=0.0005)
loss_fn = torch.nn.BCELoss()
logger = open('./logs/logs.txt', 'a')
resume = 22
if resume:
    state_model = torch.load(
        '/home/fumcomp/Desktop/poorsoltani/avs_like_lr_later_former/epoch_' + str(resume - 1) + '.pth')
    model.load_state_dict(state_model['model'])
    optimizer.load_state_dict(state_model['optimizer'])
EPOCHE = 30
loss_report = 1
for i in range(resume, EPOCHE):
    batch_losses = []
    step_100_loss = []
    cnt = 1
    # progress_bar=tqdm(enumerate(train_loader), total=(len(train_loader.batch_sampler)))
    for step, (img, label) in tqdm(enumerate(train_loader), total=(len(train_loader.batch_sampler))):
        node0 = Variable(img[0].requires_grad_()).cuda()
        node1 = Variable(img[1].requires_grad_()).cuda()
        node2 = Variable(img[2].requires_grad_()).cuda()

        label0 = Variable(label[0].requires_grad_()).cuda()
        label1 = Variable(label[1].requires_grad_()).cuda()
        label2 = Variable(label[2].requires_grad_()).cuda()

        pred0, pred1, pred2 = model(node0, node1, node2)
        loss0 = loss_fn(pred0, label0)
        loss1 = loss_fn(pred1, label1)
        loss2 = loss_fn(pred2, label2)
        loss = (loss0 + loss1 + loss2)

        optimizer.zero_grad()  # (reset gradients)
        adjust_learning_rate(optimizer, step + i * len(train_loader), max_iter=EPOCHE * len(train_loader))
        loss.backward()  # (compute gradients)
        optimizer.step()

        loss_value = loss.data.cpu().numpy()
        batch_losses.append(loss_value)
        step_100_loss.append(loss_value)
        if not (cnt % 100):
            logger.write('step= {}\t mean loss={}\n'.format(step, np.mean(batch_losses)))

        cnt += 1

    logger.write('###########################################################\n')
    logger.write('epoch= {}\t mean loss={}\n'.format(i, np.mean(batch_losses)))
    logger.write('###########################################################\n')
    logger.flush()
    torch.save({
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
    }, '/home/fumcomp/Desktop/poorsoltani/avs_like_lr_later_former/epoch_' + str(i) + '.pth')
    print('##########################################')
    print('################### epoch {} ############'.format(i))
    print('##########################################')
