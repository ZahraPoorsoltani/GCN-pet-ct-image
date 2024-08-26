from pathlib import Path
import cv2
import torch
import numpy as np
import imgaug
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import random


class LungDataset(torch.utils.data.Dataset):
    def __init__(self, root, augment_params, num_nodes=5):
        self.all_files = self.extract_files(root)
        self.augment_params = augment_params
        self.num_nodes=num_nodes

    def extract_files(self,root):
        """
        Extract the paths to all slices given the root path (ends with train or val)
        """
        files = []
        Index = {}
        for subject in root.glob("*"):  # Iterate over the subjects
            slice_path = subject / "ct"  # Get the slices for current subject
            patient=subject.parts[-1]
            start_num = len(files)

            for slice in slice_path.glob("*"):
                files.append(slice)

            end_num = len(files)
            Index[patient.strip('\n')] = np.array([start_num, end_num])
        self.Index=Index
        return files

    @staticmethod
    def change_img_to_label_path(path):
        """
        Replace data with mask to get the masks
        """
        parts = list(path.parts)
        parts[parts.index("ct")] = "mask"
        return Path(*parts)
    @staticmethod
    def change_img_to_pet_path(path):
        """
        Replace data with mask to get the masks
        """
        parts = list(path.parts)
        parts[parts.index("ct")] = "pet"
        return Path(*parts)

    @staticmethod
    def change_img_to_suv_path(path):
        """
        Replace data with mask to get the masks
        """
        parts = list(path.parts)
        parts[parts.index("ct")] = "suv"
        return Path(*parts)



    def augment(self, cts,pets, masks):
        """
        Augments slice and segmentation mask in the exact same way
        Note the manual seed initialization
        """
        ###################IMPORTANT###################
        # Fix for https://discuss.pytorch.org/t/dataloader-workers-generate-the-same-random-augmentations/28830/2
        random_seed = torch.randint(0, 1000000, (1,))[0].item()
        imgaug.seed(random_seed)
        #####################################################
        self.augment_params=self.augment_params.to_deterministic()
        mask_augs=[]
        ct_augs=[]
        pet_augs=[]
        for i in range(self.num_nodes):
            masks[i] = SegmentationMapsOnImage(masks[i], masks[i].shape)
            ct_aug, mask_aug = self.augment_params(image=cts[i], segmentation_maps=masks[i])
            ct_augs.append(ct_aug)
            mask_augs.append(mask_aug)

            pet_aug, mask_aug= self.augment_params(image=pets[i], segmentation_maps=masks[i])
            pet_augs.append(pet_aug)

            mask_augs[i] = mask_augs[i].get_arr()


        return ct_augs,pet_augs, mask_augs

    @staticmethod
    def normalize(img):
        min = np.min(img)
        max = np.max(img)
        img = img - (min)
        img = img / (max - min)
        return img

    def __len__(self):
        """
        Return the length of the dataset (length of all files)
        """
        return len(self.all_files)


    def __getitem__(self, idx):

        patient = self.all_files[idx].parts[-3]
        search_idx = self.Index[patient]
        cts = []
        suvs = []
        masks = []
        range_search = 25
        num_node = self.num_nodes
        if idx + range_search < search_idx[1] and idx - range_search >= search_idx[0]:
            search_idices = random.sample([*np.arange(idx - range_search, idx + range_search)], num_node)

        elif idx + range_search >= search_idx[1] and idx - range_search >= search_idx[0]:
            loc = search_idx[1] - idx
            search_idices = random.sample([*np.arange(idx - (range_search + (range_search - loc)), idx + loc)], num_node)
        else:
            # (idx + 10 < search_idx[1] and idx - 10 < search_idx[0]) in else situation
            loc = idx - search_idx[0]
            search_idices = random.sample([*np.arange(idx - loc, idx + (range_search + (range_search - loc)))], num_node)

        for i in search_idices:
            file_path = self.all_files[i]
            mask_path = self.change_img_to_label_path(file_path)
            suv_path = self.change_img_to_suv_path(file_path)
            ct = (np.load(file_path)) / 3071
            cts.append(ct)
            masks.append(np.load(mask_path))
            suvs.append(np.load(suv_path))

        if self.augment_params:
            cts, suvs, masks = self.augment(cts, suvs, masks)

        nodes = []
        for i in range(num_node):
            concat_img = np.zeros([2, 256, 256])
            concat_img[0, :, :] = suvs[i]
            concat_img[1, :, :] = cts[i]
            concat_img = concat_img.astype('float32')
            nodes.append(concat_img)
            masks[i] = np.expand_dims(masks[i], 0).astype('float32')

        return nodes, masks
