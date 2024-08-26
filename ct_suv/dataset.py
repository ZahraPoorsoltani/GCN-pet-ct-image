import os.path
from pathlib import Path
import cv2
import torch
import numpy as np
import imgaug
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt


from imgaug.augmentables.segmaps import SegmentationMapsOnImage
def normalize(img):
    min=np.min(img)
    max=np.max(img)
    img=img-(min)
    img=img/(max-min)
    return img

class LungDataset(torch.utils.data.Dataset):
    def __init__(self, root, augment_params,one_patient=None):
        self.one_patient=one_patient
        self.all_files = self.extract_files(root)
        self.augment_params = augment_params


    def extract_files(self,root):
        """
        Extract the paths to all slices given the root path (ends with train or val)
        """
        files = []
        for subject in root.glob("*"):  # Iterate over the subjects
            slice_path = subject / "ct"  # Get the slices for current subject
            # slice_path = subject  # Get the slices for current subject
            for slice in slice_path.glob("*"):
                files.append(slice)
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
        parts[parts.index("ct")] = "suv"
        return Path(*parts)

    def augment(self, ct,pet, mask):
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
        mask = SegmentationMapsOnImage(mask, mask.shape)
        ct_aug, mask_aug = self.augment_params(image=ct, segmentation_maps=mask)
        pet_aug, mask_aug = self.augment_params(image=pet, segmentation_maps=mask)
        mask_aug = mask_aug.get_arr()
        return pet_aug,ct_aug, mask_aug



    def __len__(self):
        """
        Return the length of the dataset (length of all files)
        """
        return len(self.all_files)

    def __getitem__(self, idx):
        """
        Given an index return the (augmented) slice and corresponding mask
        Add another dimension for pytorch
        """
        file_path = self.all_files[idx]
        mask_path = self.change_img_to_label_path(file_path)
        suv_path=self.change_img_to_pet_path(file_path)
        ct = np.load(file_path)
        ct=ct/3071

        # meta_add=os.path.join('data/dcm_info/',file_path.parts[2]+'.IMA')
        # slice= self.transform_to_hu(meta_add,slice)

        mask = np.load(mask_path)

        suv=np.load(suv_path)

        if self.augment_params:
            suv,ct,mask=self.augment(ct,suv,mask)

        mask = mask.astype('float32')
        concat_img = np.zeros([2, 256, 256])

        concat_img[0, :, :] = suv
        concat_img[1, :, :] = ct
        concat_img=concat_img.astype('float32')
        return concat_img, np.expand_dims(mask, 0)
