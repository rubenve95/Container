import numpy as np
import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from project_code.dataloader import custom_transforms as tr
from project_code.dataloader.utils import decode_segmap,encode_segmap,sort_video_files

class Loader(Dataset):
    def __init__(self, options):
        self.datafolder = options['data']['root']
        self.img_size = options['data']['img_size']
        self.use_temporal = options['data']['use_temporal']
        self.num_classes = options['data']['num_classes']
        self.split = 'train'
        images_path = os.path.join(self.datafolder, 'images')
        gt_path = os.path.join(self.datafolder, 'ground_truth')
        folders = os.listdir(images_path)
        #subfolders_gt = os.listdir(gt_path)
        self.data = []

        self.mean = (0.485, 0.456, 0.406)
        self.std =(0.229, 0.224, 0.225)
                # self.aug = Compose(
                #     [RandomRotate(aug_params['RandomRotate']),
                #     AdjustSaturation(aug_params['AdjustSaturation']), 
                #     AdjustGamma(aug_params['AdjustGamma']),
                #     AdjustBrightness(aug_params['AdjustBrightness'])],
                # )  
        for folder in folders:
            files = os.listdir(os.path.join(images_path,folder))
            is_video = (folder.split('.')[-1] == 'MP4')
            if is_video:
                files = sort_video_files(files)
            for i,f in enumerate(files):
                img = os.path.join(images_path,folder,f)
                gt = f.split('.')[0] + '.png'
                gt = os.path.join(gt_path,folder,gt)
                new_entry = {'image': img, 'ground_truth': gt}

                if self.use_temporal and is_video and i != 0:
                    prev_gt = files[i-1].split('.')[0] + '.png'
                    prev_gt = os.path.join(gt_path,folder,prev_gt)
                    new_entry['previous'] = prev_gt

                self.data.append(new_entry)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.get_sample(index)

        if self.split == "train":
            sample = self.transform_tr(sample)
        elif self.split in ['val','test']:
            sample = self.transform_val(sample)

        sample['ground_truth'] = self.make_gt(sample['ground_truth'], oneHot=False)
        if self.use_temporal:
            if 'previous' in sample.keys():
                previous = self.make_gt(sample['previous'], oneHot=True).float()
            else:
                previous = torch.zeros(self.num_classes, self.img_size[1], self.img_size[0])
            image = torch.cat((sample['image'], previous),0)
            sample = {'image': image, 'ground_truth': sample['ground_truth']}

        sample['name'] = self.data[index]['image'].split('/')[-1]

        return sample

    def make_gt(self, gt, oneHot=False):
        gt = gt.numpy()
        gt = np.array(gt).astype(np.uint8)
        segmap = encode_segmap(gt,oneHot=oneHot)
        return torch.from_numpy(segmap)

    def get_sample(self, index):
        dict = self.data[index]
        _img = Image.open(dict['image']).convert('RGB')
        _target = Image.open(dict['ground_truth'])
        sample = {'image': _img, 'ground_truth': _target}
        if self.use_temporal and 'previous' in dict.keys():
            previous_gt = Image.open(dict['previous'])
            sample['previous'] = previous_gt
        return sample

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            #tr.RandomHorizontalFlip(),
            #tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.FixedResize(size = self.img_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=self.mean, std=self.std),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            #tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.FixedResize(size = self.img_size),
            tr.Normalize(mean=self.mean, std=self.std),
            tr.ToTensor()])

        return composed_transforms(sample)

    def get_folder_indices(self, foldername):
        indices = []
        for i,datapoint in enumerate(self.data):
            path = datapoint['image']
            path = path.split('/')
            if foldername in path:
                indices.append(i)
        return indices