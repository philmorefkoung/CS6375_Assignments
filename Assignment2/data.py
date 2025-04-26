"""
CS 6375 Homework 2 Programming
Implement the __getitem__() function in this python script
"""
import torch
import torch.utils.data as data
import csv
import os, math
import sys
import time
import random
import numpy as np
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# The dataset class
class CrackerBox(data.Dataset):
    def __init__(self, image_set = 'train', data_path = 'data'):

        self.name = 'cracker_box_' + image_set
        self.image_set = image_set
        self.data_path = data_path
        self.classes = ('__background__', 'cracker_box')
        self.width = 640
        self.height = 480
        self.yolo_image_size = 448
        self.scale_width = self.yolo_image_size / self.width
        self.scale_height = self.yolo_image_size / self.height
        self.yolo_grid_num = 7
        self.yolo_grid_size = self.yolo_image_size / self.yolo_grid_num
        # split images into training set and validation set
        self.gt_files_train, self.gt_files_val = self.list_dataset()
        # the pixel mean for normalization
        self.pixel_mean = np.array([[[102.9801, 115.9465, 122.7717]]], dtype=np.float32)

        # training set
        if image_set == 'train':
            self.size = len(self.gt_files_train)
            self.gt_paths = self.gt_files_train
            print('%d images for training' % self.size)
        else:
            # validation set
            self.size = len(self.gt_files_val)
            self.gt_paths = self.gt_files_val
            print('%d images for validation' % self.size)


    # list the ground truth annotation files
    # use the first 100 images for training
    def list_dataset(self):
    
        filename = os.path.join(self.data_path, '*.txt')
        gt_files = sorted(glob.glob(filename))
        
        gt_files_train = gt_files[:100]
        gt_files_val = gt_files[100:]
        
        return gt_files_train, gt_files_val


    # TODO: implement this function
    def __getitem__(self, idx):
    
        # gt file
        filename_gt = self.gt_paths[idx]
        with open(filename_gt, 'r') as f:
            box = np.array([float(x) for x in f.readline().strip().split()])
        
        # load images
        image_filename = filename_gt.replace('-box.txt', '.jpg')
        image = cv2.imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # resize images to 448x448
        image = cv2.resize(image, (self.yolo_image_size, self.yolo_image_size))
        
        # normalize images: subtract pixel_mean, divide by 255
        image = image.astype(np.float32)
        image -= self.pixel_mean
        image /= 255.0
        
        # convert to PyTorch tensor and swap dimensions to (C, H, W)
        image_blob = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        # scale bounding box coordinates to 448x448
        box[0] *= self.scale_width  # x1
        box[1] *= self.scale_height  # y1
        box[2] *= self.scale_width  # x2
        box[3] *= self.scale_height  # y2
        
        # compute center, width, height
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        w = box[2] - box[0]
        h = box[3] - box[1]
        
        # determine which grid cell contains center
        grid_x = int(cx // self.yolo_grid_size)
        grid_y = int(cy // self.yolo_grid_size)
        
        # gt_box and gt_mask tensors
        gt_box_blob = torch.zeros(5, self.yolo_grid_num, self.yolo_grid_num)
        gt_mask_blob = torch.zeros(self.yolo_grid_num, self.yolo_grid_num)
        
        # process valid grid cell 7x7 grid
        if 0 <= grid_x < self.yolo_grid_num and 0 <= grid_y < self.yolo_grid_num:
            # norm cx, cy relative to grid cell
            cx_norm = (cx % self.yolo_grid_size) / self.yolo_grid_size
            cy_norm = (cy % self.yolo_grid_size) / self.yolo_grid_size
            w_norm = w / self.yolo_image_size
            h_norm = h / self.yolo_image_size
            confidence = 1.0
            
            # store in gt_box_blob
            gt_box_blob[:, grid_y, grid_x] = torch.tensor([cx_norm, cy_norm, w_norm, h_norm, confidence])
            
            # set gt_mask to 1 for this grid cell
            gt_mask_blob[grid_y, grid_x] = 1.0

        # this is the sample dictionary to be returned from this function
        sample = {'image': image_blob,
                  'gt_box': gt_box_blob,
                  'gt_mask': gt_mask_blob}

        return sample


    # len of the dataset
    def __len__(self):
        return self.size
        

# draw grid on images for visualization
def draw_grid(image, line_space=64):
    H, W = image.shape[:2]
    image[0:H:line_space] = [255, 255, 0]
    image[:, 0:W:line_space] = [255, 255, 0]


# the main function for testing
if __name__ == '__main__':
    dataset_train = CrackerBox('train')
    dataset_val = CrackerBox('val')
    
    # dataloader
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0)
    
    # visualize the training data
    for i, sample in enumerate(train_loader):
        
        image = sample['image'][0].numpy().transpose((1, 2, 0))
        gt_box = sample['gt_box'][0].numpy()
        gt_mask = sample['gt_mask'][0].numpy()

        y, x = np.where(gt_mask == 1)
        cx = gt_box[0, y, x] * dataset_train.yolo_grid_size + x * dataset_train.yolo_grid_size
        cy = gt_box[1, y, x] * dataset_train.yolo_grid_size + y * dataset_train.yolo_grid_size
        w = gt_box[2, y, x] * dataset_train.yolo_image_size
        h = gt_box[3, y, x] * dataset_train.yolo_image_size

        x1 = cx - w * 0.5
        x2 = cx + w * 0.5
        y1 = cy - h * 0.5
        y2 = cy + h * 0.5

        print(image.shape, gt_box.shape)
        
        # visualization
        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 1)
        im = image * 255.0 + dataset_train.pixel_mean
        im = im.astype(np.uint8)
        plt.imshow(im[:, :, (2, 1, 0)])
        plt.title('input image (448x448)', fontsize = 16)

        ax = fig.add_subplot(1, 3, 2)
        draw_grid(im)
        plt.imshow(im[:, :, (2, 1, 0)])
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='g', facecolor="none")
        ax.add_patch(rect)
        plt.plot(cx, cy, 'ro', markersize=12)
        plt.title('Ground truth bounding box in YOLO format', fontsize=16)
        
        ax = fig.add_subplot(1, 3, 3)
        plt.imshow(gt_mask)
        plt.title('Ground truth mask in YOLO format (7x7)', fontsize=16)
        plt.show()
