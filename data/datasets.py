import os
import random

import imageio
import numpy as np
from sklearn.cluster import DBSCAN
from torch.utils.data import Dataset

import data.imutils as imutils


def img_loader(path):
    img = np.array(imageio.imread(path), np.float32)
    return img


class BiTemporalDataSet(Dataset):
    def __init__(self, dataset_path, data_list, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]

        self.img_fliplr = True
        self.img_flipud = True

    def __transforms(self, aug, pre_img, post_img, label):
        if aug:
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        post_img = imutils.normalize_img(post_img)

        pre_img = np.transpose(pre_img, (2, 0, 1))  # pytorch requires channel, head, weight
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path, self.type, 'T1', self.data_list[index])
        post_path = os.path.join(self.dataset_path, self.type, 'T2', self.data_list[index])
        label_path = os.path.join(self.dataset_path, self.type, 'GT', self.data_list[index])
        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        label = self.loader(label_path) / 255  # if the labels value is [0, 1], comment out this line

        if 'train' in self.data_pro_type:
            pre_img, post_img, label = self.__transforms(True, pre_img, post_img, label)
        else:
            pre_img, post_img, label = self.__transforms(False, pre_img, post_img, label)
            label = np.asarray(label)

        data_idx = self.data_list[index]
        return pre_img, post_img, label, data_idx

    def __len__(self):
        return len(self.data_list)


class I3PEDataSet(Dataset):
    def __init__(self, dataset_path, data_list, exchange_ratio, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.exchange_ratio = exchange_ratio
        self.scale_factor = [16, 32, 64, 128]
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]

        self.img_fliplr = True
        self.img_flipud = True
        np.random.seed()

    def __transforms(self, aug, pre_img, post_img, label):
        if aug:
            color_seed = random.random()
            if color_seed < 0.5 and color_seed >= 0.25:
                pre_img = imutils.randomColor(pre_img)
            elif color_seed < 0.75 and color_seed >= 0.5:
                post_img = imutils.randomColor(post_img)
            elif color_seed >= 0.75:
                pre_img = imutils.randomColor(pre_img)
                post_img = imutils.randomColor(post_img)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        post_img = imutils.normalize_img(post_img)

        pre_img = np.transpose(pre_img, (2, 0, 1))  # pytorch requires channel, head, weight
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def __getitem__(self, index):
        pre_img_name = self.data_list[index]
        pre_img_path = os.path.join(self.dataset_path, 'T1', pre_img_name)

        exchange_type_seed = random.random()
        patch_sz = random.choice(self.scale_factor)
        if exchange_type_seed < 0.5:  # Performing Intra-Image Patch Exchange Method
            label_path = os.path.join(self.dataset_path, 'clf_map', pre_img_name)
            pre_img = self.loader(pre_img_path)
            class_label = self.loader(label_path)

            post_img, label = self.intRA_image_patch_exchange(pre_img, class_label, patch_sz=patch_sz)

        else:  # Performing Inter-Image Patch Exchange Method
            another_img_name = random.choice(self.data_list)
            post_image_path = os.path.join(self.dataset_path, 'T1', another_img_name)
            pre_object_path = os.path.join(self.dataset_path, 'object', pre_img_name[0:-4] + '.npy')
            another_object_path = os.path.join(self.dataset_path, 'object', another_img_name[0:-4] + '.npy')

            pre_img = self.loader(pre_img_path)
            another_img = self.loader(post_image_path)

            pre_object = np.load(pre_object_path)
            another_object = np.load(another_object_path)

            post_img, label = self.intER_image_patch_exchange(pre_img, another_img, pre_object, another_object,
                                                              patch_sz=patch_sz)

        pre_img, post_img, label = self.__transforms(True, pre_img, post_img, label)
        data_idx = self.data_list[index]
        return pre_img, post_img, label, data_idx

    def intRA_image_patch_exchange(self, img, class_label, patch_sz):
        patch_num_in_raw = img.shape[0] // patch_sz
        patch_idx = np.arange(patch_num_in_raw ** 2)
        np.random.shuffle(patch_idx)

        exchange_patch_num = int((patch_num_in_raw ** 2) * self.exchange_ratio)

        exchange_img = img.copy()
        change_label = np.zeros(img.shape[0:2]).astype(np.uint8)

        for i in range(0, exchange_patch_num, 2):
            first_patch_idx = np.unravel_index(patch_idx[i], (patch_num_in_raw, patch_num_in_raw))
            second_patch_idx = np.unravel_index(patch_idx[i + 1], (patch_num_in_raw, patch_num_in_raw))

            first_patch = img[patch_sz * first_patch_idx[0]: patch_sz * (first_patch_idx[0] + 1),
                          patch_sz * first_patch_idx[1]: patch_sz * (first_patch_idx[1] + 1)]
            second_patch = img[patch_sz * second_patch_idx[0]: patch_sz * (second_patch_idx[0] + 1),
                           patch_sz * second_patch_idx[1]: patch_sz * (second_patch_idx[1] + 1)]

            temp = first_patch.copy()
            exchange_img[patch_sz * first_patch_idx[0]: patch_sz * (first_patch_idx[0] + 1),
            patch_sz * first_patch_idx[1]: patch_sz * (first_patch_idx[1] + 1)] = second_patch
            exchange_img[patch_sz * second_patch_idx[0]: patch_sz * (second_patch_idx[0] + 1),
            patch_sz * second_patch_idx[1]: patch_sz * (second_patch_idx[1] + 1)] = temp

            incons_label = \
                (class_label[patch_sz * first_patch_idx[0]: patch_sz * (first_patch_idx[0] + 1),
                 patch_sz * first_patch_idx[1]: patch_sz * (first_patch_idx[1] + 1)] !=
                 class_label[patch_sz * second_patch_idx[0]: patch_sz * (second_patch_idx[0] + 1),
                 patch_sz * second_patch_idx[1]: patch_sz * (second_patch_idx[1] + 1)]).astype(np.uint8)

            #
            change_label[patch_sz * first_patch_idx[0]: patch_sz * (first_patch_idx[0] + 1),
            patch_sz * first_patch_idx[1]: patch_sz * (first_patch_idx[1] + 1)] = incons_label

            change_label[patch_sz * second_patch_idx[0]: patch_sz * (second_patch_idx[0] + 1),
            patch_sz * second_patch_idx[1]: patch_sz * (second_patch_idx[1] + 1)] = incons_label

            # uncen_idx = (class_label[patch_sz * first_patch_idx[0]: patch_sz * (first_patch_idx[0] + 1),
            #              patch_sz * first_patch_idx[1]: patch_sz * (first_patch_idx[1] + 1)] == 0) | \
            #             (class_label[patch_sz * second_patch_idx[0]: patch_sz * (second_patch_idx[0] + 1),
            #              patch_sz * second_patch_idx[1]: patch_sz * (second_patch_idx[1] + 1)] == 0)
            # change_label[patch_sz * second_patch_idx[0]: patch_sz * (second_patch_idx[0] + 1),
            # patch_sz * second_patch_idx[1]: patch_sz * (second_patch_idx[1] + 1)][uncen_idx] = 255
            #
            # change_label[patch_sz * first_patch_idx[0]: patch_sz * (first_patch_idx[0] + 1),
            # patch_sz * first_patch_idx[1]: patch_sz * (first_patch_idx[1] + 1)][uncen_idx] = 255

        return exchange_img, change_label

    def intER_image_patch_exchange(self, img_1, img_2, object_1, object_2, patch_sz):
        concat_img = np.concatenate([img_1, img_2], axis=1)

        object_2 = np.max(object_1) + 1 + object_2
        concat_object = np.concatenate([object_1, object_2], axis=1)

        obj_num = np.max(concat_object)

        feat_vect = np.zeros((obj_num + 1, 6))
        for obj_idx in range(0, obj_num + 1):
            feat_vect[obj_idx - 1] = np.concatenate(
                [np.mean(concat_img[concat_object == obj_idx], axis=0),
                 np.std(concat_img[concat_object == obj_idx], axis=0)], axis=0)

        # You need to tune these two parameters carefully
        clustering = DBSCAN(eps=7.5, min_samples=10, n_jobs=1).fit(feat_vect)
        clustered_labels = clustering.labels_

        clustered_map = np.zeros(concat_img.shape[0:2]).astype(np.uint8)
        for obj_idx in range(0, obj_num + 1):
            clustered_map[concat_object == obj_idx] = clustered_labels[obj_idx]

        label_1 = clustered_map[:, 0:img_1.shape[1]]
        label_2 = clustered_map[:, img_1.shape[1]:]

        change_label = (label_1 != label_2).astype(np.uint8)
        # change_label[label_1 == -1] = 255
        # change_label[label_2 == -1] = 255

        patch_num_in_raw = img_1.shape[0] // patch_sz
        patch_idx = np.arange(patch_num_in_raw ** 2)
        np.random.shuffle(patch_idx)

        exchange_patch_num = int(self.exchange_ratio * (patch_num_in_raw ** 2))
        exchange_patch_idx = patch_idx[0:exchange_patch_num]
        exchange_img = img_1.copy()
        exchange_change_label = np.zeros(img_1.shape[0:2]).astype(np.uint8)

        for idx in range(0, exchange_patch_num):
            patch_idx = np.unravel_index(exchange_patch_idx[idx], (patch_num_in_raw, patch_num_in_raw))

            exchange_img[patch_sz * patch_idx[0]: patch_sz * (patch_idx[0] + 1),
            patch_sz * patch_idx[1]: patch_sz * (patch_idx[1] + 1)] = \
                img_2[patch_sz * patch_idx[0]: patch_sz * (patch_idx[0] + 1),
                patch_sz * patch_idx[1]: patch_sz * (patch_idx[1] + 1)]

            exchange_change_label[patch_sz * patch_idx[0]: patch_sz * (patch_idx[0] + 1),
            patch_sz * patch_idx[1]: patch_sz * (patch_idx[1] + 1)] = \
                change_label[patch_sz * patch_idx[0]: patch_sz * (patch_idx[0] + 1),
                patch_sz * patch_idx[1]: patch_sz * (patch_idx[1] + 1)]

        return exchange_img, exchange_change_label

    def __len__(self):
        return len(self.data_list)
