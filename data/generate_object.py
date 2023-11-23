import argparse
import os

import imageio
import numpy as np
from skimage.segmentation import slic
from tqdm import tqdm


def generate_object(args):
    dataset_path = args.dataset_path
    object_path = os.path.join(dataset_path, 'st_train', 'object')
    if not os.path.exists(object_path):
        os.makedirs(object_path)
    with open(os.path.join(dataset_path, 'st_train_list.txt')) as f:
        data_name_list = [data_name.strip() for data_name in f]
    for data_name in tqdm(data_name_list):
        img = imageio.imread(os.path.join(dataset_path, 'st_train', 'T1', data_name))
        labels = slic(img, n_segments=args.n_segments, start_label=0)
        np.save(os.path.join(object_path, data_name[0:-4] + '.npy'), labels)
        # fig = plt.figure("Superpixels -- %d segments" % (np.max(labels)))
        # ax = fig.add_subplot(1, 1, 1)
        # ax.imshow(mark_boundaries(img_1, labels))
        # plt.axis("off")
        # # show the plots
        # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Objects")
    parser.add_argument('--dataset_path', type=str, default='D:/Workplace/SELF_SUPER_CD/data/SYSU')
    parser.add_argument('--n_segments', type=int, default=1000)

    args = parser.parse_args()
    generate_object(args)
