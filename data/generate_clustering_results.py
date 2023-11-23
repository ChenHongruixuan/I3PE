import argparse
import os

import imageio
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm

label_dict = {
    -1: [0, 0, 0],
    0: [255, 0, 0],
    1: [0, 255, 0],
    2: [0, 0, 255],
    3: [255, 255, 0],
    4: [255, 0, 255],
    5: [0, 255, 255],
    6: [128, 255, 128],
    7: [255, 128, 128],
    8: [128, 128, 255],
    9: [255, 128, 255],
    10: [50, 50, 255],
    12: [50, 255, 50],
    13: [128, 50, 50],
    14: [128, 50, 255],
    15: [50, 255, 255],
    16: [128, 255, 50],
    17: [128, 0, 50],
    18: [0, 255, 50],
    19: [0, 128, 50],
    20: [0, 50, 50],
}


def make_rgb(a, rgb_codes: dict = label_dict):
    """_summary_
    Args:
        a (numpy arrat): semantic label (H x W)
        rgd_codes (dict): dict of class-rgd code
        grey_codes (dict): dict of label code
    Returns:
        array: semantic label map rgb-color coded
    """
    out = np.zeros(shape=a.shape + (3,), dtype="uint8")
    for k, v in rgb_codes.items():
        out[a == k, 0] = rgb_codes[k][0]
        out[a == k, 1] = rgb_codes[k][1]
        out[a == k, 2] = rgb_codes[k][2]
    return out


def generate_clustering_results(args):
    saved_path = os.path.join(args.dataset_path, 'st_train', 'clf_map')
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
        os.makedirs(os.path.join(saved_path, 'colored'))

    with open(os.path.join(args.dataset_path, 'st_train_list.txt')) as f:
        data_name_list = [data_name.strip() for data_name in f]
    for img_name in tqdm(data_name_list):
        image = imageio.imread(os.path.join(args.dataset_path, 'st_train', 'T1', img_name))
        objects = np.load(os.path.join(args.dataset_path, 'st_train', 'object', img_name[0:-4] + '.npy'))

        obj_num = np.max(objects)
        feat_vect = np.zeros((obj_num + 1, 6))

        for obj_idx in range(0, obj_num + 1):
            feat_vect[obj_idx] = np.concatenate(
                [np.mean(image[objects == obj_idx], axis=0),

                 np.std(image[objects == obj_idx], axis=0)], axis=0)

        clustering = DBSCAN(eps=args.eps, min_samples=args.min_samples, leaf_size=50, n_jobs=12,
                            metric='euclidean').fit(feat_vect)
        clustered_labels = clustering.labels_

        clustered_map = np.zeros(image.shape[0:2])
        for obj_idx in range(obj_num + 1):
            clustered_map[objects == obj_idx] = clustered_labels[obj_idx]

        colored_clustered_map = make_rgb(clustered_map)

        clustered_map = clustered_map + 1
        imageio.imwrite(os.path.join(saved_path, img_name), clustered_map.astype(np.uint8))
        imageio.imwrite(os.path.join(saved_path, 'colored', img_name), colored_clustered_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Clustering Maps Using DBSCAN")
    parser.add_argument('--dataset_path', type=str, default='D:/Workplace/SELF_SUPER_CD/data/SYSU')
    # You need to tune these two parameters carefully to obtain decent clustering results on your own datasets
    parser.add_argument('--eps', type=int, default=7)
    parser.add_argument('--min_samples', type=int, default=10)

    args = parser.parse_args()
    generate_clustering_results(args)
