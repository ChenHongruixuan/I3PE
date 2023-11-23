import argparse
import os

import imageio
from tqdm import tqdm


def construct_st_set(args):
    # write the image file names into the list first
    with open(os.path.join(args.dataset_path, 'train_list.txt')) as f:
        data_name_list = [data_name.strip() for data_name in f]
    st_train_data_path = os.path.join(args.dataset_path, 'st_train', 'T1')
    if not os.path.exists(st_train_data_path):
        os.makedirs(st_train_data_path)
    with open(os.path.join(args.dataset_path, 'st_train_list.txt'), 'w+') as f:
        for data_name in tqdm(data_name_list):
            img_1 = imageio.imread(os.path.join(args.dataset_path, 'train', 'T1', data_name))
            img_2 = imageio.imread(os.path.join(args.dataset_path, 'train', 'T2', data_name))

            new_data_name_1 = data_name[0:5] + '_1.png'
            new_data_name_2 = data_name[0:5] + '_2.png'
            imageio.imwrite(os.path.join(st_train_data_path, new_data_name_1), img_1)
            imageio.imwrite(os.path.join(st_train_data_path, new_data_name_2), img_2)

            f.write(new_data_name_1 + '\n')
            f.write(new_data_name_2 + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct Single-Temporal Training Sets")
    # Utilize your own path here
    parser.add_argument('--dataset_path', type=str, default='D:/Workplace/SELF_SUPER_CD/data/SYSU')

    args = parser.parse_args()
    construct_st_set(args)
