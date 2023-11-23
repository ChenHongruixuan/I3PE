import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import BiTemporalDataSet, I3PEDataSet
from model.ResNetFPN import ResNetFPN
from utils.metrics import Evaluator


class Trainer(object):
    def __init__(self, args):
        self.args = args

        train_dataset = I3PEDataSet(dataset_path=args.train_dataset_path,
                                    data_list=args.train_data_name_list,
                                    exchange_ratio=args.exchange_ratio,
                                    max_iters=args.max_iters,
                                    type='train')

        self.train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle,
                                            num_workers=8,
                                            drop_last=False)
        self.evaluator = Evaluator(num_class=2)

        self.deep_model = ResNetFPN(pretrained=True)  # ResNetFPN(pretrained=False)
        self.deep_model = self.deep_model.cuda()
        self.model_save_path = os.path.join(args.model_param_path, args.dataset,
                                            args.model_type + '_' + str(time.time()))
        self.lr = args.learning_rate
        self.epoch = args.max_iters // args.batch_size

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        # self.optim = optim.SGD(self.deep_model.parameters(), lr=args.learning_rate, momentum=args.momentum,
        # weight_decay=args.weight_decay)
        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    def training(self):
        best_f1 = 0.0
        torch.cuda.empty_cache()
        weight = torch.FloatTensor([1, 1]).cuda()  # Tune this according to your own situation
        tbar = tqdm(self.train_data_loader)

        for itera, data in enumerate(tbar):
            pre_change_imgs, post_change_imgs, labels, _ = data

            pre_change_imgs = pre_change_imgs.cuda()
            post_change_imgs = post_change_imgs.cuda()
            labels = labels.cuda().long()

            output_1 = self.deep_model(pre_change_imgs, post_change_imgs)
            output_2 = self.deep_model(post_change_imgs, pre_change_imgs)
            self.optim.zero_grad()

            ce_loss_1 = F.cross_entropy(output_1, labels, weight=weight, ignore_index=255)
            ce_loss_2 = F.cross_entropy(output_2, labels, weight=weight, ignore_index=255)

            main_loss = (ce_loss_1 + ce_loss_2)

            main_loss.backward()
            self.optim.step()
            if (itera + 1) % 10 == 0:
                print(f'iter is {itera + 1}, overall loss is {main_loss}')
                if (itera + 1) % 500 == 0:
                    self.deep_model.eval()
                    f1 = self.validation()
                    if f1 > best_f1:
                        torch.save(self.deep_model.state_dict(),
                                   os.path.join(self.model_save_path, f'{itera + 1}_model.pth'))
                        best_f1 = f1
                    self.deep_model.train()

    def validation(self):
        print('---------starting evaluation-----------')
        self.evaluator.reset()

        dataset = BiTemporalDataSet(self.args.test_dataset_path, self.args.test_data_name_list, None, 'test')
        val_data_loader = DataLoader(dataset, batch_size=8, num_workers=8, drop_last=False)
        torch.cuda.empty_cache()
        for itera, data in enumerate(val_data_loader):
            pre_change_imgs, post_change_imgs, labels, _ = data
            pre_change_imgs = pre_change_imgs.cuda()
            post_change_imgs = post_change_imgs.cuda()
            labels = labels.cuda().long()

            output_1 = self.deep_model(pre_change_imgs, post_change_imgs)

            output_1 = output_1.data.cpu().numpy()
            output_1 = np.argmax(output_1, axis=1)
            labels = labels.cpu().numpy()

            self.evaluator.add_batch(labels, output_1)
        f1_score = self.evaluator.Pixel_F1_score()
        oa = self.evaluator.Pixel_Accuracy()
        rec = self.evaluator.Pixel_Recall_Rate()
        pre = self.evaluator.Pixel_Precision_Rate()
        print('Racall rate, Precision rate, OA is, F1 score is ', rec, pre, oa, f1_score)
        return f1_score


def main():
    parser = argparse.ArgumentParser(description="Training on SECOND dataset")
    parser.add_argument('--dataset', type=str, default='SYSU_I3PE')
    parser.add_argument('--train_dataset_path', type=str, default='../data/SYSU/st_train')
    parser.add_argument('--train_data_list_path', type=str, default='../data/SYSU/st_train_list.txt')
    parser.add_argument('--test_dataset_path', type=str, default='../data/SYSU')
    parser.add_argument('--test_data_list_path', type=str, default='../data/SYSU/test_list.txt')

    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--exchange_ratio', type=float, default=0.75)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str, default='ResNetFPN_SYSU_I3PE')
    parser.add_argument('--model_param_path', type=str, default='../saved_models')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    args = parser.parse_args()
    with open(args.train_data_list_path, "r") as f:
        data_name_list = [data_name.strip() for data_name in f]
    args.train_data_name_list = data_name_list
    with open(args.test_data_list_path, "r") as f:
        data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = data_name_list

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()
