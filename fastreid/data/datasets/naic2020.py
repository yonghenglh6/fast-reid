# encoding: utf-8
"""
@author:  liuhao
@contact: liuhao@megvii.com
"""

import glob
import os.path as osp
import re
import warnings
from collections import defaultdict

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Naic2020(ImageDataset):
    """Naic2020.
    """
    _junk_pids = []
    dataset_dir = ''
    dataset_url = ''
    dataset_name = "Naic2020"

    def __init__(self, root='datasets', **kwargs):
        root='/data/datasets'
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        self.data_dir = osp.join(self.data_dir, 'naic')
        

        self.train_dir = osp.join(self.data_dir, 'train')
        self.test_dir = osp.join(self.data_dir, 'test')

        self.query_dir = osp.join(self.test_dir, 'query_a')
        self.gallery_dir = osp.join(self.test_dir, 'gallery_a')

        required_files = [
            self.data_dir,
            self.train_dir,
            self.test_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir_train(self.train_dir)
        query = self.process_dir_test(self.query_dir)
        gallery = self.process_dir_test(self.gallery_dir)

        super(Naic2020, self).__init__(train[:200], query[:200], gallery[:200], **kwargs)

    def process_dir_train(self, dir_path,):
        filename = osp.join(dir_path, 'label.txt')
        dataset = []
        count_image = defaultdict(list)
        with open(filename, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                img_name, img_label = [i for i in lines.split(':')]
                count_image[img_label].append(img_name)
        
        # 这儿重新对pid进行一个编号，防止有空的
        val_imgs = {}
        pid_container = set()
        for pid, img_name in count_image.items():
            val_imgs[pid] = count_image[pid]
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        camid = 1

        for pid, img_name in val_imgs.items():
            pid = pid2label[pid]
            for img in img_name:
                dataset.append(
                    (osp.join(dir_path, 'images/', img), pid, camid))
                camid+=1

        return dataset
    def process_dir_test(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        dataset = []
        pid,camid = 0,0
        for img_path in img_paths:
            dataset.append((osp.join(dir_path, img_path), pid, camid))
            pid,camid = pid+1,camid+1
        return dataset



@DATASET_REGISTRY.register()
class Naic2020Train(ImageDataset):
    """Naic2020. 把一部分训练集当做测试集给出去
    """
    _junk_pids = []
    dataset_dir = ''
    dataset_url = ''
    dataset_name = "Naic2020Train"
    def __init__(self, root='datasets', **kwargs):
        root='/data/datasets'
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        self.data_dir = osp.join(self.data_dir, 'naic')
        
        self.train_dir = osp.join(self.data_dir, 'train')

        required_files = [
            self.data_dir,
            self.train_dir,
        ]
        self.check_before_run(required_files)

        self.pid2label = {}
        train = self.process_dir_train(self.data_dir,'train/label.txt','train/images',relabel=True)
        query = self.process_dir_train(self.data_dir, 'mini_val_query.txt','train/images',relabel=False)
        gallery = self.process_dir_train(self.data_dir, 'mini_val_gallery.txt','train/images',relabel=False)


        super(Naic2020Train, self).__init__(train, query, gallery, **kwargs)

    def process_dir_train(self, dir_path, txt_file, image_relative='images',relabel=False):
        filename = osp.join(dir_path, txt_file)
        dataset = []
        count_image = defaultdict(list)
        with open(filename, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                img_name, img_label = [i for i in lines.split(':')]
                count_image[img_label].append(img_name)
        if relabel:
            assert len(self.pid2label)==0
            # 这儿重新对pid进行一个编号，防止有空的process_dir_train
            val_imgs = {}
            pid_container = set()
            for pid, img_name in count_image.items():
                val_imgs[pid] = count_image[pid]
                pid_container.add(pid)
            
            self.pid2label = {pid: label for label, pid in enumerate(pid_container)}
            camid = 1

            for pid, img_name in val_imgs.items():
                pid = self.pid2label[pid]
                for img in img_name:
                    dataset.append(
                        (osp.join(dir_path, image_relative, img), pid, camid))
                    camid+=1
        else:
            assert len(self.pid2label)>0
            camid = 1
            for img_label in count_image:
                for img in count_image[img_label]:
                    dataset.append(
                        (osp.join(dir_path, image_relative, img), self.pid2label[img_label], camid))
                    camid+=1

        return dataset
