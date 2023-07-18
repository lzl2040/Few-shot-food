import os
import torch
from torch.utils.data import Dataset
import PIL.Image as Image
import numpy as np
import torchvision.transforms as transforms

class Food101EpisodeDataset(Dataset):
    def __init__(self,img_root,meta_root,n_cls, n_support, n_query, input_w, input_h, n_episode=2000):
        # 根目录
        self.img_root = img_root
        self.meta_root = meta_root
        # 小样本的有关设置
        self.n_cls = n_cls
        self.n_support = n_support
        self.n_query = n_query
        self.n_episode = n_episode
        self.cls_list = np.arange(101)
        # 给标签划分界限
        intType = torch.LongTensor
        floatType = torch.FloatTensor
        self.label_support = intType(n_cls * n_support)
        self.label_query = intType(n_cls * n_query)
        self.support_images = floatType(n_cls * n_support, 3, input_w, input_h)
        self.query_images = floatType(n_cls * n_query, 3, input_w, input_h)
        # 获得类别对应的数字
        self.label2num = {}
        self.num2label = {}
        class_nums = 101
        count = 0
        with open(os.path.join(meta_root, 'labels.txt'), 'r') as f:
            for line in f:
                line = line.strip().replace(" ", "_").lower()
                # print(line)
                self.label2num[line] = count
                self.num2label[count] = line
                count += 1
        # 划分界限
        # labels {0, ..., nCls-1}
        for i in range(self.n_cls):
            self.label_support[i * self.n_support: (i + 1) * self.n_support] = i
            self.label_query[i * self.n_query: (i + 1) * self.n_query] = i
        # 转换函数
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    def __len__(self):
        return self.n_episode

    def __getitem__(self, idx):
        # 选取n_cls类训练
        cls_episode = np.random.choice(self.cls_list,self.n_cls,replace=False)
        for i,cls in enumerate(cls_episode):
            # 食物名字
            food_name = self.num2label[cls]
            # 获得图片文件夹地址
            cls_path = os.path.join(self.img_root,food_name)
            # 获得该文件夹下所有的图片地址
            img_list = os.listdir(cls_path)
            img_group = np.random.choice(img_list,self.n_support + self.n_query,replace=False)
            for j in range(self.n_support):
                img_name = img_group[j]
                path = os.path.join(cls_path,img_name)
                image = Image.open(path).convert('RGB')
                self.support_images[i * self.n_support + j] = self.transform(image)
            for j in range(self.n_query):
                img_name = img_group[j + self.n_support]
                path = os.path.join(cls_path, img_name)
                image = Image.open(path).convert('RGB')
                self.query_images[i * self.n_query + j] = self.transform(image)
        # 将顺序打乱
        perm_support = torch.randperm(self.n_cls * self.n_support)
        perm_query = torch.randperm(self.n_cls * self.n_query)
        return (self.support_images[perm_support],
                self.label_support[perm_support],
                self.query_images[perm_query],
                self.label_query[perm_query])


if __name__ == '__main__':
    print("test")



