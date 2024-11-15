import argparse
import os
from tqdm import tqdm

import imageio
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, ConcatDataset

import MambaCD.changedetection.datasets.imutils as imutils

def img_loader(path):
    img = np.array(imageio.imread(path), np.float32)
    return img


def one_hot_encoding(image, num_classes=5):
    # Create a one hot encoded tensor
    one_hot = np.eye(num_classes)[image.astype(np.uint8)]

    # Move the channel axis to the front
    # one_hot = np.moveaxis(one_hot, -1, 0)

    return one_hot



class ChangeDetectionDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, label):
        if aug:
            pre_img, post_img, label = imutils.random_crop_new(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path, 'T1', self.data_list[index])
        post_path = os.path.join(self.dataset_path, 'T2', self.data_list[index])
        label_path = os.path.join(self.dataset_path, 'GT', self.data_list[index])
        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        label = self.loader(label_path)
        label = label / 255

        if 'train' in self.data_pro_type:
            pre_img, post_img, label = self.__transforms(True, pre_img, post_img, label)
        else:
            pre_img, post_img, label = self.__transforms(False, pre_img, post_img, label)
            label = np.asarray(label)

        data_idx = self.data_list[index]
        return pre_img, post_img, label, data_idx

    def __len__(self):
        return len(self.data_list)


class SemanticChangeDetectionDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, cd_label, t1_label, t2_label):
        if aug:
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_crop_mcd(pre_img, post_img, cd_label, t1_label, t2_label, self.crop_size)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_fliplr_mcd(pre_img, post_img, cd_label, t1_label, t2_label)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_flipud_mcd(pre_img, post_img, cd_label, t1_label, t2_label)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_rot_mcd(pre_img, post_img, cd_label, t1_label, t2_label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, cd_label, t1_label, t2_label

    def __getitem__(self, index):
        if 'train' in self.data_pro_type:
            pre_path = os.path.join(self.dataset_path, 'T1', self.data_list[index] + '.png')
            post_path = os.path.join(self.dataset_path, 'T2', self.data_list[index] + '.png')
            T1_label_path = os.path.join(self.dataset_path, 'GT_T1', self.data_list[index] + '.png')
            T2_label_path = os.path.join(self.dataset_path, 'GT_T2', self.data_list[index] + '.png')
            cd_label_path = os.path.join(self.dataset_path, 'GT_CD', self.data_list[index] + '.png')
        else:
            pre_path = os.path.join(self.dataset_path, 'T1', self.data_list[index])
            post_path = os.path.join(self.dataset_path, 'T2', self.data_list[index])
            T1_label_path = os.path.join(self.dataset_path, 'GT_T1', self.data_list[index])
            T2_label_path = os.path.join(self.dataset_path, 'GT_T2', self.data_list[index])
            cd_label_path = os.path.join(self.dataset_path, 'GT_CD', self.data_list[index])

        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        t1_label = self.loader(T1_label_path)
        t2_label = self.loader(T2_label_path)
        cd_label = self.loader(cd_label_path)
        cd_label = cd_label / 255

        if 'train' in self.data_pro_type:
            pre_img, post_img, cd_label, t1_label, t2_label = self.__transforms(True, pre_img, post_img, cd_label, t1_label, t2_label)
        else:
            pre_img, post_img, cd_label, t1_label, t2_label = self.__transforms(False, pre_img, post_img, cd_label, t1_label, t2_label)
            cd_label = np.asarray(cd_label)
            t1_label = np.asarray(t1_label)
            t2_label = np.asarray(t2_label)

        data_idx = self.data_list[index]
        return pre_img, post_img, cd_label, t1_label, t2_label, data_idx

    def __len__(self):
        return len(self.data_list)


class DamageAssessmentDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, loc_label, clf_label):
        if aug:
            pre_img, post_img, loc_label, clf_label = imutils.random_crop_bda(pre_img, post_img, loc_label, clf_label, self.crop_size)
            pre_img, post_img, loc_label, clf_label = imutils.random_fliplr_bda(pre_img, post_img, loc_label, clf_label)
            pre_img, post_img, loc_label, clf_label = imutils.random_flipud_bda(pre_img, post_img, loc_label, clf_label)
            pre_img, post_img, loc_label, clf_label = imutils.random_rot_bda(pre_img, post_img, loc_label, clf_label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, loc_label, clf_label

    def __getitem__(self, index):
        if 'train' in self.data_pro_type: 
            parts = self.data_list[index].rsplit('_', 2)

            pre_img_name = f"{parts[0]}_pre_disaster_{parts[1]}_{parts[2]}.png"
            post_img_name = f"{parts[0]}_post_disaster_{parts[1]}_{parts[2]}.png"

            pre_path = os.path.join(self.dataset_path, 'images', pre_img_name)
            post_path = os.path.join(self.dataset_path, 'images', post_img_name)
            
            loc_label_path = os.path.join(self.dataset_path, 'masks', pre_img_name)
            clf_label_path = os.path.join(self.dataset_path, 'masks', post_img_name)
        else:
            pre_path = os.path.join(self.dataset_path, 'images', self.data_list[index] + '_pre_disaster.png')
            post_path = os.path.join(self.dataset_path, 'images', self.data_list[index] + '_post_disaster.png')
            loc_label_path = os.path.join(self.dataset_path, 'masks', self.data_list[index]+ '_pre_disaster.png')
            clf_label_path = os.path.join(self.dataset_path, 'masks', self.data_list[index]+ '_post_disaster.png')

        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        loc_label = self.loader(loc_label_path)[:,:,0]
        clf_label = self.loader(clf_label_path)[:,:,0]

        if 'train' in self.data_pro_type:
            pre_img, post_img, loc_label, clf_label = self.__transforms(True, pre_img, post_img, loc_label, clf_label)
            clf_label[clf_label == 0] = 255
        else:
            pre_img, post_img, loc_label, clf_label = self.__transforms(False, pre_img, post_img, loc_label, clf_label)
            loc_label = np.asarray(loc_label)
            clf_label = np.asarray(clf_label)

        data_idx = self.data_list[index]
        return pre_img, post_img, loc_label, clf_label, data_idx

    def __len__(self):
        return len(self.data_list)

class MultiChangeDetectionDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type
        
        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size
        
    def __transforms(self, aug, pre_img, post_img, label):
        if aug:
            pre_img, post_img, label = imutils.random_crop_new(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path, 'T1', self.data_list[index])
        post_path = os.path.join(self.dataset_path, 'T2', self.data_list[index])
        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        
        if 'inference' in self.data_pro_type:
            pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
            pre_img = np.transpose(pre_img, (2, 0, 1))

            post_img = imutils.normalize_img(post_img)  # imagenet normalization
            post_img = np.transpose(post_img, (2, 0, 1))
            
            data_idx = self.data_list[index] 
            
            return pre_img, post_img, data_idx
        
        label_path = os.path.join(self.dataset_path, 'GT', self.data_list[index])
        label = self.loader(label_path)
    
        if 'train' in self.data_pro_type:
            pre_img, post_img, label = self.__transforms(True, pre_img, post_img, label)
            label = label.astype(np.int64)
        else:
            pre_img, post_img, label = self.__transforms(False, pre_img, post_img, label)
            label = np.asarray(label)
            label = label.astype(np.int64)
    
        data_idx = self.data_list[index]        
        return pre_img, post_img, label, data_idx

    def __len__(self):
        return len(self.data_list)

class AugmentDatset(MultiChangeDetectionDatset):
    def __init__(self, dataset_path, data_list, crop_size, target_class, aug_times, max_iters=None, type='train', data_loader=img_loader):
        super().__init__(dataset_path, data_list, crop_size, max_iters, type, data_loader)
        self.crop_size = crop_size
        self.aug_times = aug_times
        self.target_class = target_class
        
        # 타겟 클래스를 포함한 영상 개수 및 인덱스 저장
        self.class_dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        cnt = 0
        
        total_samples = len(self.data_list)
        for idx in tqdm(range(total_samples)):
            label_path = os.path.join(self.dataset_path, 'GT', self.data_list[idx])
            label = self.loader(label_path)
            unique_labels, counts = np.unique(label, return_counts=True)

            # is_target 변수를 설정
            is_target = np.isin(self.target_class, unique_labels)
            if is_target.any():
                cnt += self.aug_times
                self.data_list = self.data_list + [self.data_list[idx]]*self.aug_times
                
            if aug_times >= 3:
                target_class2 = [x for x in [1,2,3,4] if x not in self.target_class]
                is_target2 = np.isin(target_class2, unique_labels)
                if is_target2.any():
                    cnt += int(self.aug_times//3)
                    self.data_list = self.data_list + [self.data_list[idx]]*int(self.aug_times//3)
            	
            for label, count in zip(unique_labels, counts):
                if is_target.any():
                    self.class_dist[label] += count * self.aug_times
                else:
                    if aug_times >=3:
                        if is_target2.any():
                            self.class_dist[label] += count * int(self.aug_times//3)
                    self.class_dist[label] += count
        
        self.count = cnt

    def __augments(self, pre_img, post_img, label):
        # 데이터 변형
        aug_pre_img, aug_post_img, aug_label = imutils.random_crop_new(pre_img, post_img, label, self.crop_size)
        aug_pre_img, aug_post_img, aug_label = imutils.random_brightness_contrast(aug_pre_img, aug_post_img, aug_label)
        aug_pre_img, aug_post_img, aug_label = imutils.random_noise(aug_pre_img, aug_post_img, aug_label)
        aug_pre_img, aug_post_img, aug_label = imutils.random_cutout(aug_pre_img, aug_post_img, aug_label)

        aug_pre_img = imutils.normalize_img(aug_pre_img)  # imagenet normalization
        aug_pre_img = np.transpose(aug_pre_img, (2, 0, 1))

        aug_post_img = imutils.normalize_img(aug_post_img)  # imagenet normalization
        aug_post_img = np.transpose(aug_post_img, (2, 0, 1))

        return aug_pre_img, aug_post_img, aug_label

    def __getitem__(self, index):
        aug_data_list = self.data_list
        pre_path = os.path.join(self.dataset_path, 'T1', aug_data_list[index])
        post_path = os.path.join(self.dataset_path, 'T2', aug_data_list[index])
        label_path = os.path.join(self.dataset_path, 'GT', aug_data_list[index])
        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        label = self.loader(label_path)

        aug_pre_img, aug_post_img, aug_label = self.__augments(pre_img, post_img, label)
        aug_label = aug_label.astype(np.int64)
        
        aug_data_idx = aug_data_list[index]

        return aug_pre_img, aug_post_img, aug_label, aug_data_idx
        
    def __len__(self):
        return self.count # init에서 계산한 데이터 수를 바탕으로 데이터셋 길이 결정

def auto_weight(class_dist, alpha=0.7):
    # 클래스 분포 출력
    formatted_dist = {label: f"{count:,}" for label, count in class_dist.items()}
    print("클래스 분포:", formatted_dist)

    total_count = sum(class_dist.values())
    num_classes = len(class_dist)
    
    # 각 클래스의 비율 계산 및 출력
    class_percentage = {label: f"{(count / total_count * 100):.2f}%" for label, count in class_dist.items()}
    print("클래스 비율:", class_percentage)

    # 제로 디비전 체크
    counts = np.array(list(class_dist.values()))
    counts[counts == 0] = 1e-6  # 최소값으로 대체하여 제로 디비전 방지

    class_weights = (total_count / (num_classes * counts)) ** alpha
    norm_weights = class_weights / np.sum(class_weights)

    print("정규화된 클래스 가중치:", norm_weights)

    return norm_weights

def make_data_loader(args, **kwargs):  # **kwargs could be omitted
    if 'SYSU' in args.dataset or 'LEVIR-CD+' in args.dataset or 'WHU' in args.dataset:
        dataset = ChangeDetectionDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        # train_sampler = DistributedSampler(dataset, shuffle=True)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=16,
                                 drop_last=False)
        return data_loader
       
    # Multi-class 변화탐지 dataloader 추가
    elif 'AIHUB' in args.dataset or 'INNOPAM' in args.dataset:
        dataset1 = MultiChangeDetectionDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        if args.augment:
            dataset2 = AugmentDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, kwargs.get('target_class'), kwargs.get('aug_times'), args.max_iters, args.type)
            dataset = ConcatDataset([dataset1, dataset2])
            if args.auto_weight:
                class_weight = auto_weight(dataset2.class_dist)  # dataset2의 class_dist를 전달
        else:
            dataset = dataset1
            if args.auto_weight:
                dataset2 = AugmentDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, [], 1, args.max_iters, args.type)
                class_weight = auto_weight(dataset2.class_dist)  # dataset1의 class_dist를 전달
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory=True, drop_last=False)

        if args.auto_weight:
            return data_loader, class_weight
        else:
            return data_loader
        
    elif 'xBD' in args.dataset:
        dataset = DamageAssessmentDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=6,
                                 drop_last=False)
        return data_loader
    
    elif 'SECOND' in args.dataset:
        dataset = SemanticChangeDetectionDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=16,
                                 drop_last=False)
        return data_loader
    
    else:
        raise NotImplementedError


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SECOND DataLoader Test")
    parser.add_argument('--dataset', type=str, default='WHUBCD')
    parser.add_argument('--max_iters', type=int, default=10000)
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--dataset_path', type=str, default='D:/Workspace/Python/STCD/data/ST-WHU-BCD')
    parser.add_argument('--data_list_path', type=str, default='./ST-WHU-BCD/train_list.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--data_name_list', type=list)

    args = parser.parse_args()

    with open(args.data_list_path, "r") as f:
        # data_name_list = f.read()
        data_name_list = [data_name.strip() for data_name in f]
    args.data_name_list = data_name_list
    train_data_loader = make_data_loader(args)
    for i, data in enumerate(train_data_loader):
        pre_img, post_img, labels, _ = data
        pre_data, post_data = Variable(pre_img), Variable(post_img)
        labels = Variable(labels)
        print(i, "个inputs", pre_data.data.size(), "labels", labels.data.size())
