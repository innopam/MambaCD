import sys
sys.path.append('/workspace/Change_Detection')

import argparse
import os
import csv
import time
import numpy as np

from MambaCD.changedetection.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import lr_scheduler  # 추가된 부분
from tqdm import tqdm

from MambaCD.changedetection.datasets.make_data_loader import MultiChangeDetectionDatset, make_data_loader
from MambaCD.changedetection.utils_func.metrics_multi import Evaluator
from MambaCD.changedetection.models.MambaMCD import STMambaMCD

import MambaCD.changedetection.utils_func.lovasz_loss as L

class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        self.train_data_loader = make_data_loader(args)

        self.evaluator = Evaluator(num_class=5)

        self.deep_model = STMambaMCD(
            pretrained=args.pretrained_weight_path,
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            ) 
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

        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
        # Learning Rate Scheduler
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optim, T_max=self.epoch)

    def training(self):
        best_kc = 0.0
        temp_iter, best_loss, threshold = 0, 1, 0.01
        best_round = []
        elem_num = len(self.train_data_loader)
        train_enumerator = enumerate(self.train_data_loader)
        torch.cuda.empty_cache()

        # Mixed Precision Scaler 객체 생성
        scaler = GradScaler()

        log_dir = os.path.join(self.model_save_path,'logs')  # 로그 저장 폴더
        os.makedirs(log_dir, exist_ok=True)
        loss_file = os.path.join(log_dir, 'training_loss.csv')

        # CSV 파일에 헤더 작성
        with open(loss_file, mode='w') as f:
            writer = csv.writer(f)
            writer.writerow(['Iteration', 'Loss'])  # 손실 기록 헤더

        for _ in tqdm(range(elem_num)):
            itera, data = train_enumerator.__next__()
            pre_change_imgs, post_change_imgs, labels, _ = data
            pre_change_imgs = pre_change_imgs.cuda()
            post_change_imgs = post_change_imgs.cuda()
            labels = labels.cuda().long()

            with autocast():
                output_1 = self.deep_model(pre_change_imgs, post_change_imgs)
                self.optim.zero_grad()
                class_weights = torch.tensor([0.0010, 0.2999, 0.5664, 0.0511, 0.0816], device=output_1.device) # class_weight 설정
                ce_loss_1 = F.cross_entropy(output_1, labels, weight=class_weights)
                lovasz_loss = L.lovasz_softmax(F.softmax(output_1, dim=1), labels, class_weights=class_weights)
                main_loss = ce_loss_1 + 0.75 * lovasz_loss
                final_loss = main_loss.mean()

             # 손실 역전파와 옵티마이저 스텝도 Mixed Precision으로
            scaler.scale(final_loss).backward()
            scaler.step(self.optim)
            scaler.update()

            if (itera+1) % len(self.train_data_loader) == 0:
                self.scheduler.step()

            # 손실 값 저장
            with open(loss_file, mode='a') as f:
                writer = csv.writer(f)
                writer.writerow([itera + 1, final_loss.item()])

            # Best checkpoint save
            if final_loss.item() < best_loss:
                if (best_loss - final_loss.item()) < threshold:
                    best_loss = final_loss.item()
                    temp_weight = self.deep_model.state_dict()
                    temp_iter = itera+1

            if (itera + 1) % 100 == 0:
                print(f'iter is {itera + 1}, overall loss is {final_loss.item():.4f}')
                print(f'Best model epoch is {temp_iter}')
                if (itera + 1) % 5000 == 0:
                    torch.save(temp_weight, os.path.join(self.model_save_path, f'best_model_{temp_iter}.pth'))
                    self.deep_model.eval()
                    rec, pre, oa, f1_score, iou, kc = self.validation(itera)
                    if kc > best_kc:
                        torch.save(self.deep_model.state_dict(),
                                   os.path.join(self.model_save_path, f'{itera + 1}_model.pth'))
                        best_kc = kc
                        best_round = itera
                        best_val = [rec, pre, oa, f1_score, iou, kc]

        print('The accuracy of the best round is ', best_round)
        print("Best round's validation is ", best_val)

    def validation(self, itera):
        print('---------starting evaluation-----------')
        self.evaluator.reset()
        dataset = MultiChangeDetectionDatset(self.args.test_dataset_path, self.args.test_data_name_list, 377, None, 'test')
        val_data_loader = DataLoader(dataset, batch_size=6, num_workers=6, drop_last=False)
        torch.cuda.empty_cache()

        log_dir = os.path.join(self.model_save_path,'logs')
        val_file = os.path.join(log_dir, f'validation_metrics_{itera+1}.csv')
        
        with open(val_file, mode='w') as f:
            writer = csv.writer(f)
            writer.writerow(['Recall', 'Precision', 'F1', 'IoU', 'OA', 'Kappa'])  # 검증 기록 헤더
        #vbar = tqdm(val_data_loader, ncols=50)
        with torch.no_grad():
            for itera, data in tqdm(enumerate(val_data_loader)):
                pre_change_imgs, post_change_imgs, labels, _ = data
                pre_change_imgs = pre_change_imgs.cuda().float()
                post_change_imgs = post_change_imgs.cuda()
                labels = labels.cuda().long()

                output_1 = self.deep_model(pre_change_imgs, post_change_imgs)

                output_1 = output_1.data.cpu().numpy()
                output_1 = np.argmax(output_1, axis=1)
                labels = labels.cpu().numpy()

                self.evaluator.add_batch(labels, output_1)
                
        # 각 클래스별 Precision, Recall, F1, IoU 계산
        f1_scores = self.evaluator.Pixel_F1_score()  # 각 클래스별 F1 score
        oa = self.evaluator.Pixel_Accuracy()  # 전체 정확도
        rec = self.evaluator.Pixel_Recall_Rate()  # 각 클래스별 Recall
        pre = self.evaluator.Pixel_Precision_Rate()  # 각 클래스별 Precision
        iou = self.evaluator.Intersection_over_Union()  # 각 클래스별 IoU
        kc = self.evaluator.Kappa_coefficient()  # Kappa coefficient

        # 검증 결과 저장
        with open(val_file, mode='a') as f:
            writer = csv.writer(f)
            writer.writerow([rec, pre, f1_scores, iou, oa, kc])
            
        print(f'Recall: {np.mean(rec):.4f}, Precision: {np.mean(pre):.4f}, F1: {np.mean(f1_scores):.4f}')
        print(f'IoU: {np.mean(iou):.4f}, OA: {oa:.4f}, Kappa: {kc:.4f}')
    
        return np.mean(rec), np.mean(pre), oa, np.mean(f1_scores), np.mean(iou), kc


def main():
    parser = argparse.ArgumentParser(description="Training on SYSU/LEVIR-CD+/WHU-CD dataset")
    parser.add_argument('--cfg', type=str, default='/home/songjian/project/MambaCD/VMamba/classification/configs/vssm1/vssm_base_224.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str)
    parser.add_argument('--dataset', type=str, default='SYSU')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--train_dataset_path', type=str, default='/home/songjian/project/datasets/SYSU/train')
    parser.add_argument('--train_data_list_path', type=str, default='/home/songjian/project/datasets/SYSU/train_list.txt')
    parser.add_argument('--test_dataset_path', type=str, default='/home/songjian/project/datasets/SYSU/test')
    parser.add_argument('--test_data_list_path', type=str, default='/home/songjian/project/datasets/SYSU/test_list.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str, default='MambaBCD')
    parser.add_argument('--model_param_path', type=str, default='../saved_models')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    args = parser.parse_args()
    with open(args.train_data_list_path, "r") as f:
        # data_name_list = f.read()
        data_name_list = [data_name.strip() for data_name in f]
    args.train_data_name_list = data_name_list

    with open(args.test_data_list_path, "r") as f:
        # data_name_list = f.read()
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()
