import sys
sys.path.append('/workspace/Change_Detection')

import argparse
import os
import time
from tqdm import tqdm

import numpy as np

from MambaCD.changedetection.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from MambaCD.changedetection.datasets.make_data_loader import MultiChangeDetectionDatset, make_data_loader
from MambaCD.changedetection.utils_func.metrics_multi import Evaluator
from MambaCD.changedetection.utils_func.visualize import visualize_class_map
from MambaCD.changedetection.models.MambaMCD import STMambaMCD
import imageio
import MambaCD.changedetection.utils_func.lovasz_loss as L

class Inference(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

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
        self.epoch = args.max_iters // args.batch_size

        self.change_map_saved_path = os.path.join(args.result_saved_path, args.dataset, args.model_type, 'change_map')

        if not os.path.exists(self.change_map_saved_path):
            os.makedirs(self.change_map_saved_path)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, weights_only=True)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.deep_model.eval()


    def infer(self):
        torch.cuda.empty_cache()
        dataset = MultiChangeDetectionDatset(self.args.test_dataset_path, self.args.test_data_name_list, 377, None, 'test')
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=12, drop_last=False)
        torch.cuda.empty_cache()
        self.evaluator.reset()

        # vbar = tqdm(val_data_loader, ncols=50)
        with torch.no_grad():
            for itera, data in tqdm(enumerate(val_data_loader)):
                pre_change_imgs, post_change_imgs, labels, names = data
                pre_change_imgs = pre_change_imgs.cuda().float()
                post_change_imgs = post_change_imgs.cuda()
                labels = labels.cuda().long()

                output_1 = self.deep_model(pre_change_imgs, post_change_imgs)

                output_1 = output_1.data.cpu().numpy()
                output_1 = np.argmax(output_1, axis=1)
                labels = labels.cpu().numpy()
            
                self.evaluator.add_batch(labels, output_1)
            
            
                image_name = names[0][0:-4] + f'.png'

                change_map = np.squeeze(output_1)  # 멀티클래스 맵
                change_map_image = visualize_class_map(change_map)  # 클래스를 색으로 변환하는 시각화 함수
    
                imageio.imwrite(os.path.join(self.change_map_saved_path, image_name), change_map_image.astype(np.uint8))

        f1_score = self.evaluator.Pixel_F1_score()
        oa = self.evaluator.Pixel_Accuracy()
        rec = self.evaluator.Pixel_Recall_Rate()
        pre = self.evaluator.Pixel_Precision_Rate()
        iou = self.evaluator.Intersection_over_Union()
        kc = self.evaluator.Kappa_coefficient()
    
        # 클래스별 성능 지표 출력
        per_class_f1 = self.evaluator.Damage_F1_socore()  # 클래스별 F1 점수
        per_class_recall, per_class_precision = self.evaluator.calculate_per_class_metrics()[0], self.evaluator.calculate_per_class_metrics()[1]
        
        # 간단한 출력 (소수점 4자리까지)
        print(f'Recall: {rec:.4f}, Precision: {pre:.4f}, F1: {f1_score:.4f}')
        print(f'IoU: {iou:.4f}, OA: {oa:.4f}, Kappa: {kc:.4f}')
        
        print('Class-wise metrics:')
        for i, (rec, pre, f1) in enumerate(zip(per_class_recall, per_class_precision, per_class_f1)):
            print(f'Class {i}: Recall: {rec:.4f}, Precision: {pre:.4f}, F1: {f1:.4f}')
    
        print('Inference stage is done!')
            


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
    parser.add_argument('--dataset', type=str, default='LEVIR-CD+')
    parser.add_argument('--test_dataset_path', type=str, default='/home/songjian/project/datasets/SYSU/test')
    parser.add_argument('--test_data_list_path', type=str, default='/home/songjian/project/datasets/SYSU/test_list.txt')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str, default='MambaBCD_Tiny')
    parser.add_argument('--result_saved_path', type=str, default='../results')

    parser.add_argument('--resume', type=str)

    args = parser.parse_args()

    with open(args.test_data_list_path, "r") as f:
        # data_name_list = f.read()
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    infer = Inference(args)
    infer.infer()


if __name__ == "__main__":
    main()
