import os
import sys

import imageio.v2 as imageio
import argparse
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from MambaCD.changedetection.configs.config import get_config
from MambaCD.changedetection.datasets.make_data_loader import MultiChangeDetectionDatset, make_data_loader
from MambaCD.changedetection.utils_func.visualize import visualize_class_map
from MambaCD.changedetection.models.MambaMCD import STMambaMCD

class Inferencer(object):
    def __init__(self, args, mode, img_name, num_workers=8):
        self.args = args
        self.mode = mode
        self.img_name = img_name
        self.num_workers = num_workers
        self.resume = [os.path.join(args.model_path, pth) for pth in os.listdir(args.model_path) if pth.endswith('.pth')][0]
        config = get_config(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.deep_model = STMambaMCD(
            pretrained=None,
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
        self.deep_model = self.deep_model.to(self.device)

        if self.resume is not None:
            if not os.path.isfile(self.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(self.resume))
            checkpoint = torch.load(self.resume, weights_only=True)
            if 'model' in checkpoint:
                model_dict = {}
                state_dict = self.deep_model.state_dict()
                for k, v in checkpoint['model'].items():
                    if k in state_dict:
                        model_dict[k] = v
                state_dict.update(model_dict)
                self.deep_model.load_state_dict(state_dict)
                print("=> Loaded model weights from '{}'".format(self.resume))
            elif 'model_state_dict' in checkpoint:
                model_dict = {}
                state_dict = self.deep_model.state_dict()
                for k, v in checkpoint['model_state_dict'].items():
                    if k in state_dict:
                        model_dict[k] = v
                state_dict.update(model_dict)
                self.deep_model.load_state_dict(state_dict)
                print("=> Loaded model weights from '{}'".format(self.resume))
            else:
                # 가중치만 있는 경우
                print("=> No model state found, loading weights only")
                state_dict = self.deep_model.state_dict()
                for k, v in checkpoint.items():
                    if k in state_dict:
                        state_dict[k] = v
                self.deep_model.load_state_dict(state_dict)

        self.infer_dataset_path = os.path.join(args.output_path, 'patches', self.img_name)
        self.result_saved_path = os.path.join(args.output_path, 'results', self.img_name)

        self.change_map_saved_path = os.path.join(self.result_saved_path, 'pred')
        self.confidence_map_saved_path = os.path.join(self.result_saved_path, 'confidence')

        if not os.path.exists(self.change_map_saved_path):
            os.makedirs(self.change_map_saved_path)

        if not os.path.exists(self.confidence_map_saved_path):
            os.makedirs(self.confidence_map_saved_path)

        self.deep_model.eval()

    def infer(self):
        torch.cuda.empty_cache()
        dataset = MultiChangeDetectionDatset(self.infer_dataset_path, self.args.data_name_list, 256, None, 'inference')
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=self.num_workers, pin_memory=False, drop_last=False)
        torch.cuda.empty_cache()

        with torch.no_grad():
            for data in tqdm(val_data_loader, desc=f"Inferencing {self.img_name}...", dynamic_ncols=True, leave=True, file=sys.stderr):
                pre_change_imgs, post_change_imgs, names = data
                pre_change_imgs = pre_change_imgs.to(self.device).float()
                post_change_imgs = post_change_imgs.to(self.device)

                # 모델의 출력 가져오기
                output_logits = self.deep_model(pre_change_imgs, post_change_imgs)

                output_1 = output_logits.data.cpu().numpy()
                predicted_classes = np.argmax(output_1, axis=1)

                # 확률 분포로 변환
                output_probs = torch.softmax(output_logits, dim=1)

                # argmax를 사용하여 클래스를 결정
                output_classes = output_probs.data.cpu().numpy()

                # confidence score 계산 (각 픽셀에서 최대 확률)
                confidence_scores = np.max(output_classes, axis=1)  # 각 픽셀에서 가장 높은 확률 (confidence score)
                confidence_map = np.squeeze(confidence_scores) * 100  # 1채널 신뢰도 맵
            
                image_name = names[0][0:-4] + f'.png'
                confidence_map_name = names[0][0:-4] + f'_confidence.png'

                change_map = np.squeeze(predicted_classes)  # 멀티클래스 맵
                change_map_image = visualize_class_map(change_map, self.mode)  # 클래스를 색으로 변환하는 시각화 함수
            
                imageio.imwrite(os.path.join(self.change_map_saved_path, image_name), change_map_image.astype(np.uint8))
                imageio.imwrite(os.path.join(self.confidence_map_saved_path, confidence_map_name), confidence_map.astype(np.uint8))

                # 불필요한 변수 삭제 및 캐시 정리
                del pre_change_imgs, post_change_imgs, output_logits
                torch.cuda.empty_cache()                