import os
import ray
import imageio.v2 as imageio
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from MambaCD.changedetection.configs.config import get_config
from MambaCD.changedetection.datasets.make_data_loader import MultiChangeDetectionDatset
from MambaCD.changedetection.utils_func.visualize import visualize_class_map
from MambaCD.changedetection.models.MambaMCD import STMambaMCD

@ray.remote(num_gpus=0.2)
def process_batch(model, pre_imgs, post_imgs, device):
    with torch.no_grad():
        pre_imgs = pre_imgs.to(device).float()
        post_imgs = post_imgs.to(device)
        output_logits = model(pre_imgs, post_imgs)
        output_probs = torch.softmax(output_logits, dim=1)
        predicted_classes = torch.argmax(output_logits, dim=1)
        confidence_scores = torch.max(output_probs, dim=1)[0]
        
        return {
            'predicted': predicted_classes.cpu().numpy(),
            'confidence': confidence_scores.cpu().numpy(),
            'probs': output_probs.cpu().numpy()
        }

@ray.remote
def save_results(change_map, confidence_map, save_paths):
    change_map_image = visualize_class_map(np.squeeze(change_map))
    confidence_map = np.squeeze(confidence_map) * 100
    
    imageio.imwrite(save_paths['change'], change_map_image.astype(np.uint8))
    imageio.imwrite(save_paths['conf'], confidence_map.astype(np.uint8))
    return True

class Inferencer_v2(object):
    def __init__(self, args, img_name):
        ray.init(ignore_reinit_error=True)
        self.args = args
        self.img_name = img_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_model()
        self.setup_paths()
        
    def setup_model(self):
        self.resume = [os.path.join(self.args.model_path, pth) 
                      for pth in os.listdir(self.args.model_path) 
                      if pth.endswith('.pth')][0]
        
        config = get_config(self.args)
        self.deep_model = STMambaMCD(
            pretrained=None,
            patch_size=config.MODEL.VSSM.PATCH_SIZE,
            in_chans=config.MODEL.VSSM.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VSSM.DEPTHS,
            dims=config.MODEL.VSSM.EMBED_DIM,
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" 
                        else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )
        
        self.load_checkpoint()
        self.deep_model = self.deep_model.to(self.device)
        self.deep_model.eval()

    def infer(self):
        torch.cuda.empty_cache()
        dataset = MultiChangeDetectionDatset(
            self.infer_dataset_path, 
            self.args.data_name_list, 
            256, None, 'inference'
        )
        
        # 배치 크기를 GPU 메모리에 맞게 조정
        batch_size = 4  # GPU 메모리에 따라 조정
        val_data_loader = DataLoader(
            dataset, 
            batch_size=batch_size,
            num_workers=4,
            drop_last=False
        )
        
        futures = []
        with tqdm(total=len(dataset), desc=f"Inferencing...") as pbar:
            for data in val_data_loader:
                pre_change_imgs, post_change_imgs, names = data
                
                # 배치 처리를 Ray 태스크로 실행
                future = process_batch.remote(
                    self.deep_model, 
                    pre_change_imgs, 
                    post_change_imgs, 
                    self.device
                )
                
                # 결과 저장을 위한 경로 설정
                save_paths = [
                    {
                        'change': os.path.join(self.change_map_saved_path, 
                                             names[i][:-4] + '.png'),
                        'conf': os.path.join(self.confidence_map_saved_path, 
                                           names[i][:-4] + '_confidence.png')
                    }
                    for i in range(len(names))
                ]
                
                futures.append((future, save_paths))
                pbar.update(len(names))
        
        # 결과 수집 및 저장
        for future, save_paths in futures:
            result = ray.get(future)
            save_futures = [
                save_results.remote(
                    result['predicted'][i:i+1],
                    result['confidence'][i:i+1],
                    save_paths[i]
                )
                for i in range(len(save_paths))
            ]
            ray.get(save_futures)
        
        ray.shutdown()
