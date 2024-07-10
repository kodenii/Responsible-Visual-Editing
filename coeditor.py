import os
import torch
from PIL import Image
# seem
from seem.modeling.BaseModel import BaseModel as BaseModel_Seem
from seem.utils.distributed import init_distributed as init_distributed_seem
from seem.modeling import build_model as build_model_seem
from task_adapter.seem.tasks import inference_seem_pano

# semantic sam
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from semantic_sam.utils.dist import init_distributed_mode
from semantic_sam.utils.arguments import load_opt_from_config_file
from semantic_sam.utils.constants import COCO_PANOPTIC_CLASSES
from task_adapter.semantic_sam.tasks import inference_semsam_m2m_auto

# sam
from segment_anything import sam_model_registry
from task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto


import numpy as np


import matplotlib.colors as mcolors


from rve_toolkits import PCP,BCP
from argparse import ArgumentParser






'''
build args
'''
semsam_cfg = "configs/semantic_sam_only_sa-1b_swinL.yaml"
seem_cfg = "configs/seem_focall_unicl_lang_v1.yaml"

semsam_ckpt = "checkpoints/swinl_only_sam_many2many.pth"
sam_ckpt = "checkpoints/sam_vit_h_4b8939.pth"
seem_ckpt = "checkpoints/seem_focall_v1.pt"

opt_semsam = load_opt_from_config_file(semsam_cfg)
opt_seem = load_opt_from_config_file(seem_cfg)
opt_seem = init_distributed_seem(opt_seem)




@torch.no_grad()
def inference(image, slider, mode, alpha, label_mode, anno_mode, text_size=640, *args, **kwargs):

      
    if slider < 1.5:
        model_name = 'seem'
    elif slider > 2.5:
        model_name = 'sam'
    else:
        if mode == 'Automatic':
            model_name = 'semantic-sam'
            if slider < 1.64:                
                level = [1]
            elif slider < 1.78:
                level = [2]
            elif slider < 1.92:
                level = [3]
            elif slider < 2.06:
                level = [4]
            elif slider < 2.2:
                level = [5]
            elif slider < 2.34:
                level = [6]
            else:
                level = [6, 1, 2, 3, 4, 5]
        else:
            model_name = 'sam'


    if label_mode == 'Alphabet':
        label_mode = 'a'
    else:
        label_mode = '1'
    hole_scale, island_scale=100,100
    text, text_part, text_thresh = '','','0.0'
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        semantic=False

        if model_name == 'semantic-sam':
            model = BaseModel(opt_semsam, build_model(opt_semsam)).from_pretrained(semsam_ckpt).eval().cuda()
            output, mask = inference_semsam_m2m_auto(model, image['image'], level, text, text_part, text_thresh, text_size, hole_scale, island_scale, semantic, label_mode=label_mode, alpha=alpha, anno_mode=anno_mode, *args, **kwargs)

        elif model_name == 'sam':
            model = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()
            if mode == "Automatic":
                output, mask = inference_sam_m2m_auto(model, image['image'], text_size, label_mode, alpha, anno_mode)

        elif model_name == 'seem':
            model = BaseModel_Seem(opt_seem, build_model_seem(opt_seem)).from_pretrained(seem_ckpt).eval().cuda()
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float32):
                    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)
            if mode == "Automatic":
                output, mask = inference_seem_pano(model, image['image'], text_size, label_mode, alpha, anno_mode)

        del model
        return output, mask



def main(concept, image, task, Granularity, alpha, api_key, seed, device="cuda"):
    w, h = image["image"].size
    try:
        img, masks, _, ori_img = PCP(concept, image, Granularity, "Automatic", alpha, "number", ['Mark', 'Mask'], api_key=api_key, fn=inference)
        edited_image = BCP(ori_img, img, masks, task, concept, api_key, seed=seed, device=device)
    except Exception as e:
        print(e)
        return np.zeros((1,1,3), dtype=np.uint8)
    edited_image = edited_image.resize((w, h), Image.Resampling.LANCZOS)
    print("finish")
    return np.array(edited_image)

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--concept", default="Elon Musk", type=str)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--task", default="privacy", choices=["safety", "fairness", "privacy"], type=str)
    parser.add_argument("--granularity", default=1.5, type=float)
    parser.add_argument("--alpha", default=0.05, type=float)
    parser.add_argument("--output-dir", default="output", type=str)
    parser.add_argument("--apikey", default=None, type=str)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    assert args.seed >= 0 and args.seed <= 2147483647
    image = Image.open(args.image)
    mask = Image.fromarray(np.zeros_like(np.array(image)))
    image = {'image':image, 'mask':mask}

    edited_image = main(args.concept, image,args.task, args.granularity, args.alpha, args.apikey, args.seed)
    edited_image = Image.fromarray(edited_image)

    os.makedirs(args.output_dir, exist_ok=True)
    edited_image.save(os.path.join(args.output_dir, os.path.basename(args.image)))
