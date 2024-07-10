import numpy as np
import cv2
import torch
import numpy as np
import PIL.Image as Image
from diffusers import StableDiffusionInpaintPipeline
import time
import json
from task_adapter.utils.visualizer import Visualizer
import networkx as nx
from detectron2.data import MetadataCatalog
import gradio as gr
import re
from gpt4v import ConceptSearch,request_gpt4v
from torchvision import transforms


with open("instructions/general.json") as f:
    general_instructions=json.load(f)

with open("instructions/tasks.json") as f:
    tasks_instructions=json.load(f)
metadata = MetadataCatalog.get('coco_2017_train_panoptic')
def dilate_mask(mask, kernel_size=15):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((kernel_size, kernel_size), np.uint8),
        iterations=1
    )
    return mask

def sd_inpainting(img: np.ndarray, mask: np.ndarray, text: str, device="cuda"):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32,
    ).to(device)
    w, h, c = img.shape
    inpainted_img = pipe(
        prompt=text,
        image=Image.fromarray(img),
        mask_image=Image.fromarray(mask)
    ).images[0]
    inpainted_img = inpainted_img.resize((h,w), Image.Resampling.LANCZOS)
    return np.array(inpainted_img)

def get_modification_generation_inst(task_name, concept):
    if task_name == "safety":
        goal = tasks_instructions[task_name]["goal"].format(concept = concept)
    else:
        goal = tasks_instructions[task_name]["goal"]
    steps = tasks_instructions[task_name]["steps"].format(concept = concept)
    modification_generation_prompt = general_instructions["modification generation"].format(task_name = task_name,goal = goal,steps = steps)
    return modification_generation_prompt

def get_focus_generation_inst(concept, knowledge):
    return general_instructions["focus generation"].format(knowledge=knowledge,concept=concept)

class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", tool="sketch", interactive=True, **kwargs)

    def preprocess(self, x):
        return super().preprocess(x)

def MasksMerge(masks):
    G = nx.Graph()
    new_masks = []
    for i,_ in enumerate(masks):
        G.add_node(i)
    for i in range(len(masks)):
        for j in range(i+1, len(masks)):
            product = masks[i] * masks[j]
            if product.__contains__(1):
                G.add_edge(i,j)
    for c in nx.connected_components(G):
        tmp = np.zeros_like(masks[0])
        for m in c:
            tmp = tmp | masks[m]
        new_masks.append(tmp)
    return new_masks

def Mask_Reorder(image, masks, label_mode, alpha, anno_mode):
    visual = Visualizer(image, metadata=metadata)
    sorted_anns = sorted(masks, key=(lambda x: np.count_nonzero(x)), reverse=True)
    mask_map = np.zeros(image.shape, dtype=np.uint8)    
    for i, mask in enumerate(sorted_anns):
        demo = visual.draw_binary_mask_with_number(mask, text=str(i+1), label_mode=label_mode, alpha=alpha, anno_mode=anno_mode)
        mask_map[mask == 1] = i+1
    im = demo.get_image()
    return im

def get_dilate_num(x):
    if x<1.5:
         dilate_num = -500*(x-1.36)+85 
    elif x < 1.64:                
        dilate_num = -500*(x-1.5)+85 
    elif x < 1.78:
        dilate_num = -500*(x-1.64)+85 
    elif x < 1.92:
        dilate_num = -500*(x-1.78)+85 
    elif x < 2.06:
        dilate_num = -500*(x-1.92)+85 
    elif x < 2.2:
        dilate_num = -500*(x-2.06)+85 
    elif x < 2.34:
        dilate_num = -500*(x-2.20)+85 
    elif x<2.5:
        dilate_num = -500*(x-2.34)+85 
    else: 
        dilate_num = -500*(x-2.5)+85
    dilate_num = dilate_num if dilate_num>0 else 0
    dilate_num = int(round(dilate_num))
    return dilate_num

def get_knowledge(concept, api_key):
    with open("concept.json") as f:
        cs = json.load(f)
    if len(cs)==0 or not concept in cs.keys():
        concept_definition = ConceptSearch(concept, api_key=api_key)
        concept_definition = concept_definition.replace("\n","")
        judge = re.search(r'\[((\s)*".*"(\s)*,?(\s)*)*\]', concept_definition)
        concept_definition = eval(judge.group())
        cs[concept] = concept_definition[-1]
        with open("concept.json", "w+") as f:
            f.write(json.dumps(cs, indent=4))
    else:
        concept_definition=[cs[concept]]
    return concept_definition[-1]

def gpt4v_response(message, visual_prompt, api_key): 
    try:
        res = request_gpt4v(message, Image.fromarray(visual_prompt), api_key)
        return res
    except Exception as e:
        return None

def convert_idx_to_masks(res, all_masks):
    res = eval(res)
    n = len(all_masks)
    res = [r for r in res if int(r)<=n]
    sections = []
    for _, r in enumerate(res):
        mask_i = all_masks[int(r)-1]['segmentation']
        sections.append((mask_i, r))
    return sections



def PCP(concept, image, slider, mode, alpha, label_mode, anno_mode, text_size=640, api_key=None, fn=None):
    visual_prompt,all_masks = fn(image, slider, mode, alpha, label_mode, anno_mode, text_size)

    knowledge = get_knowledge(concept, api_key)
    guidance = get_focus_generation_inst(concept, knowledge)
    idx = 0
    flag = 0
    judge = None
    while True:
        idx += 1
        if idx>3:
            break
        res = gpt4v_response(guidance, visual_prompt, api_key)
        if res is None:
            break
        judge = re.search(r'\[\s*(\s*\d\s*(, )?\s*)*\s*\]', res)
        if not judge is None and eval(judge.group()) != []:
            flag = 1
            break
    if flag == 0 and judge is None:
        raise Exception("failed to detect")
    
    res = judge.group()
    sections = convert_idx_to_masks(res, all_masks)
    t = []
    t.append(transforms.Resize(int(text_size), interpolation=Image.BICUBIC))
    transform1 = transforms.Compose(t)
    original_image = np.array(transform1(image['image']))
    processed_img = original_image

    masks = [i[0] for i in sections]
    dilate_num = get_dilate_num(slider)
    masks = [dilate_mask(mask, dilate_num ) for mask in masks]

    im = None
    if len(masks)>0:
        masks = MasksMerge(masks)
        im = Mask_Reorder(processed_img, masks, label_mode, alpha, anno_mode)

    for m in masks:
        m = np.ones_like(m)-m
        m = np.stack([m, m, m]).transpose(1,2,0)
        processed_img = processed_img * m
    return processed_img, masks, im, original_image

def generating_correction_prompts(task, concept, visual_prompt, api_key):
    assert task in ["safety", "fairness", "privacy"]
    des = get_modification_generation_inst(task, concept)
    res = None
    if visual_prompt is not None:
        vp = Image.fromarray(visual_prompt)
        res = request_gpt4v(des, vp, api_key)
        res = res.replace("\n","").replace("python","")
    return res

def BCP(original_img, img, masks, task, concept, api_key, seed=None, device="cuda"):

    binary_masks = [i.astype(np.uint8) for i in masks]
    masks = [i.astype(np.uint8)*255 for i in masks]

    filled_images = []

    text_prompt = []
    print(f"the number of regions: {len(masks)}")
    for mask in masks:
        x, y, w, h = cv2.boundingRect(mask)
        tmp = original_img[y:y+h, x:x+w]
        txt = generating_correction_prompts(task, concept, tmp, api_key)
        text_prompt.append(txt)
    
    for idx, mask in enumerate(masks):
        if seed is not None:
            torch.manual_seed(seed)
        img_filled = sd_inpainting(
            original_img, mask, text_prompt[idx], device=device)
        filled_images.append(img_filled)
    img = original_img

    for idx,m in enumerate(binary_masks):
        m = np.stack([m, m, m]).transpose(1,2,0)
        img = img * (1 - m) + filled_images[idx] * m
    img = Image.fromarray(img)
    return img

