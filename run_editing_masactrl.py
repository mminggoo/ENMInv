import argparse
import json
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import os

from diffusers import DDIMScheduler

from models.masactrl.diffuser_utils import MasaCtrlPipeline
from models.masactrl.masactrl_utils import AttentionBase
from models.masactrl.masactrl_utils import regiter_attention_editor_diffusers
from models.masactrl.masactrl import MutualSelfAttentionControl
from utils.utils import load_512,txt_draw

from torchvision.io import read_image

from models.p2p.inversion import ENMInversion


def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image



class MasaCtrlEditor:
    def __init__(self, device, num_ddim_steps=50) -> None:
        self.device=device
        self.num_ddim_steps=num_ddim_steps
        # init model
        self.scheduler = DDIMScheduler(beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule="scaled_linear",
                                    clip_sample=False,
                                    set_alpha_to_one=False)
        self.model = MasaCtrlPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", scheduler=self.scheduler).to(device)
        self.model.scheduler.set_timesteps(self.num_ddim_steps)

        
    def __call__(self, 
                image_path,
                prompt_src,
                prompt_tar,
                guidance_scale,
                step=4,
                layper=10,):

        return self.edit_image_ENM_MasaCtrl(image_path,prompt_src,prompt_tar,guidance_scale,step=step,layper=layper,)


    def edit_image_ENM_MasaCtrl(self,image_path,prompt_src,prompt_tar,guidance_scale,step=4,layper=10,):
        source_image=load_image(image_path, self.device)
        image_gt = load_512(image_path)
        
        prompts=[prompt_src, prompt_tar]

        null_inversion = ENMInversion(model=self.model,
                                    num_ddim_steps=self.num_ddim_steps)

        _, _, x_stars, uncond_embeddings = null_inversion.invert(
            image_gt=image_gt, prompt=prompts, guidance_scale=guidance_scale, num_inner_steps=0)
        x_t = x_stars[-1]
        
        # hijack the attention module
        editor = MutualSelfAttentionControl(step, layper)
        regiter_attention_editor_diffusers(self.model, editor)

        # inference the synthesized image
        image_masactrl = self.model(prompts,
                            latents= x_t.expand(len(prompts), -1, -1, -1),
                            guidance_scale=[1, guidance_scale],
                            neg_prompt=prompt_src,
                            ref_intermediate_latents=x_stars,
                            prox="l0",
                            prox_step=0,
                            quantile=0.3,
                            npi_interp=1,
                            npi_step=0)
        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        out_image=np.concatenate((
                                np.array(image_instruct),
                                ((source_image[0].permute(1,2,0).detach().cpu().numpy() * 0.5 + 0.5)*255).astype(np.uint8),
                                (image_masactrl[0].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8),
                                (image_masactrl[-1].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8)),1)
        
        return Image.fromarray(out_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="scripts/walking_woman.jpg") # the editing category that needed to run
    parser.add_argument('--original_prompt', type=str, default="a woman in a hat and dress walking down a path at sunset") # the editing category that needed to run
    parser.add_argument('--editing_prompt', type=str, default="a woman in a hat and dress running down a path at sunset") # the editing category that needed to run
    parser.add_argument('--output_path', type=str, default="ENM+masactrl.jpg") # the editing category that needed to run
    args = parser.parse_args()

    masactrl_editor=MasaCtrlEditor(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') )

    original_prompt = args.original_prompt
    editing_prompt = args.editing_prompt
    image_path = args.image_path
    output_path = args.output_path

    present_image_save_path = output_path

    print(f"editing image [{image_path}]")

    edited_image = masactrl_editor(
                            image_path=image_path,
                            prompt_src=original_prompt,
                            prompt_tar=editing_prompt,
                            guidance_scale=7.5,
                            step=4,
                            layper=10,
                            )

    edited_image.save(present_image_save_path)
    
    print(f"finish")
    