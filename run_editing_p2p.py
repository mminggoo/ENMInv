import os 
import numpy as np
import argparse
import json
from PIL import Image
import torch

from models.p2p_editor import P2PEditor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="scripts/car.jpg") # the editing category that needed to run
    parser.add_argument('--original_prompt', type=str, default="a colorful car is parked on the street") # the editing category that needed to run
    parser.add_argument('--editing_prompt', type=str, default="→ A colorful motorcycle is parked on the street") # the editing category that needed to run
    parser.add_argument('--blended_word', type=str, default="car motorcycle") # the editing category that needed to run
    parser.add_argument('--output_path', type=str, default="ENM+p2p.jpg") # the editing category that needed to run
    args = parser.parse_args()
    
    original_prompt = args.original_prompt
    editing_prompt = args.editing_prompt
    image_path = args.image_path
    blended_word = args.blended_word.split(" ") if args.blended_word != "" else []
    output_path = args.output_path
    
    p2p_editor=P2PEditor(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),num_ddim_steps=50)
    
    
    present_image_save_path = output_path
    print(f"editing image [{image_path}]")
    
    edited_image = p2p_editor(
                            image_path=image_path,
                            prompt_src=original_prompt,
                            prompt_tar=editing_prompt,
                            guidance_scale=7.5,
                            cross_replace_steps=0.4,
                            self_replace_steps=0.6,
                            blend_word=(((blended_word[0], ),
                                        (blended_word[1], ))) if len(blended_word) else None,
                            eq_params={
                                "words": (blended_word[1], ),
                                "values": (2, )
                            } if len(blended_word) else None,
                            proximal="l0",
                            quantile=0.75,
                            use_inversion_guidance=True,
                            recon_lr=1,
                            recon_t=400,
                            )

    edited_image.save(present_image_save_path)
    
    print(f"finish")