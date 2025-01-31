from models.p2p.scheduler_dev import DDIMSchedulerDev
from models.p2p.inversion import ENMInversion
from models.p2p.attention_control import EmptyControl, AttentionStore, make_controller
from models.p2p.p2p_guidance_forward import p2p_guidance_forward
from models.p2p.proximal_guidance_forward import proximal_guidance_forward
from diffusers import StableDiffusionPipeline
from utils.utils import load_512, latent2image, txt_draw
from PIL import Image
import numpy as np

class P2PEditor:
    def __init__(self, device, num_ddim_steps=50) -> None:
        self.device=device
        self.num_ddim_steps=num_ddim_steps
        # init model
        self.scheduler = DDIMSchedulerDev(beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule="scaled_linear",
                                    clip_sample=False,
                                    set_alpha_to_one=False)
        self.ldm_stable = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", scheduler=self.scheduler).to(device)
        self.ldm_stable.scheduler.set_timesteps(self.num_ddim_steps)

        
    def __call__(self, 
                image_path,
                prompt_src,
                prompt_tar,
                guidance_scale=7.5,
                proximal=None,
                quantile=0.7,
                use_reconstruction_guidance=False,
                recon_t=400,
                recon_lr=0.1,
                cross_replace_steps=0.4,
                self_replace_steps=0.6,
                blend_word=None,
                eq_params=None,
                is_replace_controller=False,
                use_inversion_guidance=False,
                dilate_mask=1,):
        
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = ENMInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        
        _, _, x_stars, uncond_embeddings = null_inversion.invert(
            image_gt=image_gt, prompt=prompts, guidance_scale=guidance_scale, num_inner_steps=0)
        x_t = x_stars[-1]
        

        # reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        latents, _ = p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       latent=x_t, 
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None, 
                                       uncond_embeddings=uncond_embeddings,
                                       ref_intermediate_latents=x_stars,)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        return Image.fromarray(np.concatenate((image_instruct, image_gt, images[-2], images[-1]),axis=1))
        
