import torch

from models.p2p.attention_control import register_attention_control
from models.p2p.proximal_guidance_forward import proximal_guidance_diffusion_step
from utils.utils import init_latent

def p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents



@torch.no_grad()
def p2p_guidance_forward(
    model,
    prompt,
    controller,
    num_inference_steps: int = 50,
    guidance_scale = 7.5,
    generator = None,
    latent = None,
    uncond_embeddings=None,
    ref_intermediate_latents=None,
):
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)

    for i, t in enumerate(model.scheduler.timesteps):
        if ref_intermediate_latents is not None:
            # note that the batch_size >= 2
            latents_ref = ref_intermediate_latents[-1 - i]
            _, latents_cur = latents.chunk(2)
            latents = torch.cat([latents_ref, latents_cur])

        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])

        latents = proximal_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False,
                                           edit_stage=True, prox="l0", quantile=0.6,
                                           image_enc=None, recon_lr=1, recon_t=400,
                                           inversion_guidance=True, x_stars=ref_intermediate_latents, i=i, dilate_mask=1)


    return latents, latent
