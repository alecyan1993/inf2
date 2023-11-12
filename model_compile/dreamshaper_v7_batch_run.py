import os

os.environ["NEURON_FUSE_SOFTMAX"] = "1"

import torch
import torch.nn as nn
import torch_neuronx
import numpy as np
from copy import deepcopy

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import time
import copy
from IPython.display import clear_output

from diffusers import StableDiffusionPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
)
from transformers import CLIPFeatureExtractor
from diffusers import UniPCMultistepScheduler

from diffusers.models.cross_attention import CrossAttention

clear_output(wait=False)


# For saving compiler artifacts
COMPILER_WORKDIR_ROOT = "dreamshaper_v7_512_dynamic"

# Base model
BASE_MODEL = "runwayml/stable-diffusion-v1-5"

# Finetuned VAE and Dreamshaper v7 model path
FINETUNED_VAE = "stabilityai/sd-vae-ft-mse"
DREAMSHAPER_V7_MODEL_PATH = "models/ac614f96-1082-45bf-be9d-757f2d31c174.pt"


# Load Dreamshaper v7
def load_pipe():
    ft_vae = AutoencoderKL.from_pretrained(FINETUNED_VAE).to(torch.float32)
    base_model = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, vae=ft_vae, torch_dtype=torch.float32
    )
    checkpoint = torch.load(DREAMSHAPER_V7_MODEL_PATH, map_location="cpu")

    pipe = deepcopy(base_model)
    if "unet_state_dict" in checkpoint:
        pipe.unet.load_state_dict(checkpoint["unet_state_dict"])
    if "text_encoder_dict" in checkpoint:
        pipe.text_encoder.load_state_dict(checkpoint["text_encoder_dict"])
    if "vae_state_dict" in checkpoint:
        pipe.vae.load_state_dict(checkpoint["vae_state_dict"])

    feature_extractor = CLIPFeatureExtractor.from_pretrained(
        "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    )

    scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.scheduler = scheduler
    pipe.feature_extractor = feature_extractor

    return pipe


def load_compiled_model():
    # --- Load all compiled models ---
    pipe = load_pipe()

    text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, "text_encoder/model.pt")
    unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, "unet/model.pt")
    decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, "vae_decoder/model.pt")
    post_quant_conv_filename = os.path.join(
        COMPILER_WORKDIR_ROOT, "vae_post_quant_conv/model.pt"
    )
    safety_model_neuron_filename = os.path.join(
        COMPILER_WORKDIR_ROOT, "safety_model_neuron/model.pt"
    )

    # Load the compiled UNet onto two neuron cores.
    pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
    device_ids = [0, 1]
    pipe.unet.unetwrap = torch_neuronx.DataParallel(
        torch.jit.load(unet_filename), device_ids, set_dynamic_batching=True
    )

    # Load other compiled models onto a single neuron core.
    pipe.text_encoder = NeuronTextEncoder(pipe.text_encoder)
    pipe.text_encoder.neuron_text_encoder = torch.jit.load(text_encoder_filename)
    pipe.vae.decoder = torch.jit.load(decoder_filename)
    pipe.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)
    return pipe


if __name__ == "__main__":
    default_model_args = {
        "prompt": "a cat",
        "negative_prompt": "",
        "num_inference_steps": 10,
        "num_images_per_prompt": 2,
    }
    pipe = load_compiled_model()
    images = pipe(**default_model_args)
