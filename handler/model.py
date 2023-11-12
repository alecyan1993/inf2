from io import BytesIO
import traceback
import logging

logging.basicConfig(level=logging.DEBUG)
import math
import os
import requests
import tempfile
import torch
import time
import numpy as np
import safetensors
from copy import deepcopy
from PIL import Image

import torch.nn as nn
import torch_neuronx
from djl_python.inputs import Input
from djl_python.outputs import Output
from io import BytesIO
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
    StableDiffusionImg2ImgPipeline,
)
from transformers import CLIPFeatureExtractor
from diffusers import UniPCMultistepScheduler
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.cross_attention import CrossAttention

import zipfile


# - define some clases
def get_attention_scores(self, query, key, attn_mask):
    dtype = query.dtype

    if self.upcast_attention:
        query = query.float()
        key = key.float()

    if query.size() == key.size():
        attention_scores = cust_badbmm(key, query.transpose(-1, -2), self.scale)

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=1).permute(
            0, 2, 1
        )
        attention_probs = attention_probs.to(dtype)

    else:
        attention_scores = cust_badbmm(query, key.transpose(-1, -2), self.scale)

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.to(dtype)

    return attention_probs


def cust_badbmm(a, b, scale):
    bmm = torch.bmm(a, b)
    scaled = bmm * scale
    return scaled


class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(
        self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None
    ):
        out_tuple = self.unet(
            sample, timestep, encoder_hidden_states, return_dict=False
        )
        return out_tuple


class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.device = unetwrap.unet.device

    def forward(
        self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None
    ):
        sample = self.unetwrap(
            sample, timestep.float().expand((sample.shape[0],)), encoder_hidden_states
        )[0]
        return UNet2DConditionOutput(sample=sample)


class NeuronTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.neuron_text_encoder = text_encoder
        self.config = text_encoder.config
        self.dtype = torch.float32
        self.device = text_encoder.device

    def forward(self, emb, attention_mask=None):
        return [self.neuron_text_encoder(emb)["last_hidden_state"]]


class NeuronSafetyModelWrap(nn.Module):
    def __init__(self, safety_model):
        super().__init__()
        self.safety_model = safety_model

    def forward(self, clip_inputs):
        return list(self.safety_model(clip_inputs).values())


def load_pipe():
    ft_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(
        torch.float32
    )
    base_model = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", vae=ft_vae, torch_dtype=torch.float32
    )
    checkpoint = torch.load(
        "ac614f96-1082-45bf-be9d-757f2d31c174.pt", map_location="cpu"
    )

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
    pipe.safety_checker = None

    return pipe


# Load Dreamshaper v7 both of t2i and i2i
def load_pipe_both():
    ft_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(
        torch.float32
    )
    base_model_t2i = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", vae=ft_vae, torch_dtype=torch.float32
    )
    base_model_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", vae=ft_vae, torch_dtype=torch.float32
    )
    checkpoint = torch.load(
        "ac614f96-1082-45bf-be9d-757f2d31c174.pt", map_location="cpu"
    )

    pipe_t2i = deepcopy(base_model_t2i)
    pipe_i2i = deepcopy(base_model_i2i)
    if "unet_state_dict" in checkpoint:
        pipe_t2i.unet.load_state_dict(checkpoint["unet_state_dict"])
        pipe_i2i.unet.load_state_dict(checkpoint["unet_state_dict"])
    if "text_encoder_dict" in checkpoint:
        pipe_t2i.text_encoder.load_state_dict(checkpoint["text_encoder_dict"])
        pipe_i2i.text_encoder.load_state_dict(checkpoint["text_encoder_dict"])
    if "vae_state_dict" in checkpoint:
        pipe_t2i.vae.load_state_dict(checkpoint["vae_state_dict"])
        pipe_i2i.vae.load_state_dict(checkpoint["vae_state_dict"])

    feature_extractor = CLIPFeatureExtractor.from_pretrained(
        "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    )

    scheduler = UniPCMultistepScheduler.from_config(pipe_t2i.scheduler.config)

    pipe_t2i.scheduler = scheduler
    pipe_i2i.scheduler = scheduler

    pipe_t2i.feature_extractor = feature_extractor
    pipe_t2i.feature_extractor = feature_extractor

    return pipe_t2i, pipe_i2i


def load_apply_a1_safetensors_lora(
    base_model,
    lora_path,
    alpha,
):
    """Load a LORA model in a1 format from s3 model bucket and apply the weights to a copy of base_model.

    TODO: determine LORA base version and throw error

    :param base_model: base model to apply weights too
    :param lora_path: path to local cache of lora checkpoint (e.g. '/tmp/app/models/epi.pt')
    :param alpha: the weight of the applied lora
    :return: model with weights applied
    :raises: LeobeModelVersionError if base_model version does not match LORA version
    """
    lora_state_dict = safetensors.torch.load_file(lora_path)

    try:
        model = convert_lora_safetensor_to_diffusers(base_model, lora_state_dict, alpha)
    except RuntimeError as e:
        unet_size = base_model.unet.config.cross_attention_dim
        text_encoder_size = base_model.text_encoder.config.hidden_size
        error_message = f"Failed to load Lora, likely version mismatch. Unet cross_attention_dim: {unet_size}, TextEncoder hidden_size: {text_encoder_size}"
        known_lora_layer = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight"
        if known_lora_layer in lora_state_dict:
            lora_size = lora_state_dict[known_lora_layer].shape
            error_message += f", Lora layer {known_lora_layer} shape: {lora_size}"
        error_message += f", Error: {e}"
        raise RuntimeError(error_message) from e

    return model


def convert_lora_safetensor_to_diffusers(
    base_model,
    lora_state_dict,
    alpha,
    LORA_PREFIX_UNET: str = "lora_unet",
    LORA_PREFIX_TEXT_ENCODER: str = "lora_te",
):
    """Convert LORA state dict to diffusers and merge weights with base model."""

    # base_model = base_model.to('cpu', torch.float32)
    # # load LoRA weight from .safetensors
    # lora_state_dict = load_file(checkpoint_path)

    device = base_model.device
    dtype = base_model.unet.dtype

    visited = []

    # directly update weight in diffusers model
    for key in lora_state_dict:
        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = (
                key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            )
            curr_layer = base_model.text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = base_model.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(lora_state_dict[pair_keys[0]].shape) == 4:
            weight_up = (
                lora_state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(device, dtype)
            )
            weight_down = (
                lora_state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(device, dtype)
            )
            curr_layer.weight.data += alpha * torch.mm(
                weight_up, weight_down
            ).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = lora_state_dict[pair_keys[0]].to(device, dtype)
            weight_down = lora_state_dict[pair_keys[1]].to(device, dtype)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)

    return base_model


model_t2i = None
model_i2i = None


def load_model():
    pipe_t2i, pipe_i2i = load_pipe_both()

    # --- Load all compiled models ---
    text_encoder_filename = "text_encoder.pt"
    decoder_filename = "vae_decoder.pt"
    unet_filename = "unet.pt"
    post_quant_conv_filename = "vae_post_quant_conv.pt"

    # Load the compiled models onto neuron cores and let pipe_t2i, pipe_i2i share the accelerated models
    unet = NeuronUNet(UNetWrap(pipe_t2i.unet))
    device_ids = [0, 1]
    unet.unetwrap = torch_neuronx.DataParallel(
        torch.jit.load(unet_filename), device_ids, set_dynamic_batching=True
    )

    text_encoder = NeuronTextEncoder(pipe_t2i.text_encoder)
    text_encoder.neuron_text_encoder = torch.jit.load(text_encoder_filename)

    vae = pipe_t2i.vae
    vae.decoder = torch.jit.load(decoder_filename)
    vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)

    pipe_t2i.unet = unet
    pipe_i2i.unet = unet

    pipe_t2i.text_encoder = text_encoder
    pipe_i2i.text_encoder = text_encoder

    pipe_t2i.vae = vae
    pipe_i2i.vae = vae

    logging.info(f"Loading model: All Loaded Model:success")
    return pipe_t2i, pipe_i2i


def get_image(props):
    def download_image_to_temp(url):
        response = requests.get(url)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(response.content)
                return temp_file.name
        else:
            print(
                f"Failed to download image from {url}. Status code: {response.status_code}"
            )
            return None

    def delete_temp_file(file_path):
        os.remove(file_path)

    image_url = props["init_image"]
    temp_image_path = download_image_to_temp(image_url)
    init_image = Image.open(temp_image_path)
    init_image = init_image.resize((512, 768))
    delete_temp_file(temp_image_path)
    return init_image


def handle(inputs: Input):
    try:
        global model_t2i
        global model_i2i

        if not model_t2i or not model_i2i:
            logging.debug("Inputs Properties: ", inputs.get_properties)
            model_t2i, model_i2i = load_model()

        if inputs.is_empty():
            return None
        start_time = time.time()

        props = inputs.get_as_json()

        if "seed" in props.keys():
            generator = [
                torch.Generator().manual_seed(i)
                for i in range(
                    props["seed"], props["seed"] + props["num_images_per_prompt"]
                )
            ]
            props.pop("seed")
            if "init_image" in props.keys():
                init_image = get_image(props)
                props.pop("init_image")
                props["image"] = init_image
                images = model_i2i(**props, generator=generator).images
            else:
                images = model_t2i(**props, generator=generator).images
        else:
            if "init_image" in props.keys():
                init_image = get_image(props)
                props.pop("init_image")
                props["image"] = init_image
                images = model_i2i(**props).images
            else:
                images = model_t2i(**props).images

        # return multiple images with batch generation
        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            for i, img in enumerate(images):
                img_name = f"image_{i+1}.png"
                img_bytes = BytesIO()
                img.save(img_bytes, format="PNG")
                img_bytes.seek(0)
                zipf.writestr(img_name, img_bytes.read())

        buf = zip_buffer.getvalue()
        outputs = Output().add(buf).add_property("content-type", "application/zip")

        logging.info(f"handle: :TIME:TAKEN:f{ (time.time() - start_time) * 1000}:ms:::")
    except:
        excep_str = traceback.format_exc()
        logging.info(f"error:in handle():: traceback={excep_str}:")
        outputs = Output().error(excep_str)

    return outputs
