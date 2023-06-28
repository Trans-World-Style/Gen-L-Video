from diffusers import DDPMScheduler, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from glv.models.unet import UNet3DConditionModel
from accelerate import infer_auto_device_map

pretrained_model_path = "weights/anything-v4.0"

noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
device_map = infer_auto_device_map(noise_scheduler, max_memory={0: "16GiB", 1: "16GiB", "cpu": "24GiB"})
print(device_map)

tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
device_map = infer_auto_device_map(tokenizer, max_memory={0: "16GiB", 1: "16GiB", "cpu": "24GiB"})
print(device_map)

text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
device_map = infer_auto_device_map(text_encoder, max_memory={0: "16GiB", 1: "16GiB", "cpu": "24GiB"})
print(device_map)

vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
device_map = infer_auto_device_map(vae, max_memory={0: "16GiB", 1: "16GiB", "cpu": "24GiB"})
print(device_map)

unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")
device_map = infer_auto_device_map(unet, max_memory={0: "16GiB", 1: "16GiB", "cpu": "24GiB"})
print(device_map)

