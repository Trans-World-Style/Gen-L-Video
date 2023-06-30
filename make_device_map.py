from diffusers import DDPMScheduler, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from glv.models.unet import UNet3DConditionModel
from accelerate import infer_auto_device_map
from accelerate import init_empty_weights

# pretrained_model_path = "weights/anything-v4.0"
pretrained_model_path = "./weights/anything-v4.0"

from transformers import AutoConfig, AutoModelForCausalLM

# config = AutoConfig.from_pretrained(f'{pretrained_model_path}/vae/config.json')
# config = AutoConfig.from_pretrained(f'{pretrained_model_path}', subfolder='vae')


with init_empty_weights():
    # vae = AutoModelForCausalLM.from_config(config)
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    print(vae.hf_device_map)
    device_map = infer_auto_device_map(vae, max_memory={0: "8GiB", 1: "8GiB", "cpu": "24GiB"})
    print(device_map)

# config = AutoConfig.from_pretrained(f'{pretrained_model_path}', subfolder='unet')
# with init_empty_weights():
#     # unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")
#     unet = AutoModelForCausalLM.from_config(config)
#     device_map = infer_auto_device_map(unet, max_memory={0: "8GiB", 1: "8GiB", "cpu": "24GiB"})
#     print(device_map)

