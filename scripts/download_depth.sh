cd weights

git clone https://huggingface.co/lllyasviel/sd-controlnet-depth
rm -rf ./sd-controlnet-depth/*.bin
rm -rf ./sd-controlnet-depth/*.safetensors
wget -P ./sd-controlnet-depth/ https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/diffusion_pytorch_model.safetensors
wget -P ./sd-controlnet-depth/ https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/diffusion_pytorch_model.bin