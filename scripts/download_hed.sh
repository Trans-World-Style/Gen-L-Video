cd weights

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/lllyasviel/sd-controlnet-hed
rm -rf ./sd-controlnet-hed/*.bin
rm -rf ./sd-controlnet-hed/*.safetensors
wget -P ./sd-controlnet-hed/ https://huggingface.co/lllyasviel/sd-controlnet-hed/resolve/main/diffusion_pytorch_model.safetensors
wget -P ./sd-controlnet-hed/ https://huggingface.co/lllyasviel/sd-controlnet-hed/resolve/main/diffusion_pytorch_model.bin