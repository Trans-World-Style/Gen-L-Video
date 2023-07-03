cd weights

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/stabilityai/chilloutmix
rm -rf ./chilloutmix/**/*.bin
rm -rf ./chilloutmix/**/*.safetensors
wget -P ./chilloutmix/safety_checker/ https://huggingface.co/stabilityai/chilloutmix/resolve/main/safety_checker/pytorch_model.bin
wget -P ./chilloutmix/text_encoder/ https://huggingface.co/stabilityai/chilloutmix/resolve/main/text_encoder/pytorch_model.bin
wget -P ./chilloutmix/unet/ https://huggingface.co/stabilityai/chilloutmix/resolve/main/unet/diffusion_pytorch_model.bin
wget -P ./chilloutmix/vae/ https://huggingface.co/stabilityai/chilloutmix/resolve/main/vae/diffusion_pytorch_model.bin