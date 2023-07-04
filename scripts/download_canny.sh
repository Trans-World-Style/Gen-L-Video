cd weights

git clone https://huggingface.co/lllyasviel/sd-controlnet-canny
rm -rf ./sd-controlnet-canny/**/*.bin
rm -rf ./sd-controlnet-canny/**/*.safetensors
wget -P ./sd-controlnet-canny/ https://huggingface.co/lllyasviel/sd-controlnet-canny/resolve/main/diffusion_pytorch_model.safetensors
wget -P ./sd-controlnet-canny/ https://huggingface.co/lllyasviel/sd-controlnet-canny/resolve/main/diffusion_pytorch_model.bin