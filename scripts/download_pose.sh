cd weights

git clone https://huggingface.co/lllyasviel/sd-controlnet-openpose
rm -rf ./sd-controlnet-openpose/*.bin
rm -rf ./sd-controlnet-openpose/*.safetensors
wget -P ./sd-controlnet-openpose/ https://huggingface.co/lllyasviel/sd-controlnet-openpose/resolve/main/diffusion_pytorch_model.safetensors
wget -P ./sd-controlnet-openpose/ https://huggingface.co/lllyasviel/sd-controlnet-openpose/resolve/main/diffusion_pytorch_model.bin