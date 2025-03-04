mkdir weights
cd weights

# Vit-H SAM model.
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Part Grounding Swin-Base Model.
wget https://github.com/Cheems-Seminar/segment-anything-and-name-it/releases/download/v1.0/swinbase_part_0a0000.pth

# Grounding DINO Model. 
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

# Download the Pretrained T2I-Adapters
git clone https://huggingface.co/TencentARC/T2I-Adapter
cd T2I-Adapter
git lfs install  # if git-lfs is not installed
git lfs pull 
cd ..

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/xyn-ai/anything-v4.0
rm -rf ./anything-v4.0/**/*.bin
rm -rf ./anything-v4.0/**/*.safetensors
wget -P ./anything-v4.0/safety_checker/ https://huggingface.co/xyn-ai/anything-v4.0/resolve/main/safety_checker/model.safetensors
wget -P ./anything-v4.0/safety_checker/ https://huggingface.co/xyn-ai/anything-v4.0/resolve/main/safety_checker/pytorch_model.bin
wget -P ./anything-v4.0/text_encoder/ https://huggingface.co/xyn-ai/anything-v4.0/resolve/main/text_encoder/model.safetensors
wget -P ./anything-v4.0/text_encoder/ https://huggingface.co/xyn-ai/anything-v4.0/resolve/main/text_encoder/pytorch_model.bin
wget -P ./anything-v4.0/unet/ https://huggingface.co/xyn-ai/anything-v4.0/resolve/main/unet/diffusion_pytorch_model.bin
wget -P ./anything-v4.0/unet/ https://huggingface.co/xyn-ai/anything-v4.0/resolve/main/unet/diffusion_pytorch_model.safetensors
wget -P ./anything-v4.0/vae/ https://huggingface.co/xyn-ai/anything-v4.0/resolve/main/vae/diffusion_pytorch_model.bin
wget -P ./anything-v4.0/vae/ https://huggingface.co/xyn-ai/anything-v4.0/resolve/main/vae/diffusion_pytorch_model.safetensors

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/stabilityai/stable-diffusion-2-inpainting
rm -rf ./stable-diffusion-2-inpainting/**/*.bin
rm -rf ./stable-diffusion-2-inpainting/**/*.safetensors
wget -P ./stable-diffusion-2-inpainting/text_encoder/ https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/resolve/main/text_encoder/model.safetensors
wget -P ./stable-diffusion-2-inpainting/text_encoder/ https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/resolve/main/text_encoder/pytorch_model.bin
wget -P ./stable-diffusion-2-inpainting/unet/ https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/resolve/main/unet/diffusion_pytorch_model.bin
wget -P ./stable-diffusion-2-inpainting/unet/ https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/resolve/main/unet/diffusion_pytorch_model.safetensors
wget -P ./stable-diffusion-2-inpainting/vae/ https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/resolve/main/vae/diffusion_pytorch_model.bin
wget -P ./stable-diffusion-2-inpainting/vae/ https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/resolve/main/vae/diffusion_pytorch_model.safetensors

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/stabilityai/stable-diffusion-2-depth
rm -rf ./stable-diffusion-2-depth/**/*.bin
rm -rf ./stable-diffusion-2-depth/**/*.safetensors
wget -P ./stable-diffusion-2-depth/depth_estimator/ https://huggingface.co/stabilityai/stable-diffusion-2-depth/resolve/main/depth_estimator/model.safetensors
wget -P ./stable-diffusion-2-depth/depth_estimator/ https://huggingface.co/stabilityai/stable-diffusion-2-depth/resolve/main/depth_estimator/pytorch_model.bin
wget -P ./stable-diffusion-2-depth/text_encoder/ https://huggingface.co/stabilityai/stable-diffusion-2-depth/resolve/main/text_encoder/pytorch_model.bin
wget -P ./stable-diffusion-2-depth/unet/ https://huggingface.co/stabilityai/stable-diffusion-2-depth/resolve/main/unet/diffusion_pytorch_model.bin
wget -P ./stable-diffusion-2-depth/unet/ https://huggingface.co/stabilityai/stable-diffusion-2-depth/resolve/main/unet/diffusion_pytorch_model.safetensors
wget -P ./stable-diffusion-2-depth/vae/ https://huggingface.co/stabilityai/stable-diffusion-2-depth/resolve/main/vae/diffusion_pytorch_model.bin
wget -P ./stable-diffusion-2-depth/vae/ https://huggingface.co/stabilityai/stable-diffusion-2-depth/resolve/main/vae/diffusion_pytorch_model.safetensors


#GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/stabilityai/stable-diffusion-2-depth
#rm -rf ./stable-diffusion-2-depth/**/*.bin
#rm -rf ./stable-diffusion-2-depth/**/*.safetensors
#wget -P ./stable-diffusion-2-depth/depth_estimator/ https://huggingface.co/stabilityai/stable-diffusion-2-depth/resolve/main/depth_estimator/model.safetensors
#wget -P ./stable-diffusion-2-depth/depth_estimator/ https://huggingface.co/stabilityai/stable-diffusion-2-depth/resolve/main/depth_estimator/pytorch_model.bin
#wget -P ./stable-diffusion-2-depth/text_encoder/ https://huggingface.co/stabilityai/stable-diffusion-2-depth/resolve/main/text_encoder/pytorch_model.bin
#wget -P ./stable-diffusion-2-depth/unet/ https://huggingface.co/stabilityai/stable-diffusion-2-depth/resolve/main/unet/diffusion_pytorch_model.bin
#wget -P ./stable-diffusion-2-depth/unet/ https://huggingface.co/stabilityai/stable-diffusion-2-depth/resolve/main/unet/diffusion_pytorch_model.safetensors
#wget -P ./stable-diffusion-2-depth/vae/ https://huggingface.co/stabilityai/stable-diffusion-2-depth/resolve/main/vae/diffusion_pytorch_model.bin
#wget -P ./stable-diffusion-2-depth/vae/ https://huggingface.co/stabilityai/stable-diffusion-2-depth/resolve/main/vae/diffusion_pytorch_model.safetensors


git clone https://huggingface.co/CompVis/stable-diffusion-v1-4
wget https://huggingface.co/YueMafighting/FollowYourPose_v1/resolve/main/followyourpose_checkpoint-1000/pytorch_model.bin
rm stable-diffusion-v1-4/unet/diffusion_pytorch_model.bin stable-diffusion-v1-4/unet/diffusion_pytorch_model.safetensors
rm stable-diffusion-v1-4/text_encoder/model.safetensors stable-diffusion-v1-4/vae/diffusion_pytorch_model.safetensors 
mv pytorch_model.bin stable-diffusion-v1-4/unet/diffusion_pytorch_model.bin
wget https://huggingface.co/YueMafighting/FollowYourPose_v1/resolve/main/pose_encoder.pth
mv pose_encoder.pth T2I-Adapter/models
mv stable-diffusion-v1-4 follow-your-pose

git clone https://huggingface.co/shgao/edit-anything-v0-3


declare -a files=(
    "./sam_vit_h_4b8939.pth"
    "./swinbase_part_0a0000.pth"
    "./groundingdino_swinb_cogcoor.pth"
    "./anything-v4.0/safety_checker/model.safetensors"
    "./anything-v4.0/safety_checker/pytorch_model.bin"
    "./anything-v4.0/text_encoder/model.safetensors"
    "./anything-v4.0/text_encoder/pytorch_model.bin"
    "./anything-v4.0/unet/diffusion_pytorch_model.bin"
    "./anything-v4.0/unet/diffusion_pytorch_model.safetensors"
    "./anything-v4.0/vae/diffusion_pytorch_model.bin"
    "./anything-v4.0/vae/diffusion_pytorch_model.safetensors"
    "./stable-diffusion-2-inpainting/text_encoder/model.safetensors"
    "./stable_diffusion-2-inpainting/text_encoder/pytorch_model.bin"
    "./stable_diffusion-2-inpainting/unet/diffusion_pytorch_model.bin"
    "./stable_diffusion-2-inpainting/unet/diffusion_pytorch_model.safetensors"
    "./stable_diffusion-2-inpainting/vae/diffusion_pytorch_model.bin"
    "./stable_diffusion-2-inpainting/vae/diffusion_pytorch_model.safetensors"
    "./stable-diffusion-2-depth/depth_estimator/model.safetensors"
    "./stable-diffusion-2-depth/depth_estimator/pytorch_model.bin"
    "./stable-diffusion-2-depth/text_encoder/pytorch_model.bin"
    "./stable-diffusion-2-depth/unet/diffusion_pytorch_model.bin"
    "./stable-diffusion-2-depth/unet/diffusion_pytorch_model.safetensors"
    "./stable-diffusion-2-depth/vae/diffusion_pytorch_model.bin"
    "./stable-diffusion-2-depth/vae/diffusion_pytorch_model.safetensors"
)

for file in "${files[@]}"
do
    if [ -f "$file" ]; then
        echo "$file downloaded"
    else
        echo "warning: $file is missing"
    fi
done

echo "if some files are missing, please download them manually."