dir="checkpoints"
if [ ! -d "$dir" ];then
mkdir $dir
fi

wget -P $dir https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swinl_only_sam_many2many.pth
wget -P $dir https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v1.pt
wget -P $dir https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth