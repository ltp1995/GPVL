CUDA_VISIBLE_DEVICES=0,1,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 1322 ./pretrain.py \
--config ./config/pretrain-nuscene-captv.json \
--video_encoder_type 'clip_vit_base_16' \
--txt_encoder_type 'bert_base_uncased' \
--output_dir ./output/nuscene_pretrain_base_captv_v2 \
--contra_loss_ratio 1.5
######
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 1322 ./pretrain.py \
# --config ./config/pretrain-nuscene-captv-tiny.json \
# --video_encoder_type 'clip_vit_base_16' \
# --txt_encoder_type 'bert_base_uncased' \
# --output_dir ./output/nuscene_pretrain_tiny_captv \
# --contra_loss_ratio 1.5







