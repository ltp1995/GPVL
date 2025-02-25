basedir=output/nuscene_pretrain_base_captv
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2  --master_port 888 ./train.py \
--pretrain_dir $basedir \
--config ./config/caption-nuscene.json \
--output_dir $basedir'/nuscene-caption-5e-6-with-prevtrajs-captv'   \
--learning_rate 5e-6  \
--save_best true \
--warmup_ratio 0.05 \
--train_video_sample_num 6 \
--test_video_sample_num 6  \


# basedir=output/nuscene_pretrain_tiny_captv
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2  --master_port 888 ./train.py \
# --pretrain_dir $basedir \
# --config ./config/caption-nuscene-tiny.json \
# --output_dir $basedir'/nuscene-caption-5e-6-with-prevtrajs-captv-tiny'   \
# --learning_rate 5e-6  \
# --save_best true \
# --warmup_ratio 0.05 \
# --train_video_sample_num 6 \
# --test_video_sample_num 6  \