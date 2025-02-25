# Generative Planning with 3D-vision Language Pre-training for End-to-End Autonomous Driving (GPVL-AAAI2025)
## Overview:
Autonomous driving is a challenging task that requires perceiving and understanding the surrounding environment for safe trajectory planning. While existing vision-based end-to-end models have achieved promising results, these methods are still facing the challenges of vision understanding, decision reasoning and scene generalization. To solve these issues, a generative planning with 3D-vision language pre-training model named GPVL is proposed for end-to-end autonomous driving. The proposed paradigm has two significant aspects. On one hand, a 3D-vision language pre-training module is designed to bridge the gap between visual perception and linguistic understanding in the bird's eye view. On the other hand, a cross-modal language model is introduced to generate holistic driving decisions and fine-grained trajectories with perception and navigation information in an auto-regressive manner. Experiments on the challenging nuScenes dataset demonstrate that the proposed scheme achieves excellent performances compared with state-of-the-art methods. Besides, the proposed GPVL presents strong generalization ability and real-time potential when handling high-level commands in various scenarios. It is believed that the effective, robust and efficient performance of GPVL is crucial for the practical application of future autonomous driving systems.
## Method:
The overall pipeline of the proposed GPVL model is illustrated in Fig. 1. First, the backbone includes a 3D-vision encoder to obtain the basic BEV feature, then it is decoded into constrained detection, motion and map features. Second, the 3D-vision language pre-training module establishes the associations between vision and language features with the group-wise alignment. Finally, the cross-modal language model generates the future planning decision in an auto-regressive manner based on aligned visual feature and navigation prompt.
<p align="center">
<image src="figs/fig2.jpg" width="800">
<br/><font>Fig. 1. Overview of the proposed GPVL framework.</font>
</p>

## Results:
The proposed GPVL is compared with several state-of-the-art autonomous driving models on the nuScenes dataset. The experimental results are shown in Table 1, Table 2, Table 3 and Table 4. Then, qualitative experiments are conducted to verify the effectiveness of the proposed GPVL, as illustrated in Fig. 2. 
<p align="center">
<image src="figs/table1.png" width="650">
<br/><font>Table 1. Open-loop planning performance.</font>
</p>
<p align="center">
<image src="figs/table2.png" width="500">
<br/><font>Table 2. Statistical results of L2 distance and collision rate with turn left, turn right and go straight commands.</font>
</p>
<p align="center">
<image src="figs/table3.png" width="500">
<br/><font>Table 3. Ablation study of GPVL on nuScenes.</font>
</p>
<p align="center">
<image src="figs/table4.png" width="400">
<br/><font>Table 4. Zero-shot performance on the new city.</font>
</p>
<p align="center">
<image src="figs/fig3.jpg" width="550">
<br/><font>Fig. 2. Visualized comparison of the proposed GPVL, VAD and the ground-truth on the nuScenes dataset.</font>
</p>

## Usage:

### Preparations
- Python 3.8
- pytorch=1.9.1, cuda=11.1, torchvision=0.10.1, mmcv=0.14.0, torchaudio=0.9.1, mmdet=2.14.0, mmsegmentation=0.14.1, apex=0.1, nuscenes-devkit=1.1.9

### Data preparation
#### make the prompts of detection, motion, map and global labels
download the map infos from [google cloud](https://drive.google.com/file/d/1Vb46hXNVAGTXn6-x2f9DNsx99LgMtwc1/view?usp=drive_link) and then
python ./tools/det_motion_map_labels.py
#### extract the detection, motion and map features
bash visual/extract.sh
#### pretrained bert weights
download the weights from [google cloud](https://drive.google.com/file/d/1KyqOzQIzNcL1Q9uEGmDECHfU-8CCd4kk/view)
#### det-motion-map labels [optional]
directly download the preprocessed labels from [google_cloud](https://drive.google.com/file/d/1E4X20n9ffAY5JhsB7mTxDDcm9OWUkN9L/view?usp=drive_link) and put it into datasets folder

### 3D-vision-language training
bash scripts/pretrain.sh

### Trajectory finetuning
bash scripts/finetune_cap.sh

### Evaluation
python visual/test_by_pred_results.py

## Citation
If you find GPVL useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.
```
@article{li2025generative,
  title={Generative Planning with 3D-vision Language Pre-training for End-to-End Autonomous Driving},
  author={Li, Tengpeng and Wang, Hanli and Li, Xianfei and Liao, Wenlong and He, Tao and Peng, Pai},
  journal={arXiv preprint arXiv:2501.08861},
  year={2025}
}
```
