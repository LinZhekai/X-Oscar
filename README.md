<p align="center">
  <img width="75%" src="assets/logo and name.png"/>
</p>

# X-Oscar
A pytorch implementation of "X-Oscar: A Progressive Framework for High-quality Text-guided 3D Animatable Avatar Generation"

[![arXiv](https://img.shields.io/badge/arXiv-2405.00954-006600)](https://arxiv.org/abs/2405.00954) 
[![project_page](https://img.shields.io/badge/project_page-68BC71)](https://xmu-xiaoma666.github.io/Projects/X-Oscar/)

## 🎥 Introduction Video
[![intro]](https://github.com/LinZhekai/X-Oscar/blob/main/assets/intro_view.mp4)


## 📖 Abstract
Recent advancements in automatic 3D avatar generation guided by text have made significant progress. However, existing methods have limitations such as oversaturation and low-quality output. To address these challenges, we propose X-Oscar, a progressive framework for generating high-quality animatable avatars from text prompts. It follows a sequential Geometry->Texture->Animation paradigm, simplifying optimization through step-by-step generation. To tackle oversaturation, we introduce Adaptive Variational Parameter (AVP), representing avatars as an adaptive distribution during training. Additionally, we present Avatar-aware Score Distillation Sampling (ASDS), a novel technique that incorporates avatar-aware noise into rendered images for improved generation quality during optimization. Extensive evaluations confirm the superiority of X-Oscar over existing text-to-3D and text-to-avatar approaches. 

https://github.com/LinZhekai/X-Oscar/assets/149573107/a7520940-2429-4f6c-abbe-9b04d4f2d355

## 🛠️ Environment Setup 
- System: Unbuntu 22.04 
- Tested GPU: RTX3090
Tips: It is recommended to follow the version I provided to reproduce 100%.

```bash
git clone git@github.com:LinZhekai/X-Oscar.git
cd X-Oscar

conda env create --file environment.yml
conda activate XOscar
pip install -r requirements.txt
 
cd smplx
python setup.py install 

# download omnidata normal and depth prediction model 
mkdir data/omnidata 
cd data/omnidata 
gdown '1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI&confirm=t' # omnidata_dpt_depth_v2.ckpt
gdown '1wNxVO4vVbDEMEpnAi_jwQObf2MFodcBR&confirm=t' # omnidata_dpt_normal_v2.ckpt
```

## 📚 Data
We follow a similar way to [TADA](https://github.com/TingtingLiao/TADA) to construct data.
- [SMPL-X Model](http://smpl-x.is.tue.mpg.de/) (Download the SMPLX_NEUTRAL_2020.npz and put it into ./data/smplx/)
- [TADA Extra Data](https://download.is.tue.mpg.de/download.php?domain=tada&resume=1&sfile=tada_extra_data.zip) (Unzip it as directory ./data)
- Optional Motion Data  
  - [AIST](https://aistdancedb.ongaaccel.jp/), [AIST++](https://google.github.io/aichoreographer/)
  - [TalkShow](https://github.com/yhw-yhw/TalkSHOW)
  - [MotionDiffusion](https://github.com/GuyTevet/motion-diffusion-model)

<details><summary>Please consider cite <strong>AIST, AIST++, TalkSHOW, MotionDiffusion</strong> if they also help on your project</summary>

```bibtex

@inproceedings{aist-dance-db,
  author = {Shuhei Tsuchida and Satoru Fukayama and Masahiro Hamasaki and Masataka Goto}, 
  title = {AIST Dance Video Database: Multi-genre, Multi-dancer, and Multi-camera Database for Dance Information Processing}, 
  booktitle = {Proceedings of the 20th International Society for Music Information Retrieval Conference (ISMIR) },
  year = {2019}, 
  month = {Nov} 
}

@inproceedings{li2021learn,
  title={AI Choreographer: Music Conditioned 3D Dance Generation with AIST++}, 
  author={Ruilong Li and Shan Yang and David A. Ross and Angjoo Kanazawa},
  year={2021},
  booktitle={ICCV}
}

@inproceedings{yi2023generating,
  title={Generating Holistic 3D Human Motion from Speech},
  author={Yi, Hongwei and Liang, Hualin and Liu, Yifei and Cao, Qiong and Wen, Yandong and Bolkart, Timo and Tao, Dacheng and Black Michael J},
  booktitle={CVPR}, 
  pages={469-480},
  month={June}, 
  year={2023} 
}

@inproceedings{tevet2023human,
  title={Human Motion Diffusion Model},
  author={Guy Tevet and Sigal Raab and Brian Gordon and Yoni Shafir and Daniel Cohen-or and Amit Haim Bermano},
  booktitle={ICLR},
  year={2023},
  url={https://openreview.net/forum?id=SJ1kSyO2jwu}
}


```
</details>

## 🚀 Usage

### Training 

```python
# You can change the save path in the config/*yaml files
python -m apps.run_XOscar --config configs/XOscar.yaml --name XOscar_{subject's name} --text "{text prompt}"

# Following is an example, similar ones can be found in scripts/script.sh
python -m apps.run_XOscar --config configs/XOscar.yaml --name XOscar_Flash --text "Flash from DC"
``` 

### Animation 
- Download [AIST](https://aistdancedb.ongaaccel.jp/) or generate motions from [TalkShow](https://github.com/yhw-yhw/TalkSHOW) and [MotionDiffusion](https://github.com/GuyTevet/motion-diffusion-model). 
```python
# If you have changed the save path in the config files, you also need to change the ckpt_files under this python file
python -m apps.anime --subject "subject's name" --prompt "text prompt" --motion_type "aist, talkshow or mdm"

# Following is an example, change the motion_name in this python file to select different motions in aist
python -m apps.anime --subject "Flash" --prompt "Flash from DC" --motion_type "aist"
``` 

## ❤️ Acknowledgement
Code in this repository is built upon several public repositories. Thanks for the wonderful work [TADA](https://github.com/TingtingLiao/TADA).

## ⭐️ BibTeX
If you find this work useful for your research, please cite:
```
@article{ma2024x,
  title={X-Oscar: A Progressive Framework for High-quality Text-guided 3D Animatable Avatar Generation},
  author={Ma, Yiwei and Lin, Zhekai and Ji, Jiayi and Fan, Yijun and Sun, Xiaoshuai and Ji, Rongrong},
  journal={arXiv preprint arXiv:2405.00954},
  year={2024}
}
```




