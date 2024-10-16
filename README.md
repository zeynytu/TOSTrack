# TOSTrack: Template-Aided One-Stream Tracker	

The official implementation of The Abstract (The poster will be uploaded after the presentation) <br />
Zeyn, Muhammed and Bayraktar, Ertuğrul, TOSTrack: Template-Aided One-Stream Tracker, 5th Symposium on Pattern Recognition and Applications.
    

## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Install the environment
**Option1**: Use the Anaconda (CUDA 10.2)
```
conda create -n tostrack python=3.8
conda activate tostrack
bash install.sh
```

**Option2**: Use the Anaconda (CUDA 11.3)
```
conda env create -f tostrack_cuda113_env.yaml
```

**Option3**: Use the docker file

We provide the full docker file here.

## Training
Download pre-trained [MAE ViT-Base weights](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and put it under `$PROJECT_ROOT$/pretrained_models` (different pretrained models can also be used, see [MAE](https://github.com/facebookresearch/mae) for more details).

```
python tracking/train.py --script ostrack --config vitb_256_mae_ce_32x4_ep300 --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 1
```

Replace `--config` with the desired model config under `experiments/ostrack`. 


## Evaluation
Download the model weights from [Google Drive](https://drive.google.com/drive/folders/1PHfGDgO6lOB-oTO91MK3o10uQJUUXKZN?usp=sharing) 

Put the downloaded weights on `$PROJECT_ROOT$/output/checkpoints/train/ostrack`

## DEMO

Download the pretrained model and install the required packages from yaml file then run:

```
python tracking/video_demo.py --videofile <YOUR_VIDEO>
```


## Acknowledgement

Our idea is implemented base on the following projects. We really appreciate their excellent open-source works!

- [OSTrack](https://github.com/botaoye/OSTrack) [[related paper](https://arxiv.org/abs/2203.11991)]
- [Mixformer](https://github.com/MCG-NJU/MixFormer) [[related paper](http://arxiv.org/abs/2203.11082)]

This project is not for commercial use. For commercial use, please contact the author.

## Citation

If any parts of our paper and code help your research, please consider citing us and giving a star to our repository.

```
@InProceedings{The 5th Symposium on Pattern Recognition and Applications,
    author    = {Zeyn, Muhammed and Bayraktar, Ertuğrul},
    title     = {TOSTrack: Template-Aided One-Stream Tracker},
    booktitle = {},
    month     = {November},
    year      = {2024},
    pages     = {}
}
