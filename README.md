# TOSTrack: Template-Aided One-Stream Tracker	


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

## Training
Download pre-trained [MAE ViT-Base weights](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and put it under `$PROJECT_ROOT$/pretrained_models` (different pretrained models can also be used, see [MAE](https://github.com/facebookresearch/mae) for more details).

```
python tracking/train.py --script ostrack --config vitb_256_mae_ce_32x4_ep300 --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 1
```

Replace `--config` with the desired model config under `experiments/ostrack`. 


## Evaluation
Download the model weights from [Google Drive](https://drive.google.com/drive/folders/1PHfGDgO6lOB-oTO91MK3o10uQJUUXKZN?usp=sharing) 

Put the downloaded weights on `$PROJECT_ROOT$/output/checkpoints/train/ostrack`


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
