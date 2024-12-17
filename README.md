## [CEDNet: A Cascade Encoder-Decoder Network for Dense Prediction](https://arxiv.org/abs/2302.06052)


## Installation
Please check [INSTALL.md](INSTALL.md) for installation instructions.

## Evaluation
We give an example evaluation command for a ImageNet-1K pre-trained:

```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model xxx --eval true --input_size 224 --data_path /path/to/imagenet-1k
```

## Citation
If you find this repository helpful, please consider citing:
```
@article{ZHANG2025111072,
    title = {CEDNet: A cascade encoder–decoder network for dense prediction},
    journal = {Pattern Recognition},
    volume = {158},
    pages = {111072},
    year = {2025},
    issn = {0031-3203},
    author = {Gang Zhang and Ziyi Li and Chufeng Tang and Jianmin Li and Xiaolin Hu},
}
```