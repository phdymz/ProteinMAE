# ProteinMAE
Official PyTorch implementation of  "ProteinMAE: Masked Autoencoder for Protein Surface Self-supervised Learning". 




### Dataset 

We use Baidu Cloud Disk to share the datasets we use:: https://pan.baidu.com/s/1lkq4g5TlRz3tja9_LsQGfQ?pwd=data Password: data 


### Pre-Training

```shell
```

### Train
Ligand-binding pocket classification (init with pre-trained weight):
```shell
python train_ligand.py --ckpt ./checkpoints/ckpt-last.pth
```


### Inference
Ligand-binding pocket classification(scratch):
```shell
python test_ligand.py --checkpoint ./checkpoints/Transformer_ligand_downsample512_group768size16_new_epoch395.pth
```

Ligand-binding pocket classification:
```shell
python test_ligand.py --checkpoint ./checkpoints/Transformer_ligand_pre_downsample512_group768size16_new_epoch295.pth
```




### Citation
If you find this code useful for your work or use it in your project, please consider citing:

```shell
@article{yuan2023proteinmae,
  title={ProteinMAE: masked autoencoder for protein surface self-supervised learning},
  author={Yuan, Mingzhi and Shen, Ao and Fu, Kexue and Guan, Jiaming and Ma, Yingfan and Qiao, Qin and Wang, Manning},
  journal={Bioinformatics},
  volume={39},
  number={12},
  pages={btad724},
  year={2023},
  publisher={Oxford University Press}
}
```


### Acknowledgments
In this project we use (parts of) the official implementations of the followin works:
