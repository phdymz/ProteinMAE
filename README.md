# ProteinMAE
Official PyTorch implementation of  "ProteinMAE: Masked Autoencoder for Protein Surface Self-supervised Learning". 




### Dataset 

We use Baidu Cloud Disk to share the datasets we use:: https://pan.baidu.com/s/1lkq4g5TlRz3tja9_LsQGfQ?pwd=data Password: data 


### Pre-Training

```shell
```

### Downstream tasks

#### Train
Binding site identification (init with pre-trained weight):
```shell
python train_site.py --ckpt ./checkpoints/ckpt-last.pth
```

Protein-protein interaction prediction (init with pre-trained weight):
```shell
python train_search.py --ckpt ./checkpoints/ckpt-last.pth
```


Ligand-binding pocket classification (init with pre-trained weight):
```shell
python train_ligand.py --ckpt ./checkpoints/ckpt-last.pth
```


#### Inference
Binding site identification (scratch):
```shell
python test_site.py --checkpoint ./checkpoint/Transformer_site_batch32_yuanshi_epoch107
```

Binding site identification:
```shell
python test_site.py --checkpoint ./checkpoint/Transformer_site_batch32_yuanshi_pre6.11_epoch27.pth
```


Protein-protein interaction prediction (scratch):
```shell
python test_search.py --checkpoint ./checkpoint/Transformer_search_batch32_group512_size16_downsample512_6.15_epoch493.pth
```

Protein-protein interaction prediction:
```shell
python test_search.py --checkpoint ./checkpoint/Transformer_search_batch32_pre_group512_size16_downsample512_6.16_epoch382.pth
```


Ligand-binding pocket classification (scratch):
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
- [dMaSIF](https://github.com/FreyrS/dMaSIF) 
- [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) 
- [MaSIF](https://github.com/LPDI-EPFL/masif) 

 We thank the respective authors for open sourcing their methods. 