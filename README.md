## Exploring Probabilistic Modeling Beyond Domain Generalization for Semantic Segmentation

Official repository for Exploring Probabilistic Modeling Beyond Domain Generalization for Semantic Segmentation

[Project Page](https://pdaf-iccv.github.io/) | [Paper](https://arxiv.org/abs/2507.21367) | [Video](https://www.youtube.com/watch?v=HQlP0R-xvfI) | [Code](https://github.com/pdaf-iccv/PDAF)

## Updates
- June 2025: ✨ PDAF was accepted into ICCV 2025!

## Inference
Please prepare the dataset by referencing [RobustNet](https://github.com/shachoi/RobustNet) and set `dataset_root` in `config.py`. Download checkpoints from [Google drive](https://drive.google.com/file/d/1TF3XLQ8jSxA_uqvtODMR-VJ1S2mNGkPW/view?usp=sharing) and unzip it in folder `checkpoints`.
```
checkpoints/
├── best-dcm-bdd100k.pt
├── best-dpe-bdd100k.pt
├── best-lpe-bdd100k.pt
└── deeplab_res50.yaml
```
Run following command:
```
bash demo.sh
```
## License

This project is based on RobustNet (https://github.com/shachoi/RobustNet), developed by Sungha Choi and licensed under the BSD 3-Clause License.

Modifications have been made by pdaf-iccv in 2025.

This project retains the original BSD 3-Clause License terms.

## Reference
If you find this work useful, please consider citing us!
```
@article{chen2025exploring,
  title={Exploring Probabilistic Modeling Beyond Domain Generalization for Semantic Segmentation},
  author={Chen, I and Chang, Hua-En and Chen, Wei-Ting and Hwang, Jenq-Neng and Kuo, Sy-Yen and others},
  journal={arXiv preprint arXiv:2507.21367},
  year={2025}
}
```