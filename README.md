# [CVPR2025 Highlight] TKG-DM🥚🍚: Training-free Chroma Key Content Generation Diffusion Model
[![arXiv](https://img.shields.io/badge/arXiv-2411.15580-b31b1b.svg)](https://arxiv.org/abs/2411.15580)

> **TKG-DM: Training-free Chroma Key Content Generation Diffusion Model**  
> *Ryugo Morita, Stanislav Frolov, Brian Bernhard Moser, Takahiro Shirakawa, Ko Watanabe, Andreas Dengel, Jinjia Zhou*  

![TKG-DM Pipeline](static/images/model.svg)

## Abstract

Diffusion models have enabled the generation of high-quality images with a strong focus on realism and textual fidelity. Yet, large-scale text-to-image models, such as Stable Diffusion, struggle to generate images where foreground objects are placed over a chroma key background, limiting their ability to separate foreground and background elements without fine-tuning. To address this limitation, we present a novel Training-Free Chroma Key Content Generation Diffusion Model (TKG-DM), which optimizes the initial random noise to produce images with foreground objects on a specifiable color background. Our proposed method is the first to explore the manipulation of the color aspects in initial noise for controlled background generation, enabling precise separation of foreground and background without fine-tuning. Extensive experiments demonstrate that our training-free method outperforms existing methods in both qualitative and quantitative evaluations, matching or surpassing fine-tuned models. Finally, we successfully extend it to other tasks (e.g., consistency models and text-to-video), highlighting its transformative potential across various generative applications where independent control of foreground and background is crucial.


## Usage

To run the program from the terminal, use the following command:

```bash
python main.py --method tkg --device 0 --seed 1234
```

## Options

- **`--method`**: Choose the image generation technique:
  - `"gbp"` – Greenback Prompt method.
  - `"tkg"` – TKG noise processing method.
- **`--device`**: Specify the index of the CUDA GPU to be used (e.g., `0`).
- **`--seed`**: Set a seed for random number generation to ensure reproducibility.


## Citation
If our work has supported your research, we invite you to give our repository a star ⭐ or cite our work using the following BibTeX entry:
```bibtex
@article{morita2024tkg,
  title={TKG-DM: Training-free Chroma Key Content Generation Diffusion Model},
  author={Morita, Ryugo and Frolov, Stanislav and Moser, Brian Bernhard and Shirakawa, Takahiro and Watanabe, Ko and Dengel, Andreas and Zhou, Jinjia},
  journal={arXiv preprint arXiv:2411.15580},
  year={2024}
}
```