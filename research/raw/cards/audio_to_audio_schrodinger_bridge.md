---
license: other
license_name: nvidia-oneway-noncommercial-license
---

# PyTorch Implementation of Audio-to-Audio Schrodinger Bridges

**Zhifeng Kong, Kevin J Shih, Weili Nie, Arash Vahdat, Sang-gil Lee, Joao Felipe Santos, Ante Jukic, Rafael Valle, Bryan Catanzaro**

[[paper]](https://arxiv.org/abs/2501.11311) [[GitHub]](https://github.com/NVIDIA/diffusion-audio-restoration) [[Demo]](https://research.nvidia.com/labs/adlr/A2SB/)

This repo contains the PyTorch implementation of [A2SB: Audio-to-Audio Schrodinger Bridges](https://arxiv.org/abs/2501.11311). A2SB is an audio restoration model tailored for high-res music at 44.1kHz. It is capable of both bandwidth extension (predicting high-frequency components) and inpainting (re-generating missing segments). Critically, A2SB is end-to-end without need of a vocoder to predict waveform outputs, and able to restore hour-long audio inputs. A2SB is capable of achieving state-of-the-art bandwidth extension and inpainting quality on several out-of-distribution music test sets.

- We propose A2SB, a state-of-the-art, end-to-end, vocoder-free, and multi-task diffusion Schrodinger Bridge model for 44.1kHz high-res music restoration, using an effective factorized audio representation.

- A2SB is the first long audio restoration model that could restore hour-long audio without
boundary artifacts

## License

The model is provided under the NVIDIA OneWay NonCommercial License. 


## Citation

```
@article{kong2025a2sb,
  title={A2SB: Audio-to-Audio Schrodinger Bridges},
  author={Kong, Zhifeng and Shih, Kevin J and Nie, Weili and Vahdat, Arash and Lee, Sang-gil and Santos, Joao Felipe and Jukic, Ante and Valle, Rafael and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:2501.11311},
  year={2025}
}
```