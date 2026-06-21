---
title: "Photo-Realistic Single Image Super-Resolution Using a GAN (SRGAN)"
year: 2017
authors: ["Ledig et al."]
metrics: ["PSNR", "SSIM"]
tags: ["super-resolution", "gan", "image-quality"]
---

# SRGAN — notes

SRGAN introduces a perceptual loss (content loss + adversarial loss) for 4x
single-image super-resolution. The classic reconstruction metrics reported are
**PSNR** (peak signal-to-noise ratio, in dB, higher is better) and **SSIM**
(structural similarity, in [0, 1], higher is better).

Measurement conventions used in our experiments:

- Images are normalized to the range **[0, 1]**, so PSNR is computed with
  `data_range=1.0`.
- SSIM uses an 11-tap Gaussian window (`win_size=11`, `gaussian_weights=True`)
  and `channel_axis=-1` for color images.

A minimal reproduction baseline is bicubic upsampling, which gives a low PSNR/SSIM
floor that any learned model should beat.
