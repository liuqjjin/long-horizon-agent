---
title: "Photo-Realistic Single Image Super-Resolution Using a GAN (SRGAN)"
year: 2017
authors: ["Ledig et al."]
metrics: ["PSNR", "SSIM"]
tags: ["super-resolution", "gan", "image-quality"]
record_kind: "indexing_fixture"
source_url: "https://openaccess.thecvf.com/content_cvpr_2017/html/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.html"
---

# SRGAN 论文摘记（索引夹具）

本文提出用内容损失和对抗损失组成的感知损失处理 4 倍单图像超分辨率。论文同时
讨论 PSNR、SSIM 与感知质量，但不能只根据一个指标判断视觉效果。

这个文件供 `lha eval` 测试论文检索、引用和实验参数提取，不是 SRGAN 复现报告。
仓库示例脚本采用以下本地校验口径：

- 图像值域为 **[0, 1]**，PSNR 使用 `data_range=1.0`；
- 彩色图像的 SSIM 使用 `data_range=1.0` 和 `channel_axis=-1`，
  其余参数沿用当前 scikit-image 的默认值。

这些参数只描述仓库夹具，不能替代原论文的完整数据集、裁剪和颜色通道设置。双三次
插值在示例中只是一个对照方法，本文件不预设其他方法必须超过它。
