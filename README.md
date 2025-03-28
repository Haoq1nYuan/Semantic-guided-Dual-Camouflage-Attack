# SDCA: Towards Semantic-guided Dual Camouflage for Deceiving Human Eyes and Object Detectors
# 语义引导的双重伪装攻击框架

## Overview
This repository presents the open-source implementation of SDCA, a novel adversarial camouflage generation framework that leverages **semantic features of natural textures (e.g., color distributions and contour patterns)** to optimize adversarial textures for evading both biologic vision systems and computer vision systems.

Source code will be released soon.

## Key Contribution

### Semantic-Driven Generator (SDG)
- Utilizing programmatic noise to achieve inverse modeling of semantic features.
- Leveraging semantic features to drive the generation of the initial texture.
- Providing a priori guidance for subsequent optimization tasks.

### Semantic-Constrained Optimization (SCO)
- Forming a unique semantic perturbation based on *a priori* semantics from the initial texture.
- Constraining the optimization space of the perturbation actively.
- Maintaining semantic consistency between adversarial textures and initial textures.

### Dual-Dimensional Evaluation
- Achieves state-of-the-art naturalness while maintaining robustness and transferability.
- Attack success rate (ASR): **99.7%** (against YOLOv5x)
- SSIM: 0.7589, FSIM: **0.9348**, CSI: 0.7183 (similarity to natural textures)
- Camouflage success rate (CSR): **96.6%** (against SINet-V2)

## Dataset
### For attack performance evaluation: **[CARLA Dataset]**
- Collected via [[CARLA simulator](http://carla.org/)]
- Download link: [[Google Drive]()]

### For naturalness evaluation
1. **[RSSCN7 Dataset]** 
- Collected the forest-class subset only.
- Official link: [[URL](https://github.com/palewithout/RSSCN7)]

2. **[Unity Jungle Scene Dataset]** 
- Collected via [[Unity Jungle Scene Asset](https://naturemanufacture.com/forest-environment-set/)]
- Download link: [[Google Drive]()]

## Framework
![Overall](https://github.com/Haoq1nYuan/Semantic-guided-Dual-Camouflage-Attack/blob/main/assets/overall.png)

### Domaint Color Extracting
![Extracting](https://github.com/Haoq1nYuan/Semantic-guided-Dual-Camouflage-Attack/blob/main/assets/extracting.png)

### Semantic-Driven Generator
![SDG](https://github.com/Haoq1nYuan/Semantic-guided-Dual-Camouflage-Attack/blob/main/assets/SDG.png)

### Differentiable Scene Rendering
![Rendering](https://github.com/Haoq1nYuan/Semantic-guided-Dual-Camouflage-Attack/blob/main/assets/rendering.png)

## Attack performance
