# SDCA: Towards Semantic-guided Dual Camouflage for Deceiving Human Eyes and Object Detectors

## Overview
This repository presents the open-source implementation of SDCA, a novel adversarial camouflage generation framework that leverages **semantic features of natural textures (e.g., color distributions and contour patterns)** to optimize adversarial textures for evading both biological vision systems and computer vision systems.

![Overall](https://github.com/Haoq1nYuan/Semantic-guided-Dual-Camouflage-Attack/blob/main/assets/framework.png)

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
- Evaluating attack performance: transferability across models/scenes and robustness under different viewpoints/occlusion conditions.
- Evaluating texture naturalness via similarity metrics (SSIM, FSIM, CSI) and the camouflage object detection (COD) task.
- Achieving state-of-the-art naturalness while maintaining transferability and robustness.

## Dataset
### For attack performance evaluation: **[CARLA Dataset]**
- Collected via [[CARLA simulator](http://carla.org/)]
- Download link: [[Google Drive](https://drive.google.com/file/d/1rlBcIWk_PAHJkeBjxjbriTJLWLbvFQCy/view?usp=drive_link)]

### For naturalness evaluation
1. **[RSSCN7 Dataset]** 
- Collected the forest-class subset only.
- Official link: [[URL](https://github.com/palewithout/RSSCN7)]

2. **[Unity Jungle Scene Dataset]** 
- Collected via [[Unity Jungle Scene Asset](https://naturemanufacture.com/forest-environment-set/)]
- Download link: [[Google Drive](https://drive.google.com/file/d/1GShTBQowy_Y9Fy5hCTGI7OWiTNn0rFNl/view?usp=drive_link)]

## Attack performance

### Digital evaluation (detected by YOLOv5)
https://github.com/user-attachments/assets/d99e885e-12f6-41d0-b47e-e80a58fc4e56

### Physical evaluation (detected by YOLOv5)
https://github.com/user-attachments/assets/06f3db47-207d-4f7c-ba9d-f99ad266f4ab

## Naturalness

### 2D evaluation (detected by the Canny edge detection algorithm)
![2Dn](https://github.com/Haoq1nYuan/Semantic-guided-Dual-Camouflage-Attack/blob/main/assets/2Dn.png)

### 3D evaluation (detected by [PFNet](https://github.com/Mhaiyang/CVPR2021_PFNet?tab=readme-ov-file))
![3Dn](https://github.com/Haoq1nYuan/Semantic-guided-Dual-Camouflage-Attack/blob/main/assets/3Dn.png)