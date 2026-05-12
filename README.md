# SDCA: Towards Semantic-guided Dual Camouflage for Deceiving Human Eyes and Object Detectors

Official implementation of **SDCA**, a semantic-guided adversarial camouflage
generation framework for deceiving both human eyes and object detectors.

## News

- **2026-04-16**: The paper has been published online in *Neural Networks*.
- **2026-04-04**: The paper has been accepted for publication in *Neural Networks*.

## Paper

- Preprint: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5291137
- Final version: https://www.sciencedirect.com/science/article/abs/pii/S0893608026004077
- DOI: https://doi.org/10.1016/j.neunet.2026.108946

## Overview

SDCA leverages semantic features of natural textures, including color
distributions and contour patterns, to optimize adversarial camouflage textures
for both biological vision systems and computer vision systems.

![Overall](https://github.com/Haoq1nYuan/Semantic-guided-Dual-Camouflage-Attack/blob/main/assets/framework.png)

## Key Contributions

### Semantic-Driven Generator

- Uses programmatic noise to model semantic features of natural textures.
- Drives the generation of the initial camouflage texture with extracted scene semantics.
- Provides prior guidance for subsequent adversarial optimization.

### Semantic-Constrained Optimization

- Builds semantic perturbations from the initial texture prior.
- Constrains the optimization space of adversarial perturbations.
- Maintains semantic consistency between initial and adversarial textures.

### Dual-Dimensional Evaluation

- Evaluates attack transferability across detectors and robustness across scenes, viewpoints, and occlusions.
- Evaluates naturalness with SSIM, FSIM, CSI, and camouflage object detection.
- Maintains strong naturalness while preserving transferability and robustness.

## Project Structure

```text
SDCA/
  attack_phase1.py          # Semantic-driven initial texture optimization
  attack_phase2.py          # Semantic-constrained color perturbation optimization
  eval.py                   # Adversarial and naturalness evaluation
  extract.py                # Dominant color extraction from natural scenes
  config/                   # Training and evaluation configs
  3Dmodels/                 # Vehicle mesh, material, and UV masks
  modules/                  # Renderer, generator, evaluator, augmentor, data loader
  detectors/                # YOLOv5 and EfficientDet code; weights are external
  examples/                 # Example camouflage textures
  data/                     # Dataset mount point
  results/                  # Runtime outputs
  docs/                     # Reproduction and model weight instructions
```

## Environment

The reference environment is:

- Python 3.8
- CUDA 11.8
- PyTorch 2.4.1+cu118
- torchvision 0.19.1+cu118
- PyTorch3D 0.7.8
- ultralytics 8.3.37

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate sdca
```

Install PyTorch3D separately if the environment resolver does not provide a
compatible wheel:

```bash
pip install pytorch3d==0.7.8
```

PyTorch3D is sensitive to the Python, PyTorch, CUDA, and OS combination. If the
command above fails, follow the official PyTorch3D installation guide for your
platform.

## Datasets

### Attack Performance Evaluation: CARLA Dataset

- Collected with [CARLA simulator](http://carla.org/)
- Download link: [Google Drive](https://drive.google.com/file/d/1rlBcIWk_PAHJkeBjxjbriTJLWLbvFQCy/view?usp=drive_link)

### Naturalness Evaluation

1. **RSSCN7 Dataset**
   - Uses the forest-class subset only.
   - Official link: https://github.com/palewithout/RSSCN7

2. **Unity Jungle Scene Dataset**
   - Collected with [Unity Jungle Scene Asset](https://naturemanufacture.com/forest-environment-set/)
   - Download link: [Google Drive](https://drive.google.com/file/d/1GShTBQowy_Y9Fy5hCTGI7OWiTNn0rFNl/view?usp=drive_link)

Arrange downloaded data as:

```text
data/
  carla/
    npz/train/data*.npz
    npz/test/data*.npz
    label/train/data*.txt
    label/test/data*.txt
    stage1_idx.npz
    stage2_idx.npz
  rsscn7/
    e001.jpg
    ...
  unity/
    1.png
    ...
```

## Detector Weights

YOLO detector weights are distributed through GitHub Releases instead of Git:

https://github.com/Haoq1nYuan/Semantic-guided-Dual-Camouflage-Attack/releases/tag/detector-weights-v1.0

Download the release assets and place the files as:

```text
detectors/yolov5/yolov5n.pt
detectors/yolov5/yolov5s.pt
detectors/yolov5/yolov5m.pt
detectors/yolov5/yolov5l.pt
detectors/yolov5/yolov5x.pt
detectors/yolov8/yolov8n.pt
detectors/yolov8/yolov8s.pt
detectors/yolov8/yolov8m.pt
detectors/yolov8/yolov8l.pt
```

## Training

Run phase 1:

```bash
python attack_phase1.py --config config/config_train.yaml
```

After phase 1, update `phase2_train.spe_texture` and
`phase2_train.perturbation` in `config/config_train.yaml` to the generated phase
1 outputs, then run:

```bash
python attack_phase2.py --config config/config_train.yaml
```

Training outputs are saved under `results/train_results/`.

## Evaluation

Adversarial evaluation:

```bash
python eval.py --config config/config_eval.yaml --mode adversarial --model yolov5s
```

Available detector names:

```text
yolov5n, yolov5s, yolov5m, yolov5l, yolov5x
yolov8n, yolov8s, yolov8m, yolov8l
EfDetd0, EfDetd1, EfDetd2, FrRCNN, DETR
```

Naturalness evaluation:

```bash
python eval.py --config config/config_eval.yaml --mode natural --save
```

Color extraction:

```bash
python extract.py --image-idx 1 2 3 4 5 --cluster-cnt 10 --rounds 10
```

## Attack Performance

### Digital Evaluation Detected by YOLOv5

https://github.com/user-attachments/assets/d99e885e-12f6-41d0-b47e-e80a58fc4e56

### Physical Evaluation Detected by YOLOv5

https://github.com/user-attachments/assets/06f3db47-207d-4f7c-ba9d-f99ad266f4ab

## Naturalness

### 2D Evaluation Detected by Canny Edge Detection

![2Dn](https://github.com/Haoq1nYuan/Semantic-guided-Dual-Camouflage-Attack/blob/main/assets/2Dn.png)

### 3D Evaluation Detected by PFNet

![3Dn](https://github.com/Haoq1nYuan/Semantic-guided-Dual-Camouflage-Attack/blob/main/assets/3Dn.png)

## Citation

```bibtex
@article{yuan2026sdca,
  title = {SDCA: Towards semantic-guided dual camouflage for deceiving human eyes and object detectors},
  author = {Yuan, Haoqin and Chen, Xianyi and Cui, Qi and Liu, Fazhan and Fu, Zhangjie},
  journal = {Neural Networks},
  volume = {201},
  pages = {108946},
  year = {2026},
  doi = {10.1016/j.neunet.2026.108946}
}
```

## License

This project is released under the MIT License. Third-party detector code,
datasets, pretrained weights, and external assets retain their original licenses.
