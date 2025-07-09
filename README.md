# OA-Stereo: Self-Supervised Opti-Acoustic Stereo for Robust 3D Perception of Underwater Vehicles

## Software Architecture
* scripts: Scripts for running various training/visualization tasks.
* src: Core source code, including network architecture, datasets, loss functions, and general utility code.

## Environment Setup

Tested Environment: PyTorch 2.3.0, CUDA 12.1

Installation Process:

```bash
conda create -n oastereo python=3.10
conda activate oastereo
conda install pytorch==2.3.0 torchvision==0.18.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Datasets

### Dave Sonar Datasets:

A custom dataset collected based on the [DAVE project](https://github.com/Field-Robotics-Lab/DAVE), used for both training and model validation.  
**Download via Hugging Face**: [Anony-Mous-123/dave_sonar](https://huggingface.co/datasets/Anony-Mous-123/dave_sonar)

### Dataset Directory Structure:

```bash
├── datasets
    ├── dave_sonar
        ├── scene1
        │   ├── cam_depth.yaml
        │   ├── cam_left
        │   ├── cam_left.yaml
        │   ├── cam_right
        │   ├── cam_right.yaml
        │   ├── depthl
        │   ├── displ
        │   ├── sonar
        │   ├── sonar_view
        │   ├── sonar_view_origin
        │   └── sonar.yaml
        ├── scene2
        │   ├── cam_depth.yaml
        │   ├── cam_left
        │   ├── cam_left.yaml
        │   ├── cam_right
        │   ├── cam_right.yaml
        │   ├── depthl
        │   ├── displ
        │   ├── sonar
        │   ├── sonar_view
        │   ├── sonar_view_origin
        │   └── sonar.yaml
        ├── scene3
        │   ├── cam_depthl.yaml
        │   ├── cam_depthr.yaml
        │   ├── cam_depth.yaml
        │   ├── cam_left
        │   ├── cam_left.yaml
        │   ├── cam_right
        │   ├── cam_right.yaml
        │   ├── depthl
        │   ├── depthr
        │   ├── displ
        │   ├── sonar
        │   ├── sonar_view
        │   ├── sonar_view_origin
        │   └── sonar.yaml
        └── scene4
            ├── cam_depthl.yaml
            ├── cam_depthr.yaml
            ├── cam_depth.yaml
            ├── cam_left
            ├── cam_left.yaml
            ├── cam_right
            ├── cam_right.yaml
            ├── depthl
            ├── depthr
            ├── displ
            ├── sonar
            ├── sonar_view
            ├── sonar_view_origin
            └── sonar.yaml
```

## **Visualization, Validation, and Training**

- **Visualization**: Use the script `scripts/show.sh` to visualize stereo matching results.
- **Validation**: Use the script `scripts/valid.sh` to evaluate the model on validation datasets.
- **Training**: Use the script `scripts/train.sh` to train the model from scratch.


## Acknowledgments

Part of the code is adopted from previous works: [CoEx](https://github.com/antabangun/coex), [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo), [UWStereo](https://github.com/kskin/UWStereo), [Monodepth2](https://github.com/nianticlabs/monodepth2)