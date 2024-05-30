# Animate Anyone

## Overview

This repository currently provides the unofficial pre-trained weights and inference code of [Animate Anyone](https://humanaigc.github.io/animate-anyone). It is inspired by the implementation of the [MooreThreads/Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone) repository and we made some adjustments to the training process and datasets.

## Samples

<table class="center">
    <tr><td><video controls autoplay loop src="https://github.com/novitalabs/AnimateAnyone/assets/4327933/01e5e5b5-0735-4d6e-8efb-d4859e9d29d3">Demo 1</video></td></tr>
    <tr><td><video controls autoplay loop src="https://github.com/novitalabs/AnimateAnyone/assets/4327933/9f535975-1aff-4ed9-a584-64e40c7b3f80">Demo 2</video></td></tr>
    <tr><td><video controls autoplay loop src="https://github.com/novitalabs/AnimateAnyone/assets/4327933/158baada-d092-4a7b-9c4a-3263d983aace">Demo 3</video></td></tr>
    <tr><td><video controls autoplay loop src="https://github.com/novitalabs/AnimateAnyone/assets/4327933/e0ba3d49-babf-45dd-be63-e637e2c50bcb">Demo 4</video></td></tr>
</table>

## Quickstart

### Build Environtment

We Recommend a python version `>=3.10` and cuda version `=11.7`. Then build environment as follows:

```shell
# [Optional] Create a virtual env
python -m venv .venv
source .venv/bin/activate
# Install with pip:
pip install -r requirements.txt
```

### Download weights

**Automatically downloading**: You can run the following command to download weights automatically:

```shell
python tools/download_weights.py
```

Weights will be placed under the `./pretrained_weights` direcotry. The whole downloading process may take a long time.

### Inference

Here is the cli command for running inference scripts:

```shell
python -m scripts.pose2vid --config ./configs/prompts/animation.yaml -W 512 -H 784 -L 64
```

You can refer the format of `animation.yaml` to add your own reference images or pose videos. To convert the raw video into a pose video (keypoint sequence), you can run with the following command:

```shell
python tools/vid2pose.py --video_path /path/to/your/video.mp4
```

## Or try it on Novita AI

We've deployed this model on Novita AI, and you can try it out with Playground ➡️ https://novita.ai/model/playground#animate-anyone .

## Acknowledgements

This project is based on [MooreThreads/Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone) which is licensed under the Apache License 2.0. We thank to the authors of [Animate Anyone](https://humanaigc.github.io/animate-anyone) and [MooreThreads/Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone), for their open research and exploration.
