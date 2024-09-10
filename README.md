# Description
This repository is a PyTorch + PyTorch Lightning implementation of [Crop mapping from image time series: deep learning with multi-scale label hierarchies](https://arxiv.org/pdf/2102.08820). The initial repository unfortunately contains inaccuracies that have been copied many times.

Thank you [Mehmet Ozgur Turkoglu](https://github.com/0zgur0), one of the authors of the paper, for guidance and support.
This current implementation is used in a production solution of an AgriTech company I'm helping with in R&D.

# How to
## Sources:
* https://github.com/0zgur0/multi-stage-convSTAR-network/tree/master?tab=readme-ov-file
* https://github.com/0zgur0/STAckable-Recurrent-network

## Setup
Create a virtual environment from the `requirements.txt` file.

## Starting
A simpler approach with PyTorch Lightning and some StarCell adjustments. We don't split explicitly into folds but this can be done. Check the [original repository](https://github.com/0zgur0/multi-stage-convSTAR-network/tree/master?tab=readme-ov-file) for such info.

<span style="color: red;">NB:</span> The original repository implements a GRU cell instead of a ConvStar cell. Check the [old](https://github.com/0zgur0/STAckable-Recurrent-network) TensorFlow code for comparison.

---
- Download the ZueriCrop Dataset:
    1) Download the dataset via https://polybox.ethz.ch/index.php/s/uXfdr2AcXE3QNB6
    2) Create a `storage` folder here and rename the downloaded file to `data.h5`
    3) Adjust `config.py` per your structure.
- Run the `train.py`
