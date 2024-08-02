# How to
## Sources:
* https://github.com/0zgur0/multi-stage-convSTAR-network/tree/master?tab=readme-ov-file
* https://github.com/0zgur0/STAckable-Recurrent-network

## Setup
Create a virtual env from the `requirements.txt` file.

## Starting
A simpler approach with pytorch lightning and some StarCell adjustments. We don't split explicitly into folds but that can be done. Check the [original repository](https://github.com/0zgur0/multi-stage-convSTAR-network/tree/master?tab=readme-ov-file) for such info.

<span style="color: red;">NB</span> The original repository is implementing a GRU cell instead of a ConvStar cell. Check the [old](https://github.com/0zgur0/STAckable-Recurrent-network) tensorflow code for correctness

---
- Download the ZueriCrop Dataset from
    1) Download the dataset via https://polybox.ethz.ch/index.php/s/uXfdr2AcXE3QNB6
    2) Create a `storage` folder here and rename the downloaded file to `data.h5`
    3) adjust `config.py` per your structure.
- Run the `train.py`