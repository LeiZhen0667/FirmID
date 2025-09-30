# A Fine-Grained Framework for Online IoT Device Firmware Version Identification via Version Evolution Analysis

We have currently released a demo dataset and demo scripts. Upon acceptance of the paper, we will release the full dataset and all related scripts.

## Installation

1. Clone the repository:

    ```bash
    Download all files
    cd FirmID
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have a version of PyTorch with CUDA support installed if using a GPU.

## Usage Instructions

### Dataset

The dataset is located in the `dataset_demo` folder and contains embedded web pages from online IoT devices, covering 1,000 firmware images from eight manufacturers: D-Link, ASUS, TP-Link, Belkin, Linksys, NETGEAR, TRENDnet, and Zyxel.

### Hierarchical Multimodal Attention Network

The `HMANet.py` script implements the architecture of a Hierarchical Multimodal Attention Network, which jointly models deep representations from Text, DOM tree , and Code modalities. The framework leverages a Cross-Modal Fusion Layer, a Dynamic Multi-Head Attention mechanism, and a Hard Negative Mining Contrastive Loss to effectively perform firmware version identification.

```bash
python HMANet.py
```
The trained model , checkpoint files, and evaluation metrics can be gained after running this script.

### Firmware Identification with Monte Carlo Tree Search

The `FIMCTS-vs-ARGUS.py` script implements and compares two algorithms, ARGUS and FIMCTS, designed to improve firmware version identification efficiency through multi-round simulations.

```bash
python FIMCTS-vs-ARGUS.py
```
After running the script, evaluation metrics such as accuracy, efficiency, and consistency for both algorithms can be obtained, along with a quantified measure of efficiency improvement.

## Notes

- Ensure consistent `device` settings across all scripts to avoid tensor mismatch issues between GPU/CPU.
- Data file paths should be correctly set in the code.
- When using a GPU for training and testing, ensure CUDA is available and properly configured.
