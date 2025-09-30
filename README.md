# A Fine-Grained Framework for Online IoT Device Firmware Version Identification via Version Evolution Analysis

The rapid expansion of IoT networks has outpaced the capabilities of firmware management protocols, leaving numerous Internet-connected devices operating on outdated firmware that contains exploitable vulnerabilities. As vulnerabilities are closely tied to specific firmware versions, fine-grained version identification is critical for effective device management and security risk assessment. However, high firmware heterogeneity and subjective biases in feature selection pose significant challenges to online firmware version identification (OFVI) of IoT devices. To address these challenges, we first construct a dataset comprising 444,195 embedded web pages extracted from 1,000 successfully simulated firmware images. Through analyzing update patterns of embedded web interfaces during firmware version evolution, we propose FirmID, a novel OFVI framework for IoT devices that utilizes directory and content changes in embedded web interfaces. To handle the heterogeneity of firmware across different vendors, we introduce the Hierarchical Multimodal Attention Network (HMANet), a machine learning model specifically designed to capture differences across structural, textual, and functional modalities. To overcome the challenge of distinguishing hard samples caused by the frequent reuse of web pages in firmware iteration versions, we design a Hard Negative Mining Contrastive Loss that enhances intra-class compactness and inter-class separability. Moreover, to improve identification efficiency under uncertain network conditions, FirmID incorporates a complementary heuristic search algorithm, Firmware Identification with Monte Carlo Tree Search (FIMCTS). Experimental results demonstrate that FirmID surpasses state-of-the-art methods by 30.2% in accuracy and reduces file requests by 23.3% in recognition efficiency.

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

The dataset is located in the [`dataset_demo`](https://drive.google.com/file/d/15CTGfaYVG4PxpNUfj1aB9hoSWeRJjKQA/view?usp=sharing) and contains embedded web pages from online IoT devices, covering 1,000 firmware images from eight manufacturers: D-Link, ASUS, TP-Link, Belkin, Linksys, NETGEAR, TRENDnet, and Zyxel. The complete dataset will be refined and uploaded to the same repository soon.

### Hierarchical Multimodal Attention Network

The `HMANet.py` script implements the architecture of a Hierarchical Multimodal Attention Network, which jointly models deep representations from Text, DOM tree , and Code modalities. The framework leverages a Cross-Modal Fusion Layer, a Dynamic Multi-Head Attention mechanism, and a Hard Negative Mining Contrastive Loss to effectively perform firmware version identification.

```bash
python HMANet.py
```
The trained model , checkpoint files, and evaluation metrics can be gained after running this script.

### Firmware Identification with Monte Carlo Tree Search

The `FIMCTS-vs-ARGUS.py` script implements and compares two algorithms, ARGUS and FIMCTS, designed to improve firmware version identification efficiency through multi-round simulations.
Before running the script, ensure you obtain the test dataset: [`FIMCTS-test`](https://drive.google.com/file/d/1VB44gPlFC7gV4Nq8R_n6Anxubuz8azTs/view?usp=sharing)

```bash
python FIMCTS-vs-ARGUS.py
```
After running the script, evaluation metrics such as accuracy, efficiency, and consistency for both algorithms can be obtained, along with a quantified measure of efficiency improvement.

## Notes

- Ensure consistent `device` settings across all scripts to avoid tensor mismatch issues between GPU/CPU.
- Data file paths should be correctly set in the code.
- When using a GPU for training and testing, ensure CUDA is available and properly configured.
