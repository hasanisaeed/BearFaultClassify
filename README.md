# Deep Learning for Bearing Fault Detection
> **Project**: Bearing Fault Detection and Classification Using Temporal Convolutions and LSTM Networks in Induction Machine Systems.

This project applies deep learning to detect and classify faults in bearings of induction machines. The model leverages Temporal Convolutional and LSTM layers to effectively identify various fault types.

---

## Related Publication
For further reading and in-depth research details, refer to our published paper:

**Title**: Classification of Fault Severity in Induction Machine Systems Based on Temporal Convolutions and Recurrent Networks  
**Authors**: V. Mashayekhi, S. Hasani Borzadaran, M. Hoseintabar Marzebali  
**Published**: 16 February 2022  
**DOI**: [https://doi.org/10.1155/2022/4224356](https://doi.org/10.1155/2022/4224356)

---

## Requirements
To reproduce the results, install the dependencies in a Python 3.7 environment.

- **Python**: 3.7
- **Required Packages**: Listed in `requirements.txt`

## Dataset Summary
The dataset consists of bearing fault signals, pre-processed for experiments with optional downsampling. The table below summarizes the data structure:

| Downsampling | Sequence Length | Training Samples | Test Samples | Classes |
| :---: | :---: | :---: | :---: | :---: |
| Yes | 899 | 3095 | 890 | 16 |
| No | 1000 | - | - | 16 |

---

## Model Evaluation
The following table shows the performance of GRU-based models across various configurations, using different downsampling and architecture options to optimize classification accuracy.

| Run | Epochs | Batch Size | Architecture | Weights | Downsampling | Accuracy |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 600 | 128 | [#1](#1-architecture) | 1 | Yes | 71.46% |
| 2 | 1200 | 128 | [#1](#1-architecture) | 2 | Yes | 84.26% |
| 3 | 1200 | 128 | [#2](#2-architecture) | 3 | Yes | 84.94% |
| **4** | 1200 | 128 | [#3](#3-architecture) | 4 | Yes | **87.52%** |
| 5 | 900 | 64 | [#1](#1-architecture) | 5 | Yes | 83.37% |
| _ | _ | _ | _ | _ | _ | _ |
| 6 | 200 | 128 | [#4](#4-architecture) | - | No | 71.35% |
| 7 | 500 | 128 | [#4](#4-architecture) | - | No | 94.65% |
| 8 | 650 | 128 | [#4](#4-architecture) | - | No | 95.37% |
| 9 | 700 | 128 | [#4](#4-architecture) | - | No | 95.48% |
| 10 | 900 | 128 | [#4](#4-architecture) | - | No | 95.62% |
| **11** | 1050 | 128 | [#4](#4-architecture) | - | No | **95.77%** |
| _ | _ | _ | _ | _ | _ | _ |
| 12 | 600 | - | [#5](#5-architecture) | - | No | 91.91% |
| **13** | 850 | - | [#5](#5-architecture) | - | No | **93.70%** |
| 14 | 1000 | - | [#5](#5-architecture) | - | No | 92.43% |
| 15 | 1050 | - | [#5](#5-architecture) | - | No | 92.43% |

---

### #1 Architecture

```python
Conv1D(128) => Conv1D(128) => Conv1D(128) => Conv1D(64)
```

### #2 Architecture

```python
Conv1D(128) => Conv1D(256) => Conv1D(256) => Conv1D(64)
```

### #3 Architecture

```python
Conv1D(64) => Conv1D(128) => Conv1D(256) => Conv1D(64)
```

## Downsampling (No)

### #4 Architecture

```python
Conv1D(128) => Conv1D(128) => Conv1D(128) => Conv1D(64)
```

<p>
    <img src="images/run-01.png" alt="run-01">
    <em>#4 Architecture Results</em>
</p>

### #5 Architecture

```python
Conv1D(64) => Conv1D(64) => Conv1D(64) => Conv1D(128) => Conv1D(64)`
 ```

<p>
    <img src="images/run-02.png" alt="run-02">
    <em>#5 Architecture Results</em>
</p>
