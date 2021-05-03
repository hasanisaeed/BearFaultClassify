## Deep Learning
>Bearing Fault Detection and Classification Based on Temporal Convolutions and LSTM Network in Induction Machine Systems.

## GRU Results
|Downsampling|Sequence Length|Train (Samples)|Test (Samples)|Classes|
|---|-------------|-------------|------|---------|
|Yes|899|3095 |890|16|

|Run |Epoch|Batch Size|Architecture|Weights|Downsampling |Accuracy|
|---|---|---|---|---|---|---|
|1|600|128|[#1](#1-architecture)|1|Yes|71.46%|
|2|1200|128|[#1](#1-architecture)|2|Yes|84.26%|
|3|1200|128|[#2](#2-architecture)|3|Yes|84.94%|
|4|1200|128|[#3](#3-architecture)|4|Yes|87.52%|
|5|900|64|[#1](#1-architecture)|5|Yes|83.37%|

### Path of the weight matrix
1) ~~./weights/grufcn_64_cells_weights/run_12_gru_with_softmax_weights.h5~~
2) ~~./weights/grufcn_64_cells_weights/run_12_gru_with_softmax_weights.h5~~
3) ~~./weights/grufcn2_64_cells_weights/run_12_gru_with_softmax_weights.h5~~
4) ~~./weights/grufcn3_64_cells_weights/run_12_gru_with_softmax_weights.h5~~
5) ~~./weights/grufcn_64_cells_weights/run_12_gru_with_softmax_weights.h5~~

### #1 Architecture
`Conv1D(128) => Conv1D(128) => Conv1D(128) => Conv1D(64)`

### #2 Architecture
`Conv1D(128) => Conv1D(256) => Conv1D(256) => Conv1D(64)`

### #3 Architecture
`Conv1D(64) => Conv1D(128) => Conv1D(256) => Conv1D(64)`