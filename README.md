Overview
This solution implements a RoBERTa-based model for predicting text readability scores. The approach uses cross-validation, attention mechanisms, and a carefully designed training process to create a robust model for readability assessment.
Model Architecture
Base Model

Used RoBERTa-base as the foundation model
Added custom attention and regression layers
Modified the model configuration to:

Enable output of hidden states
Set hidden dropout probability to 0.0
Set layer normalization epsilon to 1e-7



Custom Layers
The model includes two additional components on top of RoBERTa:

Attention Layer:

pythonCopyself.attention = nn.Sequential(
    nn.Linear(768, 512),
    nn.Tanh(),
    nn.Linear(512, 1),
    nn.Softmax(dim=1)
)

Regression Layer:

pythonCopyself.regressor = nn.Sequential(
    nn.Linear(768, 1)
)
Training Process
Data Preparation

Maximum sequence length: 248 tokens
Batch size: 16
Used 5-fold cross-validation
Removed incomplete entries (where target and standard_error are both 0)

Optimization Strategy
The training process uses a sophisticated layer-wise learning rate approach:

Different learning rates for different layers of RoBERTa:

Base layers: 2e-5
Middle layers (69+): 5e-5
Upper layers (133+): 1e-4


Adam optimizer with custom weight decay
Cosine scheduler with warmup steps

Training Configuration

Number of epochs: 5
Evaluation schedule based on RMSE thresholds:

RMSE >= 0.50: Evaluate every 16 steps
RMSE >= 0.49: Evaluate every 8 steps
RMSE >= 0.48: Evaluate every 4 steps
RMSE >= 0.47: Evaluate every 2 steps
Below 0.47: Evaluate every step



Early Stopping

Implemented patience-based early stopping
Minimum delta: 1e-4
Patience: 10 evaluation cycles
Saves best model based on validation RMSE
