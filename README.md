# RoBERTa-based Solution for Text Readability Assessment

## Overview
This solution implements a RoBERTa-based model for predicting text readability scores. The approach uses cross-validation, attention mechanisms, and a carefully designed training process to create a robust model for readability assessment.

## Model Architecture

### Base Model
- Used RoBERTa-base as the foundation model
- Added custom attention and regression layers
- Modified the model configuration to:
  - Enable output of hidden states
  - Set hidden dropout probability to 0.0
  - Set layer normalization epsilon to 1e-7

### Custom Layers
The model includes two additional components on top of RoBERTa:

1. Attention Layer:
```python
self.attention = nn.Sequential(
    nn.Linear(768, 512),
    nn.Tanh(),
    nn.Linear(512, 1),
    nn.Softmax(dim=1)
)
```

2. Regression Layer:
```python
self.regressor = nn.Sequential(
    nn.Linear(768, 1)
)
```

## Training Process

### Data Preparation
- Maximum sequence length: 248 tokens
- Batch size: 16
- Used 5-fold cross-validation
- Removed incomplete entries (where target and standard_error are both 0)

### Optimization Strategy
The training process uses a sophisticated layer-wise learning rate approach:
- Different learning rates for different layers of RoBERTa:
  - Base layers: 2e-5
  - Middle layers (69+): 5e-5
  - Upper layers (133+): 1e-4
- Adam optimizer with custom weight decay
- Cosine scheduler with warmup steps

### Early Stopping
- Implemented patience-based early stopping
- Minimum delta: 1e-4
- Patience: 10 evaluation cycles
- Saves best model based on validation RMSE

## Implementation Details

### Dataset Class
The custom dataset class (LitDataset) handles:
- Text tokenization with padding
- Attention mask generation
- Target value processing
- Support for both training and inference modes

### Model Training Features
- GPU support with automatic device selection
- Memory management with garbage collection
- Seed setting for reproducibility
- Progress tracking and timing measurements
- Comprehensive validation metrics

## Key Components

### Evaluation Function
```python
def eval_mse(model, val_dataloader, device):
    model.eval()
    total_loss, total_samples = 0, 0
    with torch.no_grad():
        for (input_ids, attention_mask, target) in val_dataloader:
            output = model(input_ids.to(device), attention_mask.to(device))
            loss = nn.MSELoss(reduction='sum')(output.flatten(), target.to(device))
            total_loss += loss.item()
            total_samples += target.size(0)
    return total_loss / total_samples
```

### Optimizer Creation
The optimizer creation function implements layer-wise learning rates and weight decay:
- Separate parameter groups for attention and regressor layers
- Layer-specific learning rates for RoBERTa layers
- Custom weight decay settings for bias terms

## Performance Monitoring
- Tracks training time per step
- Records validation RMSE at scheduled intervals
- Maintains best model state
- Implements early stopping based on validation performance

## Future Improvements
Potential areas for enhancement:
1. Data augmentation techniques
2. Ensemble methods with other models
3. Hyperparameter optimization
4. Additional feature engineering
5. More sophisticated learning rate scheduling


This implementation provides a solid foundation for text readability assessment and can be extended with additional features or modified for similar NLP tasks.
