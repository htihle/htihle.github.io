---
title: "Task Prompt: Shuffle (Easy)"
---
~~~~
This is a challenge to test how good you are at doing machine learning on an unknown dataset. The data consists of scrambled image patches that need to be arranged in the correct order.

The training data is an array of dimension (1000, 9, 3, 9, 9) which consists of 1000 samples, each containing 9 patches of size 9x9 pixels. These patches are taken from rgb images, 
with pixel values normalized between 0 and 1. The 9 patches should form a 27x27 square subset of a 64x64 image when arranged correctly, but they have been randomly shuffled. The position
of the 27x27 square subset within the original 64x64 image is also random.

The labels are an array of dimension (1000, 9), where each row is a permutation of the numbers 0-8. This permutation tells you where each patch should go in the correct arrangement. 
For example, if the first row of labels is [3,1,4,2,0,5,7,6,8], this means that the first patch in the input should go to position 3, the second patch should go to position 1, and so 
on. The positions are numbered from left to right, top to bottom (like reading a book).

Your goal is to train a machine learning model that can predict the correct arrangement of patches for each sample in the test set, with 1000 samples. The test set  
follows the same distribution as the training set. The model should take in 9 patches of 9x9 pixels each and predict the permutation that would arrange them correctly. 
The target metric is the accuracy of the model in predicting the correct permutation for each sample in the test set. Accuracy is defined on the flattened prediction and label arrays
for the test set, and ranges from 0 to 1.

Here is example code showing how to load the data and output predictions:

```python
import torch
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

# Load data
train_data = np.load('data/train_data.npy')
train_labels = np.load('data/train_labels.npy').astype(np.int64)
test_data = np.load('data/test_data.npy')

print(f'Training data shape: {train_data.shape}')    # (1000, 9, 3, 9, 9)
print(f'Training labels shape: {train_labels.shape}') # (1000, 9)
print(f'Test data shape: {test_data.shape}')         # (1000, 9, 3, 9, 9)

# Create dummy predictions (random permutations)
test_preds = np.array([np.random.permutation(9) for _ in range(test_data.shape[0])])

# Save predictions
np.savetxt('results/test_preds.txt', test_preds, fmt='%d')
```

The challenge is to achieve the highest accuracy possible in predicting the correct arrangement of patches.
~~~~
