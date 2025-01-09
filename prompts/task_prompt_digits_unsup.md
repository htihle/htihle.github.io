---
title: "Task Prompt: Unsupervised Digit Recognition"
---
~~~~
This is a challenge to test how good you are at doing machine learning on an unknown dataset. The data consists of images of digits displayed over a varying background.

The training data consists of a large array of dimension (15962, 15, 7, 3) which consists of 15962 samples of unlabeled images. Each image is a 15x7 image 
(i.e. 15 vertical pixels, 7 horizontal pixels, and 3 color channels). The pixel values are integers between 0 and 255. Note that there is a very uneven distribution of
samples in the unlabeled dataset, with some classes being very overrepresented, and others being very rare. There are also 26 labeled training samples,
each with a label from 1 to 6. The test data consists of 500 samples of 15x7 images with labels from 1 to 6. The test data follows the same distribution as the unlabeled training data,
while the labeled training data does not follow the same distribution, but has relatively more samples from the more rare classes.

Your goal is to train a machine learning model that can predict the correct label for each sample in the test set. The test accuracy is defined as the fraction of
test images with the correct label, and it and ranges from 0 to 1.

Here is example code showing how to load the data and output predictions:

```python
import torch
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

# Load data
train_data_unlabeled = np.load('data/train_data_unlabeled.npy')
train_data_labeled = np.load('data/train_data_labeled.npy')
train_labels = np.load('data/train_labels.npy').astype(np.int64)
test_data = np.load('data/test_data.npy')

print(f'Unlabeled training data shape: {train_data_unlabeled.shape}')    # (15962, 15, 7, 3)
print(f'Labeled training data shape: {train_data_labeled.shape}')        # (26, 15, 7, 3)
print(f'Training labels shape: {train_labels.shape}') # (26,)
print(f'Test data shape: {test_data.shape}')         # (500, 15, 7, 3)

# Create dummy predictions (random labels) for the test set between 1 and 6
test_preds = np.random.randint(1, 7, size=(test_data.shape[0])) 

# Save predictions
np.savetxt('results/test_preds.txt', test_preds, fmt='%d')
```

The challenge is to achieve the highest accuracy possible in predicting the correct label for the test set.
~~~~