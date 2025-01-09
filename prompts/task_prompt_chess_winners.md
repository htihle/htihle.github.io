---
title: Task Prompt: Chess Game Outcome Prediction
---
~~~~
This is a challenge to test how good you are at doing machine learning on an unknown dataset. The data consists of chess games 
played by beginners (players rated below 1300), where each game is represented as a sequence of moves in standard algebraic 
notation (SAN).

The training data consists of two files: 'train_games.txt' which contains 1000 chess games, with one game per line, and 
'train_labels.npy' which contains the corresponding labels. Each game in the text file is a string of moves like 
"1.e4 e5 2.Nf3 Nc6..." where the moves follow standard chess notation. Note that with 50% probability, the final single 
move (either white's or black's last move) of each game has been removed. For example, if a game ends with 
"1.e4 e5 2.Nf3 Nc6 3.Bb5 a6", it might be truncated to "1.e4 e5 2.Nf3 Nc6 3.Bb5". The labels are stored as integers 
(0: draw, 1: white wins, 2: black wins) in a numpy array.

The test set follows the same format and distribution, with 'test_games.txt' containing 1000 games and 'test_labels.npy' 
containing their labels. Your goal is to train a machine learning model that can predict the outcome of a chess game given 
its sequence of moves. The test accuracy is defined as the fraction of correctly predicted outcomes and ranges from 0 to 1.

Here is example code showing how to load the data and output predictions:

```python
import torch
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

# Load data
with open('data/train_games.txt', 'r', encoding='utf-8') as f:
    train_games = [line.strip() for line in f]
train_labels = np.load('data/train_labels.npy').astype(np.int64)
with open('data/test_games.txt', 'r', encoding='utf-8') as f:
    test_games = np.array([line.strip() for line in f])

print(f'Training games len: {len(train_games)}')    # 1000
print(f'Training labels shape: {train_labels.shape}')  # (1000,)
print(f'Test games len: {len(test_games)}')          # 1000

# Create dummy predictions (random labels between 0 and 2)
test_preds = np.random.randint(0, 3, size=(test_games.shape[0]))

# Save predictions
np.savetxt('results/test_preds.txt', test_preds, fmt='%d')
```

The challenge is to achieve the highest accuracy possible in predicting the outcome of chess games. 
~~~~