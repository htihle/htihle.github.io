---
layout: default
title: WeirdML
---


# WeirdML Benchmark

## Introduction

This benchmark evaluates Large Language Models (LLMs) on their ability to solve novel machine learning tasks, specifically it tests an LLM's capability to:
1. Understand machine learning problem descriptions
2. Generate working PyTorch code
3. Debug and improve solutions based on feedback
4. Handle resource constraints effectively

Each task is presented with a clear problem description and example code for loading data and saving predictions. The LLMs must then generate complete, working solutions that run within specified computational constraints.


## Results

<div style="text-align: center">
    <img src="../images/average_accuracy_across_tasks.png" width="800"/>
    <p><em>Average accuracy across all tasks for each model. Grey markers indicate performance on individual tasks, bars show the mean across tasks.</em></p>
</div>

<div style="text-align: center" class='task-performance-table'>
    <html>
    <head>
        <style>
            body { 
                font-family: Arial, sans-serif;
                font-size: 12px;  /* Smaller base font size */
            }
            table { 
                border-collapse: collapse; 
                width: 100%;
                margin: 20px 0;
                table-layout: fixed;  /* Enable fixed column widths */
            }
            th, td { 
                padding: 4px;  /* Reduced padding */
                text-align: center;
                border: 1px solid #ddd;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            th:first-child, td:first-child {
                width: 25%;  /* Make model column wider */
            }
            th:not(:first-child), td:not(:first-child) {
                width: 7%;  /* Other columns share remaining space */
            }
            th { 
                background-color: #f5f5f5;
                font-weight: bold;
                font-size: 12px;  /* Match base font size */
            }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .model-cell {
                text-align: left;
                font-weight: bold;
                padding-left: 15px;
                color: black !important;  /* Force black text */
            }
            .header-cell {
                background-color: #f0f0f0;
                font-weight: bold;
                color: black;
            }
        </style>
    </head>
    <body>
    <table>
        <tr>
            <th class="header-cell">Model</th>
    <th class="header-cell">Shapes Easy</th><th class="header-cell">Shapes Hard</th><th class="header-cell">Shuffle Easy</th><th class="header-cell">Shuffle Hard</th><th class="header-cell">Digits Unsup</th><th class="header-cell">Chess Winners</th><th class="header-cell">Average</th></tr><tr style="background-color: #d75f3a25"><td class="model-cell">claude-3-5-sonnet-20241022</td><td>84.79%</td><td>29.12%</td><td>47.39%</td><td>11.46%</td><td>80.27%</td><td>52.62%</td><td><strong>50.94%</strong></td></tr><tr style="background-color: #4c4c4c25"><td class="model-cell">o1-preview-2024-09-12</td><td>98.70%</td><td>24.70%</td><td>50.02%</td><td>12.07%</td><td>34.40%</td><td>53.50%</td><td><strong>45.56%</strong></td></tr><tr style="background-color: #4c4c4c25"><td class="model-cell">o1-mini-2024-09-12</td><td>86.92%</td><td>23.80%</td><td>44.57%</td><td>11.42%</td><td>43.20%</td><td>57.14%</td><td><strong>44.51%</strong></td></tr><tr style="background-color: #d75f3a25"><td class="model-cell">claude-3-5-haiku-20241022</td><td>84.39%</td><td>23.85%</td><td>52.23%</td><td>10.07%</td><td>43.33%</td><td>48.66%</td><td><strong>43.75%</strong></td></tr><tr style="background-color: #2dcc7025"><td class="model-cell">gemini-2.0-flash-thinking-exp-1219</td><td>68.48%</td><td>21.63%</td><td>45.84%</td><td>11.20%</td><td>55.54%</td><td>53.37%</td><td><strong>42.68%</strong></td></tr><tr style="background-color: #2dcc7025"><td class="model-cell">gemini-2.0-flash-exp</td><td>49.46%</td><td>21.76%</td><td>38.65%</td><td>9.97%</td><td>60.78%</td><td>54.02%</td><td><strong>39.11%</strong></td></tr><tr style="background-color: #7f7f7f25"><td class="model-cell">deepseek-v3</td><td>72.74%</td><td>21.71%</td><td>32.78%</td><td>10.77%</td><td>33.40%</td><td>52.31%</td><td><strong>37.29%</strong></td></tr><tr style="background-color: #4c4c4c25"><td class="model-cell">gpt-4o-2024-11-20</td><td>68.95%</td><td>23.28%</td><td>36.75%</td><td>10.90%</td><td>29.77%</td><td>47.22%</td><td><strong>36.15%</strong></td></tr><tr style="background-color: #2dcc7025"><td class="model-cell">gemini-exp-1206</td><td>43.25%</td><td>21.82%</td><td>38.65%</td><td>11.86%</td><td>37.55%</td><td>49.95%</td><td><strong>33.85%</strong></td></tr><tr style="background-color: #7f7f7f25"><td class="model-cell">grok-2-1212</td><td>70.59%</td><td>21.40%</td><td>14.53%</td><td>5.40%</td><td>14.36%</td><td>39.45%</td><td><strong>27.62%</strong></td></tr><tr style="background-color: #ff7e0025"><td class="model-cell">qwq:32b-preview-q8_0</td><td>57.06%</td><td>18.44%</td><td>16.56%</td><td>6.46%</td><td>10.94%</td><td>45.32%</td><td><strong>25.80%</strong></td></tr><tr style="background-color: #ff7e0025"><td class="model-cell">qwen2.5-coder:32b-instruct-q8_0</td><td>55.75%</td><td>18.67%</td><td>16.16%</td><td>6.52%</td><td>10.53%</td><td>44.89%</td><td><strong>25.42%</strong></td></tr><tr style="background-color: #4c4c4c25"><td class="model-cell">gpt-4o-mini-2024-07-18</td><td>32.93%</td><td>19.95%</td><td>18.83%</td><td>4.87%</td><td>27.00%</td><td>38.28%</td><td><strong>23.64%</strong></td></tr><tr style="background-color: #0079f725"><td class="model-cell">llama3.3:70b-instruct-q8_0</td><td>40.86%</td><td>20.85%</td><td>9.86%</td><td>2.52%</td><td>22.35%</td><td>44.44%</td><td><strong>23.48%</strong></td></tr><tr style="background-color: #0079f725"><td class="model-cell">llama3.1:405b-instruct-q4_K_M</td><td>41.91%</td><td>19.41%</td><td>5.19%</td><td>2.27%</td><td>31.85%</td><td>37.40%</td><td><strong>23.00%</strong></td></tr><tr style="background-color: #2dcc7025"><td class="model-cell">gemini-1.5-flash-002</td><td>42.03%</td><td>21.31%</td><td>3.56%</td><td>11.27%</td><td>-</td><td>-</td><td><strong>19.54%</strong></td></tr><tr style="background-color: #7f7f7f25"><td class="model-cell">phi4:14b-q8_0</td><td>26.48%</td><td>5.73%</td><td>3.30%</td><td>2.44%</td><td>12.92%</td><td>28.08%</td><td><strong>13.16%</strong></td></tr><tr style="background-color: #0079f725"><td class="model-cell">llama3.1:70b-instruct-q8_0</td><td>25.61%</td><td>9.61%</td><td>0.69%</td><td>-</td><td>-</td><td>-</td><td><strong>11.97%</strong></td></tr><tr style="background-color: #2dcc7025"><td class="model-cell">gemma2:27b-instruct-q8_0</td><td>21.51%</td><td>8.76%</td><td>0.00%</td><td>1.49%</td><td>0.00%</td><td>0.00%</td><td><strong>5.30%</strong></td></tr><tr style="background-color: #0079f725"><td class="model-cell">llama3.1:8b-instruct-q8_0</td><td>6.77%</td><td>2.22%</td><td>0.66%</td><td>0.76%</td><td>1.35%</td><td>0.00%</td><td><strong>1.96%</strong></td></tr>
    </table>
    </body>
    </html>
    <p><em>Table caption.</em></p>
</div>

## Evaluation Setup

The evaluation uses an automated pipeline that:
1. Presents the task to the LLM
2. Executes the generated code in an isolated environment
3. Evaluates the results against ground truth
4. Provides feedback to the LLM for improvement

### System Architecture
<div style="text-align: center">
    <img src="../images/evaluation_setup_diagram.png" width="500"/>
    <p><em>Evaluation pipeline showing the flow from LLM code generation through isolated execution to metric evaluation and feedback, with fixed computational constraints enforced via Docker.</em></p>
</div>
The system executes code in a Docker container with strict resource limits (TITAN V GPU with 12GB memory, 600-second timeout). This ensures fair comparison between models and tests their ability to work within realistic constraints. 

Each 'run' is 5 iterations, i.e. the LLM gets 5 submissions, and 4 rounds of feedback, allowing them to learn from feedback and improve their solutions ([full system prompt](prompts/system_prompt.md)). The accuracy of each run is the maximum accuracy achieved over all the submissions in that run.

For each task we give each model (at least) 15 runs, in order to take into account the large variance in performance that we see for the same model on the same task. The final score for each model on that task is the mean accuracy over all the runs.


## Tasks
The LLMs are evaluated on several different machine learning tasks. These tasks are intended to be possible to solve with a very limited amount of data, while still being hard to solve. They should also require the LLMs to think clearly and actually understand the data and its properties, not just blindly apply a standard ML recipe. 

### Shapes (Easy)

<div style="text-align: center">
    <img src="../images/train_examples_easy.png" width="600"/>
    <p><em>Example data from the Shapes (Easy) task. The shapes are always centered and have fixed orientation and size, making this the simpler variant of the shape recognition tasks.</em></p>
</div>


A shape classification task ([task prompt](prompts/task_prompt_shapes_easy.md)) where models must identify one of five shapes (circle, square, triangle, pentagon, star) from a set of 512 2D coordinates. Only some of the points make up the shape, the other points are noise. The shapes are always centered and have fixed orientation and size, making this the simpler variant of the shape recognition tasks.

Here the model needs to come up with a way to encode the data that is invariant to permutations of the points. The distribution of points along the shape also varies greatly, so the model needs to combine information from many points to make a good prediction. 

<div style="text-align: center">
    <img src="../images/shapes_easy_max_accuracy_comparison.png" width="800"/>
    <p><em>Maximum accuracy for each run on the Shapes (Easy) task by each model. The bars show the mean value over all the runs. Error bars represent the standard deviation over runs (not the error on the mean). The grey dots represent individual runs, and the violin plots shows the distribution of accuracies over all the runs.</em></p>
</div>

### Shapes (Hard)
<div style="text-align: center">
    <img src="../images/train_examples_hard.png" width="600"/>
    <p><em>Example data from the Shapes (Hard) task. The shapes are randomly positioned, oriented, and sized, making this a more challenging variant of the shape recognition tasks.</em></p>
</div>

Similar to Shapes (Easy), but with random positioning, orientation, and size of the shapes ([task prompt](prompts/task_prompt_shapes_hard.md)). This tests the model's ability to create translation, rotation, and scale invariant features.

Here the model needs to come up with a way to encode the data that is (at least somewhat) invariant to translations, rotations, and scaling. Here it is crucial for the different points to be processed together, as it is the relative positions of the points that determine the shape. Good data augmentation is also crucial on this one. 

<div style="text-align: center">
    <img src="../images/shapes_hard_max_accuracy_comparison.png" width="800"/>
    <p><em>Maximum accuracy for each run on the Shapes (Hard) task by each model. The bars show the mean value over all the runs. Error bars represent the standard deviation over runs (not the error on the mean). The grey dots represent individual runs, and the violin plots shows the distribution of accuracies over all the runs.</em></p>
</div>

### Image Patch Shuffling (Easy)
<div style="text-align: center">
    <img src="../images/scrambled_vs_unscrambled_easy.png" width="500"/>
    <p><em>Example data from the Image Patch Shuffling (Easy) task. Models must arrange 9 shuffled grayscale image patches (9x9 pixels each) to reconstruct the original 27x27 image.</em></p>
</div>


Models must arrange 9 shuffled grayscale image patches (9x9 pixels each) to reconstruct the original 27x27 image. All patches are guaranteed to be part of a single, coherent image ([task prompt](prompts/task_prompt_shuffle_easy.md)).

The original images here are from the fashion MNIST dataset, which is a greyscale dataset of 28x28 images of fashion items, with the items of clothing in the middle against a black background. This means that the position of an individual patch can often be inferred from the patch itself, since for example, a patch in the left of the image will tend to contain the left side of the item of clothing etc. This allows you to get a decent score even if you are not combining the information from the different patches in a good way.

<div style="text-align: center">
    <img src="../images/shuffle_easy_max_accuracy_comparison.png" width="800"/>
    <p><em>Maximum accuracy for each run on the Image Patch Shuffling (Easy) task by each model. The bars show the mean value over all the runs. Error bars represent the standard deviation over runs (not the error on the mean). The grey dots represent individual runs, and the violin plots shows the distribution of accuracies over all the runs.</em></p>
</div>

### Image Patch Shuffling (Hard)
<div style="text-align: center">
    <img src="../images/scrambled_vs_unscrambled_hard.png" width="500"/>
    <p><em>Example data from the Image Patch Shuffling (Hard) task. Models must arrange 9 shuffled RGB image patches (9x9 pixels each) from a random 27x27 subset of a larger 64x64 image.</em></p>
</div>

A more challenging version where patches are in RGB and taken from a random 27x27 subset of a larger 64x64 image ([task prompt](prompts/task_prompt_shuffle_hard.md)). The setup here is very similar to the easy version, but now you cannot infer the position of a patch from the patch itself, as the patches are taken from a random subset of the image (so a left patch can be taken from the center of the image). The original images are now also taken from imagnette (a subset of imagenet), which has a much more varied background and which makes it harder to infer the position of the individual patches. This means that the model needs to combine information from the different patches, and use the fact that the patches are supposed to fit well next to each other to make a good prediction.

<div style="text-align: center">
    <img src="../images/shuffle_hard_max_accuracy_comparison.png" width="800"/>
    <p><em>Maximum accuracy for each run on the Image Patch Shuffling (Hard) task by each model. The bars show the mean value over all the runs. Error bars represent the standard deviation over runs (not the error on the mean). The grey dots represent individual runs, and the violin plots shows the distribution of accuracies over all the runs.</em></p>
</div>

### Chess Game Outcome Prediction
<div style="text-align: center">
    <img src="../images/chess-games.png" width="600"/>
    <p><em>Example data from the Chess Game Outcome Prediction task. Models must predict the outcome of chess games (white wins, black wins, or draw) from game move sequences given as strings (here truncated).</em></p>
</div>

Predict the outcome of chess games (white wins, black wins, or draw) from game move sequences ([task prompt](prompts/task_prompt_chess_winners.md)). The data consists of games played by beginners (rated below 1300), with moves in standard algebraic notation. Note that with 50% probability, the last move (for a single player) is removed, to prevent models using who moves last as a signal for the outcome.

Here the models need to split the string into moves, then convert the string for each move into some kind of hand-crafted or learned features, and finally use these features to predict the outcome of the game, while dealing with the vaiable length of the chess games. Once some good features are found, there should be plenty of patterns that can be used to do significantly better than chance on predicting the outcome of the games.

<div style="text-align: center">
    <img src="../images/chess_winners_max_accuracy_comparison.png" width="800"/>
    <p><em>Maximum accuracy for each run on the Chess Game Outcome Prediction task by each model. The bars show the mean value over all the runs. Error bars represent the standard deviation over runs (not the error on the mean). The grey dots represent individual runs, and the violin plots shows the distribution of accuracies over all the runs.</em></p>
</div>

### Unsupervised Digit Recognition
<div style="text-align: center">
    <img src="../images/train_test_data.png" width="600"/>
    <p><em>Example data from the Unsupervised Digit Recognition task. Models must classify digits with only 26 labeled examples and a large set of unlabeled data.</em></p>
</div>

A semi-supervised learning task where models must classify digits with only 26 labeled examples and a large set of unlabeled data ([task prompt](prompts/task_prompt_digits_unsup.md)). The challenge is complicated by uneven class distribution in the unlabeled set. 

This is perhaps the most straightforward task, as a fairly standard semi-supervised learning recipe can be applied, but it is at least a dataset that the models have not seen before, and making semi-supervised learning work is not trivial.

<div style="text-align: center">
    <img src="../images/digits_unsup_max_accuracy_comparison.png" width="800"/>
    <p><em>Maximum accuracy for each run on the Unsupervised Digit Recognition task by each model. The bars show the mean value over all the runs. Error bars represent the standard deviation over runs (not the error on the mean). The grey dots represent individual runs, and the violin plots shows the distribution of accuracies over all the runs.</em></p>
</div>

## Failure Rates
<div style="text-align: center">
    <img src="../images/average_failure_rate_across_tasks.png" width="800"/>
    <p><em>Failure rate for each model on each task. The bars show the mean value over all the tasks. The grey markers represent failure rates on individual tasks.</em></p>
</div>

Failure here means an LLM response that does not produce any valid results. This could be that either the LLM response did not contain any valid python code, the code produced an error when run, or the code produced results that were not in the correct format (or for some other reason resulted in an accuracy of 0). 

Note that the failure rate here is defined for each submission (of which there are 5 per run), and not for each run. This means that a model can have fairly high failure rates and still get a good score, as long as it is able to produce some valid submissions, which produce good results, within the 5 tries it gets.
