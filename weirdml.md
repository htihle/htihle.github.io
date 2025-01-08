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

Each 'run' is 5 iterations, i.e. the LLM gets 5 submissions, and 4 rounds of feedback, allowing them to learn from feedback and improve their solutions. The accuracy of each run is the maximum accuracy achieved over all the submissions in that run.

For each task we give each model (at least) 15 runs, in order to take into account the large variance in performance that we see for the same model on the same task. The final score for each model on that task is the mean accuracy over all the runs.


## Tasks
The LLMs are evaluated on several different machine learning tasks. These tasks are intended to be possible to solve with a very limited amount of data, while still being hard to solve. They should also require the LLMs to think clearly and actually understand the data and its properties, not just blindly apply a standard ML recipe. 

### Shapes (Easy)

<div style="text-align: center">
    <img src="../images/train_examples_easy.png" width="800"/>
    <p><em>Example data from the Shapes (Easy) task. The shapes are always centered and have fixed orientation and size, making this the simpler variant of the shape recognition tasks.</em></p>
</div>


A shape classification task where models must identify one of five shapes (circle, square, triangle, pentagon, star) from a set of 512 2D coordinates. Only some of the points make up the shape, the other points are noise. The shapes are always centered and have fixed orientation and size, making this the simpler variant of the shape recognition tasks.

Here the model needs to come up with a way to encode the data that is invariant to permutations of the points. The distribution of points along the shape also varies greatly, so the model needs to combine information from many points to make a good prediction. 

<div style="text-align: center">
    <img src="../images/shapes_easy_max_accuracy_comparison.png" width="800"/>
    <p><em>Maximum accuracy for each run on the Shapes (Easy) task by each model. The bars show the mean value over all the runs. Error bars represent the standard deviation over runs (not the error on the mean). The grey dots represent individual runs, and the violin plots shows the distribution of accuracies over all the runs.</em></p>
</div>

### Shapes (Hard)
<div style="text-align: center">
    <img src="../images/train_examples_hard.png" width="800"/>
    <p><em>Example data from the Shapes (Hard) task. The shapes are randomly positioned, oriented, and sized, making this a more challenging variant of the shape recognition tasks.</em></p>
</div>

Similar to Shapes (Easy), but with random positioning, orientation, and size of the shapes. This tests the model's ability to create translation, rotation, and scale invariant features.

Here the model needs to come up with a way to encode the data that is (at least somewhat) invariant to translations, rotations, and scaling. Here it is crucial for the different points to be processed together, as it is the relative positions of the points that determine the shape. Good data augmentation is also crucial on this one. 

<div style="text-align: center">
    <img src="../images/shapes_hard_max_accuracy_comparison.png" width="800"/>
    <p><em>Maximum accuracy for each run on the Shapes (Hard) task by each model. The bars show the mean value over all the runs. Error bars represent the standard deviation over runs (not the error on the mean). The grey dots represent individual runs, and the violin plots shows the distribution of accuracies over all the runs.</em></p>
</div>

### Image Patch Shuffling (Easy)
<div style="text-align: center">
    <img src="../images/shuffle_easy_max_accuracy_comparison.png" width="800"/>
    <p><em>Maximum accuracy for each run on the Image Patch Shuffling (Easy) task by each model. The bars show the mean value over all the runs. Error bars represent the standard deviation over runs (not the error on the mean). The grey dots represent individual runs, and the violin plots shows the distribution of accuracies over all the runs.</em></p>
</div>

Models must arrange 9 shuffled grayscale image patches (9x9 pixels each) to reconstruct the original 27x27 image. All patches are guaranteed to be part of a single, coherent image. 

The original images here are from the fashion MNIST dataset, which is a greyscale dataset of 28x28 images of fashion items, with the items of clothing in the middle against a black background. This means that the position of an individual patch can often be inferred from the patch itself, since for example, a patch in the left of the image will tend to contain the left side of the item of clothing etc. This allows you to get a decent score even if you are not combining the information from the different patches in a good way.

### Image Patch Shuffling (Hard)
<div style="text-align: center">
    <img src="../images/shuffle_hard_max_accuracy_comparison.png" width="800"/>
    <p><em>Maximum accuracy for each run on the Image Patch Shuffling (Hard) task by each model. The bars show the mean value over all the runs. Error bars represent the standard deviation over runs (not the error on the mean). The grey dots represent individual runs, and the violin plots shows the distribution of accuracies over all the runs.</em></p>
</div>


A more challenging version where patches are in RGB and taken from a random 27x27 subset of a larger 64x64 image. The setup here is very similar to the easy version, but now you cannot infer the position of a patch from the patch itself, as the patches are taken from a random subset of the image (so a left patch can be taken from the center of the image). The original images are now also taken from imagnette (a subset of imagenet), which has a much more varied background and which makes it harder to infer the position of the individual patches. This means that the model needs to combine information from the different patches, and use the fact that the patches are supposed to fit well next to each other to make a good prediction.

### Chess Game Outcome Prediction
<div style="text-align: center">
    <img src="../images/chess_winners_max_accuracy_comparison.png" width="800"/>
    <p><em>Maximum accuracy for each run on the Chess Game Outcome Prediction task by each model. The bars show the mean value over all the runs. Error bars represent the standard deviation over runs (not the error on the mean). The grey dots represent individual runs, and the violin plots shows the distribution of accuracies over all the runs.</em></p>
</div>
Predicts the outcome of chess games (white wins, black wins, or draw) from game move sequences. The data consists of games played by beginners (rated below 1300), with moves in standard algebraic notation.

Here the models need to split the string into moves, then convert the string for each move into some kind of hand-crafted or learned features, and finally use these features to predict the outcome of the game, while dealing with the vaiable length of the chess games. Once some good features are found, there should be plenty of patterns that can be used to do significantly better than chance on predicting the outcome of the games.

### Unsupervised Digit Recognition
<div style="text-align: center">
    <img src="../images/digits_unsup_max_accuracy_comparison.png" width="800"/>
    <p><em>Maximum accuracy for each run on the Unsupervised Digit Recognition task by each model. The bars show the mean value over all the runs. Error bars represent the standard deviation over runs (not the error on the mean). The grey dots represent individual runs, and the violin plots shows the distribution of accuracies over all the runs.</em></p>
</div>
A semi-supervised learning task where models must classify digits with only 26 labeled examples and a large set of unlabeled data. The challenge is complicated by uneven class distribution in the unlabeled set. 

This is perhaps the most straightforward task, as a fairly standard semi-supervised learning recipe can be applied, but it is at least a dataset that the models have not seen before, and making semi-supervised learning work is not trivial.


## Failure Rates
<div style="text-align: center">
    <img src="../images/average_failure_rate_across_tasks.png" width="800"/>
    <p><em>Failure rate for each model on each task. The bars show the mean value over all the tasks. The grey markers represent failure rates on individual tasks.</em></p>
</div>

Failure here means an LLM response that does not produce any valid results. This could be that either the LLM response did not contain any valid python code, the code produced an error when run, or the code produced results that were not in the correct format (or for some other reason resulted in an accuracy of 0). 

Note that the failure rate here is defined for each submission (of which there are 5 per run), and not for each run. This means that a model can have fairly high failure rates and still get a good score, as long as it is able to produce some valid submissions, which produce good results, within the 5 tries it gets.
