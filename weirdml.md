---
layout: default
title: WeirdML
---


# WeirdML Benchmark

## Introduction

This benchmark evaluates Large Language Models (LLMs) on their ability to solve novel machine learning tasks. Unlike traditional benchmarks that test language understanding or mathematical reasoning, this benchmark specifically tests an LLM's capability to:
1. Understand machine learning problem descriptions
2. Generate working PyTorch code
3. Debug and improve solutions based on feedback
4. Handle resource constraints effectively

Each task is presented with a clear problem description and example code for loading data and saving predictions. The LLMs must then generate complete, working solutions that run within specified computational constraints.

## Evaluation Setup

The evaluation uses an automated pipeline that:
1. Presents the task to the LLM
2. Executes the generated code in an isolated environment
3. Evaluates the results against ground truth
4. Provides feedback to the LLM for improvement

### System Architecture
<div style="text-align: center">
    <img src="../images/evaluation_setup_diagram.png" width="500"/>
    <p><em></em></p>
</div>
The system executes code in a Docker container with strict resource limits (TITAN V GPU with 12GB memory, 600-second timeout). This ensures fair comparison between models and tests their ability to work within realistic constraints.

Each LLM gets multiple attempts (typically 5) per task, allowing them to learn from feedback and improve their solutions. The final performance metrics are based on the best performing attempt.

## Results

### Average results
<div style="text-align: center">
    <img src="../images/average_accuracy_across_tasks.png" width="800"/>
    <p><em>Evaluation pipeline showing the flow from LLM code generation through isolated execution to metric evaluation and feedback, with fixed computational constraints enforced via Docker.</em></p>
</div>
### Tasks

#### Shapes (Easy)
A shape classification task where models must identify one of five shapes (circle, square, triangle, pentagon, star) from a set of 2D coordinates. The shapes are always centered and have fixed orientation and size, making this the simpler variant of the shape recognition tasks.

#### Shapes (Hard)
Similar to Shapes (Easy), but with random positioning, orientation, and size of the shapes. This tests the model's ability to create translation, rotation, and scale invariant features.

#### Image Patch Shuffling (Easy)
Models must arrange 9 shuffled grayscale image patches (9x9 pixels each) to reconstruct the original 27x27 image. All patches are guaranteed to be part of a single, coherent image.

#### Image Patch Shuffling (Hard)
A more challenging version where patches are in RGB and taken from a random 27x27 subset of a larger 64x64 image, requiring more sophisticated visual understanding and spatial reasoning.

#### Chess Game Outcome Prediction
Predicts the outcome of chess games (white wins, black wins, or draw) from game move sequences. The data consists of games played by beginners (rated below 1300), with moves in standard algebraic notation.

#### Unsupervised Digit Recognition
A semi-supervised learning task where models must classify digits with only 26 labeled examples and a large set of unlabeled data. The challenge is complicated by uneven class distribution in the unlabeled set.



Shows the performance breakdown by task, highlighting which models excel at particular types of problems.

### Failure Analysis



Analysis of failure rates, showing how often models fail to produce working solutions within the given constraints.

## Conclusions
