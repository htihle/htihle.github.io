---
layout: default
title: WeirdML
---
<style>
.highlight {
    background-color: #f3f7fc;
    border-left: 4px solid #0366d6;
    padding: 15px;
    margin: 20px 0;
    border-radius: 0 5px 5px 0;
}
</style>


# WeirdML Benchmark

<div class="highlight">
  <strong>New Developments:</strong> We're pleased to announce that WeirdML is now included in <a href="https://epoch.ai/data/ai-benchmarking-dashboard">Epoch AI's Benchmarking Hub</a>, we're also grateful to <a href="https://metr.org">METR</a> for supporting the API costs of this project.
</div>

Most recent updates and discussion of the results can be found on [<img src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/x.svg" width="11px" />](https://x.com/htihle).

## Introduction

How good are Large Language Models (LLMs) at doing machine learning on novel datasets? The WeirdML benchmark presents LLMs with weird and unusual machine learning tasks, designed to require careful thinking and actual understanding to solve, and tests an LLM's ability to:
1. Actually understand the properties of the data and the problem
2. Come up with an appropriate ML architecture and training setup for the problem, and generate working PyTorch code that implements the solution
3. Debug and improve the solution over 5 iterations based on terminal output and the accuracy on the test set
4. Make good use of limited computational resources and time

Each task comes with a task prompt describing the problem precisely and some example code for loading data and saving predictions. The different tasks pose various challenges: some require heavy data augmentation, others need careful feature engineering, or require combining information from many different parts of the input.

## UPDATED RESULTS

<div style="text-align: center">
    <img src="../images/average_accuracy_across_tasks_updated.png" width="800"/>
    <p><em>Average accuracy across all tasks for each model. Grey markers indicate performance on individual tasks, bars show the mean across tasks.</em></p>
</div>

<div style="text-align: center" class='task-performance-table-outer'>
    <div class='task-performance-table'>
        <style>
            .task-performance-table {
                font-size: 12px;  /* Scoped to just this table */
            }
            .task-performance-table table { 
                border-collapse: collapse; 
                width: 100%;
                margin: 20px 0;
                table-layout: fixed;  /* Enable fixed column widths */
            }
            .task-performance-table th, 
            .task-performance-table td { 
                padding: 4px;  /* Reduced padding */
                text-align: center;
                border: 1px solid #ddd;
                white-space: normal;  /* Allow text wrapping */
                word-wrap: break-word;
            }
            .task-performance-table th:first-child, 
            .task-performance-table td:first-child {
                width: 25%;  /* Make model column wider */
            }
            .task-performance-table th:not(:first-child), 
            .task-performance-table td:not(:first-child) {
                width: 7%;  /* Other columns share remaining space */
            }
            .task-performance-table th { 
                background-color: #f5f5f5;
                font-weight: bold;
            }
            .task-performance-table tr:nth-child(even) { 
                background-color: #f9f9f9; 
            }
            .task-performance-table .model-cell {
                text-align: left;
                font-weight: bold;
                padding-left: 15px;
                color: black !important;
            }
            .task-performance-table .header-cell {
                background-color: #f0f0f0;
                font-weight: bold;
                color: black;
            }
        </style>
        <table>
            <tr>
                <th class="header-cell">Model</th>
    <th class="header-cell">Shapes Easy</th><th class="header-cell">Shapes Hard</th><th class="header-cell">Shuffle Easy</th><th class="header-cell">Shuffle Hard</th><th class="header-cell">Digits Unsup</th><th class="header-cell">Chess Winners</th><th class="header-cell">Average</th></tr><tr style="background-color: #2dcc7025"><td class="model-cell">gemini-2.5-pro-exp-03-25</td><td>96.70%</td><td>37.33%</td><td>79.83%</td><td>11.77%</td><td>85.55%</td><td>55.09%</td><td><strong>61.05%</strong></td></tr><tr style="background-color: #4c4c4c25"><td class="model-cell">gpt-4.5-preview-2025-02-27</td><td>94.86%</td><td>33.88%</td><td>75.42%</td><td>11.89%</td><td>85.24%</td><td>60.30%</td><td><strong>60.26%</strong></td></tr><tr style="background-color: #4c4c4c25"><td class="model-cell">o4-mini-2025-04-16 (high)</td><td>97.55%</td><td>57.47%</td><td>33.35%</td><td>24.31%</td><td>79.92%</td><td>65.47%</td><td><strong>59.68%</strong></td></tr><tr style="background-color: #4c4c4c25"><td class="model-cell">o1-2024-12-17 (high)</td><td>93.86%</td><td>41.12%</td><td>50.66%</td><td>19.97%</td><td>89.40%</td><td>61.80%</td><td><strong>59.47%</strong></td></tr><tr style="background-color: #d75f3a25"><td class="model-cell">claude-3-7-sonnet-20250219 (thinking 8k)</td><td>95.74%</td><td>33.24%</td><td>45.48%</td><td>11.35%</td><td>83.00%</td><td>65.32%</td><td><strong>55.69%</strong></td></tr><tr style="background-color: #4c4c4c25"><td class="model-cell">o3-2025-04-16 (high)</td><td>96.40%</td><td>35.64%</td><td>36.16%</td><td>20.81%</td><td>72.08%</td><td>69.68%</td><td><strong>55.13%</strong></td></tr><tr style="background-color: #2dcc7025"><td class="model-cell">gemini-2.5-flash-preview (thinking 16k)</td><td>93.73%</td><td>24.58%</td><td>49.07%</td><td>11.65%</td><td>89.18%</td><td>54.36%</td><td><strong>53.76%</strong></td></tr><tr style="background-color: #d75f3a25"><td class="model-cell">claude-3-7-sonnet-20250219</td><td>92.99%</td><td>32.10%</td><td>29.15%</td><td>11.99%</td><td>85.19%</td><td>66.77%</td><td><strong>53.03%</strong></td></tr><tr style="background-color: #4c4c4c25"><td class="model-cell">gpt-4.1-mini-2025-04-14</td><td>93.43%</td><td>29.56%</td><td>56.07%</td><td>11.83%</td><td>68.84%</td><td>57.27%</td><td><strong>52.83%</strong></td></tr><tr style="background-color: #7f7f7f25"><td class="model-cell">grok-3-beta</td><td>89.13%</td><td>29.83%</td><td>54.32%</td><td>11.74%</td><td>70.67%</td><td>56.79%</td><td><strong>52.08%</strong></td></tr><tr style="background-color: #4d6bfe25"><td class="model-cell">deepseek-R1</td><td>95.21%</td><td>24.47%</td><td>56.61%</td><td>12.17%</td><td>66.51%</td><td>53.67%</td><td><strong>51.44%</strong></td></tr><tr style="background-color: #4c4c4c25"><td class="model-cell">o3-mini-2025-01-31 (high)</td><td>94.65%</td><td>30.61%</td><td>24.17%</td><td>11.76%</td><td>87.39%</td><td>59.42%</td><td><strong>51.33%</strong></td></tr><tr style="background-color: #d75f3a25"><td class="model-cell">claude-3-5-sonnet-20241022</td><td>84.79%</td><td>29.12%</td><td>47.39%</td><td>11.46%</td><td>80.27%</td><td>52.62%</td><td><strong>50.94%</strong></td></tr><tr style="background-color: #4d6bfe25"><td class="model-cell">deepseek-r1-0528</td><td>94.25%</td><td>27.39%</td><td>47.75%</td><td>11.61%</td><td>64.18%</td><td>57.01%</td><td><strong>50.37%</strong></td></tr><tr style="background-color: #4d6bfe25"><td class="model-cell">deepseek-chat-v3-0324</td><td>91.99%</td><td>27.31%</td><td>71.05%</td><td>11.49%</td><td>45.29%</td><td>52.59%</td><td><strong>49.95%</strong></td></tr><tr style="background-color: #4c4c4c25"><td class="model-cell">gpt-4.1-2025-04-14</td><td>91.97%</td><td>29.09%</td><td>33.89%</td><td>12.91%</td><td>69.15%</td><td>58.19%</td><td><strong>49.20%</strong></td></tr><tr style="background-color: #4c4c4c25"><td class="model-cell">o1-preview-2024-09-12</td><td>98.02%</td><td>32.00%</td><td>56.03%</td><td>11.80%</td><td>36.08%</td><td>58.96%</td><td><strong>48.82%</strong></td></tr><tr style="background-color: #7f7f7f25"><td class="model-cell">grok-3-mini-beta (high)</td><td>88.18%</td><td>28.89%</td><td>23.00%</td><td>11.58%</td><td>78.71%</td><td>55.69%</td><td><strong>47.67%</strong></td></tr><tr style="background-color: #ff7e0025"><td class="model-cell">qwen3-235b-a22b</td><td>88.56%</td><td>24.77%</td><td>42.93%</td><td>11.70%</td><td>51.85%</td><td>55.44%</td><td><strong>45.87%</strong></td></tr><tr style="background-color: #4c4c4c25"><td class="model-cell">o1-mini-2024-09-12</td><td>87.73%</td><td>26.49%</td><td>44.51%</td><td>11.44%</td><td>47.41%</td><td>55.89%</td><td><strong>45.58%</strong></td></tr><tr style="background-color: #7f7f7f25"><td class="model-cell">llama-4-maverick</td><td>66.57%</td><td>21.27%</td><td>43.09%</td><td>10.94%</td><td>70.35%</td><td>53.21%</td><td><strong>44.24%</strong></td></tr><tr style="background-color: #2dcc7025"><td class="model-cell">gemini-2.0-flash-thinking-exp-01-21</td><td>50.17%</td><td>21.53%</td><td>35.85%</td><td>11.09%</td><td>86.41%</td><td>58.37%</td><td><strong>43.90%</strong></td></tr><tr style="background-color: #d75f3a25"><td class="model-cell">claude-3-5-haiku-20241022</td><td>84.39%</td><td>23.85%</td><td>52.23%</td><td>10.07%</td><td>43.33%</td><td>48.66%</td><td><strong>43.75%</strong></td></tr><tr style="background-color: #ff7e0025"><td class="model-cell">qwen3-30b-a3b</td><td>63.87%</td><td>21.87%</td><td>44.45%</td><td>11.46%</td><td>52.61%</td><td>52.08%</td><td><strong>41.06%</strong></td></tr><tr style="background-color: #ff7e0025"><td class="model-cell">qwq-32b</td><td>83.94%</td><td>20.58%</td><td>24.98%</td><td>10.99%</td><td>45.96%</td><td>52.87%</td><td><strong>39.89%</strong></td></tr><tr style="background-color: #2dcc7025"><td class="model-cell">gemini-2.0-flash-exp</td><td>49.25%</td><td>21.76%</td><td>38.74%</td><td>9.83%</td><td>60.75%</td><td>53.98%</td><td><strong>39.05%</strong></td></tr><tr style="background-color: #2dcc7025"><td class="model-cell">gemini-2.0-pro-exp-02-05</td><td>47.41%</td><td>22.05%</td><td>39.53%</td><td>11.33%</td><td>57.15%</td><td>50.71%</td><td><strong>38.03%</strong></td></tr><tr style="background-color: #4d6bfe25"><td class="model-cell">deepseek-v3</td><td>72.65%</td><td>21.70%</td><td>32.99%</td><td>10.78%</td><td>33.82%</td><td>52.30%</td><td><strong>37.37%</strong></td></tr><tr style="background-color: #4c4c4c25"><td class="model-cell">gpt-4o-2024-11-20</td><td>68.95%</td><td>23.28%</td><td>36.75%</td><td>10.90%</td><td>29.77%</td><td>47.22%</td><td><strong>36.15%</strong></td></tr><tr style="background-color: #4c4c4c25"><td class="model-cell">gpt-4-turbo-2024-04-09</td><td>53.58%</td><td>23.09%</td><td>21.57%</td><td>9.85%</td><td>56.92%</td><td>36.57%</td><td><strong>33.60%</strong></td></tr><tr style="background-color: #7f7f7f25"><td class="model-cell">mistral-medium-3</td><td>64.53%</td><td>22.99%</td><td>26.60%</td><td>8.42%</td><td>23.35%</td><td>45.61%</td><td><strong>31.92%</strong></td></tr><tr style="background-color: #7f7f7f25"><td class="model-cell">grok-2-1212</td><td>70.59%</td><td>21.40%</td><td>14.53%</td><td>5.40%</td><td>14.36%</td><td>39.45%</td><td><strong>27.62%</strong></td></tr><tr style="background-color: #ff7e0025"><td class="model-cell">qwq:32b-preview-q8_0</td><td>57.06%</td><td>18.44%</td><td>16.56%</td><td>6.46%</td><td>10.94%</td><td>45.32%</td><td><strong>25.80%</strong></td></tr><tr style="background-color: #4c4c4c25"><td class="model-cell">gpt-4.1-nano-2025-04-14</td><td>71.63%</td><td>21.85%</td><td>16.61%</td><td>9.75%</td><td>20.76%</td><td>13.76%</td><td><strong>25.73%</strong></td></tr><tr style="background-color: #ff7e0025"><td class="model-cell">qwen2.5-coder:32b-instruct-q8_0</td><td>55.75%</td><td>18.67%</td><td>16.16%</td><td>6.52%</td><td>10.53%</td><td>44.89%</td><td><strong>25.42%</strong></td></tr><tr style="background-color: #4c4c4c25"><td class="model-cell">gpt-4o-mini-2024-07-18</td><td>32.93%</td><td>19.95%</td><td>18.83%</td><td>4.87%</td><td>27.00%</td><td>38.28%</td><td><strong>23.64%</strong></td></tr><tr style="background-color: #7f7f7f25"><td class="model-cell">llama3.3:70b-instruct-q8_0</td><td>40.86%</td><td>20.85%</td><td>9.86%</td><td>2.52%</td><td>22.35%</td><td>44.44%</td><td><strong>23.48%</strong></td></tr><tr style="background-color: #7f7f7f25"><td class="model-cell">llama3.1:405b-instruct-q4_K_M</td><td>41.91%</td><td>19.41%</td><td>5.19%</td><td>2.27%</td><td>31.85%</td><td>37.40%</td><td><strong>23.00%</strong></td></tr><tr style="background-color: #7f7f7f25"><td class="model-cell">phi4:14b-q8_0</td><td>26.48%</td><td>5.73%</td><td>3.30%</td><td>2.44%</td><td>12.92%</td><td>28.08%</td><td><strong>13.16%</strong></td></tr><tr style="background-color: #2dcc7025"><td class="model-cell">gemma-3-27b-it</td><td>32.75%</td><td>20.95%</td><td>0.87%</td><td>0.74%</td><td>0.00%</td><td>23.54%</td><td><strong>13.14%</strong></td></tr><tr style="background-color: #7f7f7f25"><td class="model-cell">llama-4-scout</td><td>22.77%</td><td>13.38%</td><td>5.70%</td><td>0.76%</td><td>3.87%</td><td>3.37%</td><td><strong>8.31%</strong></td></tr><tr style="background-color: #2dcc7025"><td class="model-cell">gemma2:27b-instruct-q8_0</td><td>21.51%</td><td>8.76%</td><td>0.00%</td><td>1.49%</td><td>0.00%</td><td>0.00%</td><td><strong>5.30%</strong></td></tr><tr style="background-color: #4c4c4c25"><td class="model-cell">gpt-3.5-turbo-0125</td><td>15.77%</td><td>11.03%</td><td>0.00%</td><td>0.00%</td><td>0.00%</td><td>3.63%</td><td><strong>5.07%</strong></td></tr><tr style="background-color: #7f7f7f25"><td class="model-cell">llama3.1:8b-instruct-q8_0</td><td>6.77%</td><td>2.22%</td><td>0.66%</td><td>0.76%</td><td>1.35%</td><td>0.00%</td><td><strong>1.96%</strong></td></tr>
        </table>
    </div>
    <p><em>Average accuracy across all six tasks for each model.</em></p>
</div>

These are updated results with newly available models. All results and discussion below is from the original release, and does not consider these updates. 
    
## Results

<div style="text-align: center">
    <img src="../images/average_accuracy_across_tasks.png" width="800"/>
    <p><em>Average accuracy across all tasks for each model. Grey markers indicate performance on individual tasks, bars show the mean across tasks.</em></p>
</div>

<div style="text-align: center" class='task-performance-table-outer'>
    <div class='task-performance-table'>
        <style>
            .task-performance-table {
                font-size: 12px;  /* Scoped to just this table */
            }
            .task-performance-table table { 
                border-collapse: collapse; 
                width: 100%;
                margin: 20px 0;
                table-layout: fixed;  /* Enable fixed column widths */
            }
            .task-performance-table th, 
            .task-performance-table td { 
                padding: 4px;  /* Reduced padding */
                text-align: center;
                border: 1px solid #ddd;
                white-space: normal;  /* Allow text wrapping */
                word-wrap: break-word;
            }
            .task-performance-table th:first-child, 
            .task-performance-table td:first-child {
                width: 25%;  /* Make model column wider */
            }
            .task-performance-table th:not(:first-child), 
            .task-performance-table td:not(:first-child) {
                width: 7%;  /* Other columns share remaining space */
            }
            .task-performance-table th { 
                background-color: #f5f5f5;
                font-weight: bold;
            }
            .task-performance-table tr:nth-child(even) { 
                background-color: #f9f9f9; 
            }
            .task-performance-table .model-cell {
                text-align: left;
                font-weight: bold;
                padding-left: 15px;
                color: black !important;
            }
            .task-performance-table .header-cell {
                background-color: #f0f0f0;
                font-weight: bold;
                color: black;
            }
        </style>
        <table>
            <tr>
                <th class="header-cell">Model</th>
    <th class="header-cell">Shapes Easy</th><th class="header-cell">Shapes Hard</th><th class="header-cell">Shuffle Easy</th><th class="header-cell">Shuffle Hard</th><th class="header-cell">Digits Unsup</th><th class="header-cell">Chess Winners</th><th class="header-cell">Average</th></tr><tr style="background-color: #d75f3a25"><td class="model-cell">claude-3-5-sonnet-20241022</td><td>84.79%</td><td>29.12%</td><td>47.39%</td><td>11.46%</td><td>80.27%</td><td>52.62%</td><td><strong>50.94%</strong></td></tr><tr style="background-color: #4c4c4c25"><td class="model-cell">o1-preview-2024-09-12</td><td>98.02%</td><td>32.00%</td><td>56.03%</td><td>11.80%</td><td>36.08%</td><td>58.96%</td><td><strong>48.82%</strong></td></tr><tr style="background-color: #4c4c4c25"><td class="model-cell">o1-mini-2024-09-12</td><td>87.73%</td><td>26.49%</td><td>44.51%</td><td>11.44%</td><td>47.41%</td><td>55.89%</td><td><strong>45.58%</strong></td></tr><tr style="background-color: #d75f3a25"><td class="model-cell">claude-3-5-haiku-20241022</td><td>84.39%</td><td>23.85%</td><td>52.23%</td><td>10.07%</td><td>43.33%</td><td>48.66%</td><td><strong>43.75%</strong></td></tr><tr style="background-color: #2dcc7025"><td class="model-cell">gemini-2.0-flash-thinking-exp-1219</td><td>68.65%</td><td>21.62%</td><td>46.05%</td><td>11.20%</td><td>56.08%</td><td>53.31%</td><td><strong>42.82%</strong></td></tr><tr style="background-color: #2dcc7025"><td class="model-cell">gemini-2.0-flash-exp</td><td>49.25%</td><td>21.76%</td><td>38.74%</td><td>9.83%</td><td>60.75%</td><td>53.98%</td><td><strong>39.05%</strong></td></tr><tr style="background-color: #7f7f7f25"><td class="model-cell">deepseek-v3</td><td>72.65%</td><td>21.70%</td><td>32.99%</td><td>10.78%</td><td>33.82%</td><td>52.30%</td><td><strong>37.37%</strong></td></tr><tr style="background-color: #4c4c4c25"><td class="model-cell">gpt-4o-2024-11-20</td><td>68.95%</td><td>23.28%</td><td>36.75%</td><td>10.90%</td><td>29.77%</td><td>47.22%</td><td><strong>36.15%</strong></td></tr><tr style="background-color: #2dcc7025"><td class="model-cell">gemini-exp-1206</td><td>43.25%</td><td>21.89%</td><td>40.62%</td><td>11.14%</td><td>37.55%</td><td>49.95%</td><td><strong>34.07%</strong></td></tr><tr style="background-color: #7f7f7f25"><td class="model-cell">grok-2-1212</td><td>70.59%</td><td>21.40%</td><td>14.53%</td><td>5.40%</td><td>14.36%</td><td>39.45%</td><td><strong>27.62%</strong></td></tr><tr style="background-color: #ff7e0025"><td class="model-cell">qwq:32b-preview-q8_0</td><td>57.06%</td><td>18.44%</td><td>16.56%</td><td>6.46%</td><td>10.94%</td><td>45.32%</td><td><strong>25.80%</strong></td></tr><tr style="background-color: #ff7e0025"><td class="model-cell">qwen2.5-coder:32b-instruct-q8_0</td><td>55.75%</td><td>18.67%</td><td>16.16%</td><td>6.52%</td><td>10.53%</td><td>44.89%</td><td><strong>25.42%</strong></td></tr><tr style="background-color: #4c4c4c25"><td class="model-cell">gpt-4o-mini-2024-07-18</td><td>32.93%</td><td>19.95%</td><td>18.83%</td><td>4.87%</td><td>27.00%</td><td>38.28%</td><td><strong>23.64%</strong></td></tr><tr style="background-color: #0079f725"><td class="model-cell">llama3.3:70b-instruct-q8_0</td><td>40.86%</td><td>20.85%</td><td>9.86%</td><td>2.52%</td><td>22.35%</td><td>44.44%</td><td><strong>23.48%</strong></td></tr><tr style="background-color: #0079f725"><td class="model-cell">llama3.1:405b-instruct-q4_K_M</td><td>41.91%</td><td>19.41%</td><td>5.19%</td><td>2.27%</td><td>31.85%</td><td>37.40%</td><td><strong>23.00%</strong></td></tr><tr style="background-color: #2dcc7025"><td class="model-cell">gemini-1.5-flash-002</td><td>42.03%</td><td>21.31%</td><td>3.56%</td><td>11.27%</td><td>-</td><td>-</td><td><strong>19.54%</strong></td></tr><tr style="background-color: #7f7f7f25"><td class="model-cell">phi4:14b-q8_0</td><td>26.48%</td><td>5.73%</td><td>3.30%</td><td>2.44%</td><td>12.92%</td><td>28.08%</td><td><strong>13.16%</strong></td></tr><tr style="background-color: #0079f725"><td class="model-cell">llama3.1:70b-instruct-q8_0</td><td>25.61%</td><td>9.61%</td><td>0.69%</td><td>-</td><td>-</td><td>-</td><td><strong>11.97%</strong></td></tr><tr style="background-color: #2dcc7025"><td class="model-cell">gemma2:27b-instruct-q8_0</td><td>21.51%</td><td>8.76%</td><td>0.00%</td><td>1.49%</td><td>0.00%</td><td>0.00%</td><td><strong>5.30%</strong></td></tr><tr style="background-color: #0079f725"><td class="model-cell">llama3.1:8b-instruct-q8_0</td><td>6.77%</td><td>2.22%</td><td>0.66%</td><td>0.76%</td><td>1.35%</td><td>0.00%</td><td><strong>1.96%</strong></td></tr>
        </table>
    </div>
    <p><em>Average accuracy across all six tasks for each model.</em></p>
</div>

## Evaluation Setup

The evaluation uses an automated pipeline that:
1. Presents the task to the LLM
2. Executes the generated code in an isolated environment
3. Evaluates the results against the test set
4. Provides feedback (terminal output from the code execution and test accuracy) to the LLM for improvement


<div style="text-align: center">
    <img src="../images/evaluation_setup_diagram.png" width="500"/>
    <p><em>Evaluation pipeline showing the flow from LLM code generation through isolated execution to metric evaluation and feedback, with fixed computational constraints enforced via Docker.</em></p>
</div>

### System Architecture
The system executes code in a Docker container with strict resource limits (TITAN V GPU with 12GB memory, 600-second timeout). This ensures fair comparison between models and tests their ability to work within realistic constraints. 

Each 'run' is 5 iterations, i.e., the LLM gets 5 submissions, and 4 rounds of feedback, allowing them to learn from feedback and improve their solutions ([full system prompt](prompts/system_prompt.md)). The accuracy of each run is the maximum test accuracy achieved over all the 5 submissions in that run.

For each task we give each model (at least) 15 runs (due to the high cost, o1-preview only gets 5 runs), in order to take into account the large variance in performance that we see for the same model on the same task. The final score for each model on that task is the mean accuracy over all the runs.


## Tasks
The LLMs are evaluated on several different machine learning tasks. These tasks are intended to be possible to solve with a very limited amount of data, while still being hard to solve. They should also require the LLMs to think clearly and actually understand the data and its properties, not just blindly apply a standard ML recipe. 



<div style="text-align: center">
    <img src="../images/train_examples_easy.png" width="600"/>
    <p><em>Example data from the Shapes (Easy) task. The shapes are always centered and have fixed orientation and size, making this the simpler variant of the shape recognition tasks.</em></p>
</div>

### Shapes (Easy)
A shape classification task ([task prompt](prompts/task_prompt_shapes_easy.md)) where models must identify one of five shapes (circle, square, triangle, pentagon, star) from a set of 512 2D coordinates. Only some of the points make up the shape, the other points are noise. The shapes are always centered and have fixed orientation and size, making this the simpler variant of the shape recognition tasks. The training set has 1000 samples. 

Here the model needs to come up with a way to encode the data that is invariant to permutations of the points. The distribution of points along the shape also varies greatly, so the model needs to combine information from many points to make a good prediction. 

<div style="text-align: center">
    <img src="../images/shapes_easy_max_accuracy_comparison.png" width="800"/>
    <p><em>Maximum accuracy for each run on the Shapes (Easy) task by each model. The bars show the mean value over all the runs. Error bars represent the standard deviation over runs (not the error on the mean). The grey dots represent individual runs, and the violin plots shows the distribution of accuracies over all the runs.</em></p>
</div>
We can see from the model performance that this is the easiest task. If you are not careful in your architecture, it is very easy to completely overfit on the training data, but if you do something somewhat reasonable, you should be able to get a decent score on this task. o1-preview got an average accuracy of 98% after 5 runs on this task, which is probably about the ceiling for this task.


<div style="text-align: center">
    <img src="../images/train_examples_hard.png" width="600"/>
    <p><em>Example data from the Shapes (Hard) task. The shapes are randomly positioned, oriented, and sized, making this a more challenging variant of the shape recognition tasks.</em></p>
</div>

### Shapes (Hard)
Similar to Shapes (Easy), but with random positioning, orientation, and size of the shapes ([task prompt](prompts/task_prompt_shapes_hard.md)). This tests the model's ability to create translation, rotation, and scale invariant features. Good data augmentation is also crucial on this one. 

<div style="text-align: center">
    <img src="../images/shapes_hard_max_accuracy_comparison.png" width="800"/>
    <p><em>Maximum accuracy for each run on the Shapes (Hard) task by each model. The bars show the mean value over all the runs. Error bars represent the standard deviation over runs (not the error on the mean). The grey dots represent individual runs, and the violin plots shows the distribution of accuracies over all the runs.</em></p>
</div>

While similar in structure to the easy version, this task is much harder. In the easy task, when the shapes are always in the same positions, the model can learn what positions correspond to what shapes. This is not possible here, now you need to use the relative position of the different points in a rotationally invariant and scale invariant way, which is much harder. 

The task is definitely solvable, but no models get consistently good results, and only a few models manage to sometimes get good runs here, with the best scores a bit above 60%, from claude-3-5-sonnet and o1-mini. Another notable result is qwq:32b managing a score of about 40% for its best run, which is impressive from such a small model. 


<div style="text-align: center">
    <img src="../images/scrambled_vs_unscrambled_easy.png" width="500"/>
    <p><em>Example data from the Image Patch Shuffling (Easy) task. Models must arrange 9 shuffled grayscale image patches (9x9 pixels each) to reconstruct the original 27x27 image.</em></p>
</div>

### Image Patch Shuffling (Easy)
Models must arrange 9 shuffled grayscale image patches (9x9 pixels each) to reconstruct the original 27x27 image. All patches are guaranteed to be part of a single, coherent image ([task prompt](prompts/task_prompt_shuffle_easy.md)). The training set has 1000 images. 

The original images here are from the fashion MNIST dataset, which is a greyscale dataset of 28x28 images of fashion items, with the items of clothing in the middle against a black background. This means that the position of an individual patch can often be inferred from the patch itself, since for example, a patch in the left of the image will tend to contain the left side of the item of clothing etc. This allows you to get a decent score even if you are not combining the information from the different patches in a good way.

<div style="text-align: center">
    <img src="../images/shuffle_easy_max_accuracy_comparison.png" width="800"/>
    <p><em>Maximum accuracy for each run on the Image Patch Shuffling (Easy) task by each model. The bars show the mean value over all the runs. Error bars represent the standard deviation over runs (not the error on the mean). The grey dots represent individual runs, and the violin plots shows the distribution of accuracies over all the runs.</em></p>
</div>
This is a task with the largest variations in the results for each single model. All models sometimes fail, or at leas get very low scores on this task, but most models also sometimes get a very good result. The patterns in the data should be easy to find if you have a reasonable architecture, but it may be a bit complicated to put all the pieces of the code together without making any mistakes, which the relatively high failure rate on this task suggests.


<div style="text-align: center">
    <img src="../images/scrambled_vs_unscrambled_hard.png" width="500"/>
    <p><em>Example data from the Image Patch Shuffling (Hard) task. Models must arrange 9 shuffled RGB image patches (9x9 pixels each) from a random 27x27 subset of a larger 64x64 image.</em></p>
</div>

### Image Patch Shuffling (Hard)
A more challenging version where patches are in RGB and taken from a random 27x27 subset of a larger 64x64 image ([task prompt](prompts/task_prompt_shuffle_hard.md)). The setup here is very similar to the easy version, but now you cannot infer the position of a patch from the patch itself, as the patches are taken from a random subset of the image (so a left patch can be taken from the center of the image). The original images are now also taken from imagnette (a subset of imagenet), which has a much more varied background and which makes it harder to infer the position of the individual patches. This means that the model needs to combine information from the different patches, and use the fact that the patches are supposed to fit well next to each other to make a good prediction.

<div style="text-align: center">
    <img src="../images/shuffle_hard_max_accuracy_comparison.png" width="800"/>
    <p><em>Maximum accuracy for each run on the Image Patch Shuffling (Hard) task by each model. The bars show the mean value over all the runs. Error bars represent the standard deviation over runs (not the error on the mean). The grey dots represent individual runs, and the violin plots shows the distribution of accuracies over all the runs.</em></p>
</div>

This is the task that the models struggle the most with. No models do significantly better than chance here. The main insight that (as far as I have seen) none of the models use is that they are given all the patches, and their correct positions, for the training data. This means that you can do the following data augmentation procedure:
1. Use the patches and the correct positions to recreate the original image
2. Apply standard image augmentation techniques to the recreated image
3. Divide into new patches and shuffle them in a new random order

Using this procedure will increase the effective size of the training set by a large factor. Combining this with crafting specific features that measure the smooth trasitions between the edges of the different patches, should allow the models to do significantly better on this task. It is unclear to me what the ceiling is for this task, but just looking at a few of the images, it seems that it should be possible to get a pretty good score here, if you use the right approach. 


<div style="text-align: center">
    <img src="../images/chess-games.png" width="600"/>
    <p><em>Example data from the Chess Game Outcome Prediction task. Models must predict the outcome of chess games (white wins, black wins, or draw) from game move sequences given as strings (here truncated).</em></p>
</div>

### Chess Game Outcome Prediction
Predict the outcome of chess games (white wins, black wins, or draw) from game move sequences ([task prompt](prompts/task_prompt_chess_winners.md)). The data consists of games played by beginners (rated below 1300), with moves in standard algebraic notation. Note that with 50% probability, the last move (for a single player) is removed, to prevent models using who moves last as a signal for the outcome. The training set has 1000 games.

Here the models need to split the string into moves, then convert the string for each move into some kind of hand-crafted or learned features, and finally use these features to predict the outcome of the game, while dealing with the variable length of the chess games. Once some good features are found, there should be plenty of patterns that can be used to do significantly better than chance on predicting the outcome of the games.

<div style="text-align: center">
    <img src="../images/chess_winners_max_accuracy_comparison.png" width="800"/>
    <p><em>Maximum accuracy for each run on the Chess Game Outcome Prediction task by each model. The bars show the mean value over all the runs. Error bars represent the standard deviation over runs (not the error on the mean). The grey dots represent individual runs, and the violin plots shows the distribution of accuracies over all the runs.</em></p>
</div>

Simply guessing white wins always will give you about 50% here, which is why I put the "random chance" line at 50% for this task. Most of the models manage to, at least sometimes get to about 60% accuracy, but struggle to do better than this. The best run is from claude-3-5-sonnet, which gets an accuracy of 74% using 20 handcrafted features. I suspect that with better handcrafted features (in principle you could track the full board state and craft features from that) you should be able to reach 90% accuracy or more, even with only 1000 games, but this is just a guess. 

<div style="text-align: center">
    <img src="../images/train_test_data.png" width="600"/>
    <p><em>Example data from the Unsupervised Digit Recognition task. Models must classify digits with only 26 labeled examples and a large set of unlabeled data.</em></p>
</div>


### Unsupervised Digit Recognition
A semi-supervised learning task where models must classify digits with only 26 labeled examples and a large set of unlabeled data ([task prompt](prompts/task_prompt_digits_unsup.md)). The challenge is complicated by uneven class distribution in the unlabeled set. The unlabeled training set is almost 16000 samples. 

This is perhaps the most straightforward task, as a fairly standard semi-supervised machine learning recipe can be applied, but it is at least a dataset that the models have not seen before, and making semi-supervised learning work at all is not trivial.

<div style="text-align: center">
    <img src="../images/digits_unsup_max_accuracy_comparison.png" width="800"/>
    <p><em>Maximum accuracy for each run on the Unsupervised Digit Recognition task by each model. The bars show the mean value over all the runs. Error bars represent the standard deviation over runs (not the error on the mean). The grey dots represent individual runs, and the violin plots shows the distribution of accuracies over all the runs.</em></p>
</div>

This task had by far the highest failure rate, with the models struggling to implement a complete semi-supervised training pipeline without making any mistakes. Once you get a working pipeline, however, you can get very good results, as it is a fairly easy dataset to classify. Given the high failure rate of the other models it is even more impressive how consistently great the results from claude-3-5-sonnet are, getting an average accuracy of 80%, and a median of over 90%. 

## Further Analysis
We have performed some very basic additional analysis of the results here. 


<div style="text-align: center">
    <img src="../images/average_failure_rate_across_tasks.png" width="800"/>
    <p><em>Failure rate for each model on each task. The bars show the mean value over all the tasks. The grey markers represent failure rates on individual tasks.</em></p>
</div>

### Failure Rate
Failure here means an LLM response that does not produce any valid results. This could be that either the LLM response did not contain any valid python code, the code produced an error when run, or the code produced results that were not in the correct format (or for some other reason resulted in an accuracy of 0). 

Note that the failure rate here is defined for each submission (of which there are 5 per run), and not for each run. This means that a model can have fairly high failure rates and still get a good score, as long as it is able to produce some valid submissions, which produce good results, within the 5 tries it gets.


<div style="text-align: center">
    <img src="../images/iteration_comparison.png" width="800"/>
    <p><em>Mean accuracy across all tasks for each model after 1, 2, 3, 4, and 5 iterations.</em></p>
</div>

### Model Performance by Number of Iterations
Here we see the mean accuracy over all the tasks after different number of iterations (the 5 iteration result here is the main result shown above). We see that the models do substantially better with more iterations. While there is clearly diminishing returns, it also seems that the accuracy will continue to increase with more than 5 iterations. Some models, like o1-preview show a steep increase in accuracy from 1 to 5 iterations, while others, like deepseek-v3, show much less improvement. 

Several factors are at play here, including the models ability to utilize the feedback, the models general failure rate, and many iterations simply giving you more tries to get a good result. Teasing out the different factors is hard based on the limited data here, but the next section does bring some more light to the question. All of this is surely very task dependent as well. Adding more tasks and more detailed analysis of the results in the future will help.

### Maximum of k First Submissions (max@k)
Similar to how pass@k means that at least one of k tries passes, max@k can be defined as the maximum accuracy of k tries. Here we use this to mean k first iterations (so the model gets no feedback). 3 of the models had over 50 runs on all the tasks, so there we actually have a decent number of first tries to look at for those models.

Comparing the performance of 5 first tries to 5 iterations with feedback tells you if the model actually uses the feedback productively or if it is better to use completely independent tries. As the models get smarter, they will be better at using the feedback efficiently, and the difference between the two measures should increase, so this is something to keep an eye on.
<div style="text-align: center">
    <img src="../images/maxk_comparison.png" width="800"/>
    <p><em>Maximuim mean accuracy across all tasks for each model after different number of first tries (max@k). Dashed lines show the mean result after 5 iterations for comparison.</em></p>
</div>
In the figure we see that for these three models, the 5 iteration result is better than the 5 first tries result, so the models are able to use the feedback, but the difference is small, suggesting that most of the benefit of more iterations comes from just getting more tries, and not from the actual feedback. 

It is interesting to note that the model with the largest benefit of 5 iterations over 5 independent tries is gemini-2.0-flash-thinking, Googles reasoning model. This suggest that the reasoning model is using the feedback more efficiently than the other models, and that its better overall results compared to gemini-2.0-flash is mostly due to this. Based on this one datapoint, we should not conclude much, but this observation is also consistent with o1-mini and o1-preview, OpenAIs reasoning models, having a larger relative improvement from 1 iteration to 5 iterations than for example claude-3-5-sonnet.
