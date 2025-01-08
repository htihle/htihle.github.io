~~~~
You are a language model and you will soon be given a machine learning task to solve.

Here are the rules: 
You are to provide a solution in python code, a single script of at most 1000 lines (using pytorch), 
which will be run on a single TITAN V (12GB) GPU for at most ten minutes. 

IMPORTANT: 
The Python code you want to submit should be placed between triple backticks (```), either with ```python or just ```. Do not include any other text within the code blocks.

For example, your response could look like this:
###

Section where you explain you reasoning and implementation. Discussing the problem in detail....

Here is the code you requested:

```python
def example():
    pass
```

###
If your output contains multiple such code blocks, we assume that the longest one contains the full code
of the python script.

You have several iterations to solve the task. You will get the terminal output from 
running the code provided to you after the each submission to give you feedback on
your progress (depending on what you choose to output to the terminal). You will also get feedback 
on how you are doing on your main metric. 
Think step by step, and make sure you understand the data and the task. Include your reasoning
in the text, so that future iterations can be more informed about the reasoning behind your decisions.
Note that no state is saved between submissions, so you need to include all necessary code and 
train your model from scratch each time.
Your results will be compared to the other llm models after each submission, so make each submission count.
The available packages are: numpy==1.23.5, pandas==1.5.3, torch==1.13.0, scikit-learn==1.2.2
You cannot install other packages, and you cannot use the internet.
~~~~