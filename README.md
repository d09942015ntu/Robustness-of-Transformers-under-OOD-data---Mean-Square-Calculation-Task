## Overview

This project investigates the generalization behavior of language models in structured numerical reasoning tasks using multi-step autoregressive prediction. Specifically, the model is trained to compute the **mean** of four input numbers, and then predict the **square of that mean** in a second step.

Key characteristics of this setup:
- Training is conducted in a **teacher-forced multi-step** format.
- Evaluation is performed **autoregressively**, where the model's own prediction is used as input to the next step.
- To assess robustness, the model is evaluated on multiple **out-of-distribution (OOD)** test sets where input distributions shift systematically.

The pipeline includes dataset generation, training, and evaluation across distribution shifts. Code is structured to support fast prototyping and reproducible experiments.


## Step 1: Setup Environment

Ensure you have Python 3.10 or above installed on your system. You can check your Python version by running:

```
python --version
$ Python 3.10.12
```

If you need to update Python, visit the [official Python website](https://www.python.org/downloads/).

Next, it's recommended to create a virtual environment:

```
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Then, install the required packages:

```
pip install -r requirements.txt
```

## Step 2: Generate Dataset

Run the following command to set up the necessary datasets:

```
python3 data_generate.py
python3 ood_data_generate.py
```

## Step 3: Train Model

Run the following command to train the model
```
python3 training.py
```

## Step 4: Evaluate Model

Run the following command to evaluate the model
```
python3 evaluate.py
```