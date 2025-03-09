# WritingAssistance

This repository evaluates the performance of three models—DeBERTa, XLM-RoBERTa, and GPT-4o-mini—in predicting sentence formality using a dataset from
[Huggingface](https://huggingface.co/datasets/osyvokon/pavlick-formality-scores).

## Navigation

The source code is located in the `src/` directory, with additional scripts in the `scripts/` folder. Notebooks can be found in the `notebooks/` directory. I prepared report `report.pdf` that summarizes my work also in notebook form in `notebooks/results`.

## Installation

If you want to setup project locally

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/Micz26/WritingAssistance.git

   ```

2. Open cloned repository and create new virtual environment:

   If you use _conda_

   ```
   conda create --name your-environment-name python=3.10
   ```

   Alternatively use any other virtual enviroment manager of your choice.

3. Activate environment

   ```
   conda activate your-environment-name
   ```

4. Make sure you use recent _pip_ version

   ```
   python -m pip install --upgrade pip
   ```

5. Install packages

   ```
   python -m pip install -e .
   ```

6. [Optional] If you wish to use openai's GPT formality detector, create `.env` file and paste your OpenAI API Key

   ```
   OPEN_AI_API_KEY = "<yourkey>"
   ```

7. [Optional] Enable pre-commit for development

   ```
   pre-commit install
   ```

After these steps project scripts are ready to launch

## Scripts

1. run_evaluation_pipeline.py

This script evaluates a selected model on a chosen dataset. It runs the entire evaluation process on the dataset, calculating various metrics such as RMSE, MAE, R².
Usage:

```
python scripts/run_evaluation_pipeline.py
```

2. detect_formality.py

This script allows you to input any text and the name of a model, then it predicts the formality of the given sentence using the specified model.
Usage:

```
python scripts/detect_formality.py
```
