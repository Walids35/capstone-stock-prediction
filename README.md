# Stock Prediction Framework

A comprehensive framework for predicting stock prices by analyzing the effect of financial news on stock movements. This project integrates data collection, sentiment analysis, feature engineering, and machine learning modeling to study and forecast stock price trends, with a focus on leveraging news sentiment as a predictive signal.

## Project Description

This repository provides tools and pipelines for:
- Collecting and processing financial news and stock market data
- Performing sentiment analysis on news articles using state-of-the-art NLP models
- Engineering features from both news and market data
- Training and evaluating machine learning models (including deep learning models in PyTorch) for stock price prediction
- Visualizing results and generating reports for analysis

The framework is modular, allowing for easy extension and experimentation with different models, features, and data sources.

## Project Structure

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         stock_prediction and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── stock_prediction   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes stock_prediction a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## Getting started

### Getting Started

1. **Create a Python virtual environment**  
   We recommend using [virtualenv](https://virtualenv.pypa.io/) or [venv](https://docs.python.org/3/library/venv.html) to manage your project dependencies.

2. **Install project requirements**  
   ```
   make requirements
   ```

3. **Run sentiment analysis preprocessing**  
   ```
   make sentiment_analysis
   ```

4. **Process and prepare the dataset**  
   ```
   make data
   ```

5. **Train or run models**  
   Use the provided scripts in `stock_prediction/modeling/` to train or make predictions. For example:
   ```
   make train
   make predict
   ```

--------

