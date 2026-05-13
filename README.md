# Sentiment-Driven Market Analysis: Twitter Financial News

This repository contains a comprehensive pipeline for predicting market sentiment from financial tweets. The project benchmarks traditional statistical NLP models against modern Transformer-based architectures and introduces a hybrid inference approach for improved accuracy.

## Project Overview

The objective of this analysis is to classify financial tweets into three distinct sentiment categories:

* **0**: Bearish
* **1**: Bullish
* **2**: Neutral

The project utilizes the `zeroshot/twitter-financial-news-sentiment` dataset from the Hugging Face Hub, which consists of 9,543 training examples and 2,388 validation examples.

## Key Features

* **Pragmatic Preprocessing**: Text cleaning specialized for financial data, including emoji handling and tokenization.
* **Exploratory Data Analysis (EDA)**: Visualizations of label distributions and token frequencies using Seaborn and WordClouds.
* **Multi-Model Benchmarking**: Implementation and comparison of four distinct modeling strategies:
1. **TF-IDF + Logistic Regression** (Baseline)
2. **TF-IDF + Linear SVM**
3. **Fine-tuned BERT-base**
4. **Fine-tuned FinBERT + Rule-based Gating** (Proposed Hybrid)


* **Evaluation & Error Analysis**: Comprehensive performance tracking using Accuracy, F1-Score, Precision, and Recall, supplemented by confusion matrices and pragmatic error analysis.

## Prerequisites

To run the notebook, you will need the following Python libraries installed:

```bash
pip install datasets transformers accelerate evaluate scikit-learn emoji seaborn wordcloud

```

## Usage

1. **Open the Notebook**: Launch `twitter-sentiment-analysis-market.ipynb` in your Jupyter environment.
2. **Install Dependencies**: Run the initial setup cells to install and import the required libraries.
3. **Data Loading**: The notebook automatically fetches the dataset from Hugging Face.
4. **Execution**: Run cells sequentially to perform preprocessing, training, and evaluation.
* *Note: Using a GPU (CUDA) is recommended for fine-tuning the BERT and FinBERT models.*



## Results Summary

The notebook compares the models across standard metrics to determine which architecture best captures the nuances of financial language. The "Hybrid Inference" model aims to combine the deep contextual understanding of FinBERT with rule-based logic to handle common market-specific edge cases.

## License

This project is intended for educational and research purposes in the field of financial NLP.
