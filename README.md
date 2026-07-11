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
* **Multi-Model Benchmarking**: Implementation and comparison of five distinct modeling strategies:
1. **TF-IDF + Logistic Regression** (Baseline)
2. **TF-IDF + Linear SVM**
3. **DistilBERT-base-uncased (fine-tuned)**
4. **FinBERT (fine-tuned)**
5. **FinBERT + Pragmatic Gating (Hybrid)**

* **Web Application**: The finalized model (`geraldadli/twitter-sentiment-nlp`) is integrated into a Streamlit web interface for single-tweet inferences and batch analysis.
* **Evaluation & Error Analysis**: Comprehensive performance tracking using Accuracy, F1-Score, Precision, and Recall, supplemented by confusion matrices and pragmatic error analysis.

---

## 📊 Market Sentiment Analyser — Web Application

### App Name
**Market Sentiment Analyser**

### Description
A web-based tool that classifies financial tweets into Bullish, Bearish, or Neutral sentiments in real time. The app is powered by a fine-tuned DistilBERT model hosted on Hugging Face Hub (`geraldadli/twitter-sentiment-nlp`) and provides instant sentiment predictions with confidence scores.

### Main Features
* **Single-Tweet Analysis**: Paste any financial tweet and get an instant sentiment prediction with confidence scores across all three classes, token count, and processing time.
* **Batch Analysis**: Upload a CSV file containing multiple tweets for bulk sentiment classification with an aggregated sentiment distribution summary.
* **Confidence Visualization**: Interactive probability bars showing the model's confidence for each sentiment class (Bullish, Bearish, Neutral).
* **Token Usage Tracking**: Displays how many tokens the input uses relative to the model's maximum context length (64 tokens).
* **Financial Emoji Handling**: Automatically converts financial emojis (📈, 📉, 🚀, etc.) into meaningful tokens before classification.
* **Dark-Themed UI**: A polished, dark-mode interface built with custom CSS and Google Fonts (Space Grotesk & JetBrains Mono).

### Technology Used
| Category | Technology |
|---|---|
| Frontend / UI | Streamlit |
| Language | Python |
| Model | DistilBERT-base-uncased (fine-tuned) |
| Model Hub | Hugging Face Hub (`geraldadli/twitter-sentiment-nlp`) |
| ML Framework | PyTorch, Hugging Face Transformers |
| Data Processing | Pandas, NumPy |
| Text Preprocessing | Regex, custom emoji tokenizer |
| Deployment | Streamlit Cloud |

### How to Run the App

**Option 1 — Live Demo (no setup needed)**

Visit the deployed app at: [https://twitter-sentiment-nlp-tfidf.streamlit.app/](https://twitter-sentiment-nlp-tfidf.streamlit.app/)

**Option 2 — Run Locally**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/geraldadli/twitter-sentiment-nlp.git
   cd twitter-sentiment-nlp
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the app**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** at `http://localhost:8501` and start analyzing tweets.

> **Note**: The model weights are downloaded automatically from Hugging Face Hub on the first run. No GPU is required — the app runs on CPU.

---

## Running the Notebook

To run the training and evaluation notebook, you will need the following Python libraries installed:

```bash
pip install datasets transformers accelerate evaluate scikit-learn emoji seaborn wordcloud
```

1. **Open the Notebook**: Launch `twitter-sentiment-analysis-market.ipynb` in your Jupyter environment.
2. **Install Dependencies**: Run the initial setup cells to install and import the required libraries.
3. **Data Loading**: The notebook automatically fetches the dataset from Hugging Face.
4. **Execution**: Run cells sequentially to perform preprocessing, training, and evaluation.
   * *Note: Using a GPU (CUDA) is recommended for fine-tuning the BERT and FinBERT models.*

## Results Summary

The notebook compares the models across standard metrics to determine which architecture best captures the nuances of financial language. The "Hybrid Inference" model aims to combine the deep contextual understanding of FinBERT with rule-based logic to handle common market-specific edge cases.

## License

This project is intended for educational and research purposes in the field of financial NLP.
