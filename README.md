# Sarcasm Detection on Twitter

A machine learning and NLP project for detecting sarcasm in Twitter/X text using handcrafted linguistic features, TF-IDF representations, traditional ML models, and transformer-based models.

## Overview

Sarcasm detection is challenging because sarcastic text often expresses the opposite of its literal meaning. This project studies sarcasm detection using Twitter conversation data, where both the reply and its parent context are available.

The project focuses on feature engineering and model comparison to understand which text patterns help identify sarcasm.

## Dataset

The dataset is loaded from the Educational Testing Service sarcasm shared task repository.

Dataset used:

* FigLang 2020 Twitter Sarcasm Detection dataset
* Training data: Twitter replies with conversational context
* Testing data: Separate held-out test set
* Labels:

  * `1` → Sarcastic
  * `0` → Non-sarcastic

Each record contains:

* `response`: the target tweet/reply
* `parent`: the immediate parent/context tweet
* `full_context`: complete available context
* `label`: sarcasm label

## Features Extracted

A total of **55 handcrafted features** were extracted from the text.

### Feature Groups

| Group                     | Description                                                                            |
| ------------------------- | -------------------------------------------------------------------------------------- |
| Standard NLP              | VADER sentiment, POS ratios, lexical diversity, word count                             |
| Rhetorical Features       | polarity gap, fake agreement openers, rhetorical questions, negation-positive patterns |
| Casing Features           | uppercase ratio, all-caps words, mixed casing                                          |
| Punctuation Features      | exclamation marks, question marks, repeated punctuation, interrobang, ellipsis         |
| Intensifiers              | intensifier count, density, weighted intensifier score                                 |
| Sentiment Incongruity     | sentence-level sentiment variance, sentiment range, sign flip                          |
| Context Mismatch          | parent-response sentiment gap, polarity flip, TF-IDF cosine similarity                 |
| Emoji Features            | emoji count, sarcastic emoji flags, sentiment mismatch                                 |
| Twitter-Specific Features | hashtags, mentions, URLs, retweets, sarcasm hashtags                                   |
| Hyperbole & Irony         | superlatives, exaggeration markers, irony openers, contrast words                      |

## Text Representation

The project uses two types of representations:

1. **TF-IDF features**

   * Unigrams and bigrams
   * Maximum 8000 features
   * Twitter-aware preprocessing for mentions, hashtags, and URLs

2. **Handcrafted features**

   * 55 engineered linguistic and contextual features

These are combined into a final feature matrix for traditional ML models.

## Models Used

The following models were trained and compared:

* Logistic Regression
* XGBoost
* RoBERTa fine-tuned on parent + response pairs
* DistilBERT fine-tuned on parent + response pairs

## Model Performance

| Model                | Accuracy | F1 Score | Precision | Recall |
| -------------------- | -------: | -------: | --------: | -----: |
| Logistic Regression  |   68.78% |   70.76% |    66.54% | 75.56% |
| XGBoost              |   69.00% |   70.79% |    66.93% | 75.11% |
| RoBERTa + Context    |   75.83% |   77.79% |    71.95% | 84.67% |
| DistilBERT + Context |   72.78% |   75.81% |    68.21% | 85.33% |

RoBERTa achieved the best overall F1 score, while XGBoost was useful for interpreting handcrafted feature importance.

## Ablation Study

An ablation study was performed to measure the contribution of different feature groups.

| Feature Variant    | Accuracy | F1 Score |
| ------------------ | -------: | -------: |
| Baseline           |   58.61% |   58.22% |
| + Casing           |   60.50% |   61.34% |
| + Context          |   61.22% |   61.77% |
| + Twitter Features |   63.56% |   65.40% |
| All Features       |   66.28% |   68.04% |

The results show that Twitter-specific features, context mismatch, and casing features contributed strongly to sarcasm detection.

## Explainability

The project includes explainability using:

* SHAP feature importance
* SHAP beeswarm plots
* Local SHAP explanation for individual predictions
* Rule-based human-readable explanations

The explanation layer identifies cues such as:

* sentiment incongruity
* polarity gap
* intense punctuation
* casing patterns
* context mismatch
* sarcastic emojis
* irony and hyperbole markers

## Visualizations

The notebook includes:

* Label distribution
* Response length distribution
* Emoji and punctuation analysis
* Model comparison chart
* F1-score comparison
* Ablation study chart
* Radar chart
* SHAP feature importance
* SHAP beeswarm plot
* Confusion matrices
* Feature correlation heatmap
* Top discriminating words and bigrams
* Word clouds for sarcastic and non-sarcastic tweets
* Interactive Plotly ablation chart

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* NLTK
* VADER Sentiment Analyzer
* Transformers
* PyTorch
* SHAP
* Matplotlib
* Seaborn
* Plotly
* WordCloud
* Jupyter Notebook

## Project Structure

```text
Sarcasm-Detection/
│
├── Sarcasm_detection_in_text.ipynb
└── README.md
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/NagaJahnaviD/Sarcasam-Detection.git
cd Sarcasam-Detection
```

2. Install the required libraries:

```bash
pip install vaderSentiment shap transformers datasets scikit-learn xgboost plotly seaborn wordcloud ipywidgets emoji tqdm requests
```

3. Open the notebook:

```bash
jupyter notebook Sarcasm_detection_in_text.ipynb
```

4. Run all cells.

For RoBERTa and DistilBERT training, using a GPU runtime is recommended.

## Key Takeaways

* Sarcasm detection improves when conversational context is considered.
* Handcrafted linguistic features help explain why a text may be sarcastic.
* Transformer models perform better overall, especially RoBERTa with parent-response context.
* XGBoost provides a good interpretable baseline using engineered features.
* Features like sentiment mismatch, punctuation, casing, Twitter-specific markers, and irony cues are useful for sarcasm classification.

