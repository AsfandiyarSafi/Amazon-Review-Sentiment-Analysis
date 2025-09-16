
# 📘 Sentiment Elicitation: Exploratory Data Analysis and Sentiment Classification on Amazon Reviews

## 📝 Project Overview

This project performs **sentiment analysis** on Amazon product reviews using **exploratory data analysis (EDA)** and multiple **machine learning models**. The aim is to classify reviews into **positive**, **neutral**, or **negative** sentiments using both traditional and transformer-based deep learning approaches.

---

## 📂 Dataset

- **Source**: Amazon reviews dataset (`Reviews.csv`)
- **Features Used**: Review text, score, timestamp
- **Label Mapping**:
  - Scores ≥ 4 → **Positive**
  - Score = 3 → **Neutral**
  - Scores ≤ 2 → **Negative**

---

## 📊 Exploratory Data Analysis (EDA)

- Visualized sentiment distribution (positive, neutral, negative)
- Examined sentiment trends over time with log-scaled plots
- Derived initial insights on customer satisfaction and patterns

---

##  Models Implemented

### 1.  RoBERTa in PyTorch
- Used `RobertaForSequenceClassification` from Hugging Face Transformers
- Custom dataset class for tokenization and batch processing
- Trained on a small dataset subset
- Evaluated using classification accuracy and standard metrics

### 2.  CNN with GloVe Embeddings
- Text preprocessing and vocabulary building
- Word representation with **pre-trained GloVe embeddings**
- CNN architecture using convolutional and pooling layers
- Balanced training dataset
- Evaluation using confusion matrix and classification report

### 3.  DistilBERT with Hugging Face Transformers

#### Initial Implementation
- Fine-tuned on a 5,000-row dataset
- Tokenization using Hugging Face Datasets
- Trainer API used for training and evaluation
- Metrics: Accuracy, Precision, Recall, F1-score
- Visuals: Confusion matrix and per-class metric plots
- Macro vs Weighted average score comparisons

#### Final Implementation with Class Balancing & Weighted Training
- Improved preprocessing and data tokenization
- Oversampling of minority classes to balance training data
- Custom Trainer subclass implementing weighted loss
- Extended training schedule and learning rate management
- Comprehensive evaluation with:
  - Raw and normalized confusion matrices
  - Classification reports
  - Per-class F1 score visualizations
- Final model, tokenizer, and label encoder saved for inference
- Sample predictions on new review texts included


- Final Model Performance Scores
    - Overall Accuracy: 0.9552
    - Weighted F1 Score: 0.9553
    - Class-wise F1 Scores:
        - Negative: 0.9663
        - Neutral: 0.9399
        - Positive: 0.9575
---

##  Evaluation and Visualization

- Confusion matrices (raw and normalized)
- Classification reports (Precision, Recall, F1-score)
- ROC curves for multi-class classification
- Bar charts comparing macro vs. weighted averages
- Per-class performance metrics

---

##  Dependencies and Setup

**Python Libraries**:
```bash
pip install pandas numpy matplotlib seaborn nltk torch transformers datasets scikit-learn tqdm imbalanced-learn
```

**Additional Setup**:
- Download GloVe embeddings (if not already present)
- NLTK stopwords/tokenizers required
- GPU acceleration is supported if available

---

##  Usage Instructions

- Run the notebook cells sequentially for EDA, model training, and evaluation
- Modify dataset paths if necessary
- Use saved models and label encoders to make predictions on new data

---

## Summary

This project highlights the effectiveness of combining traditional NLP models like **CNN with GloVe** and modern **transformer-based architectures** such as **RoBERTa** and **DistilBERT**.

- RoBERTa offers strong baseline transformer performance.
- CNN delivers meaningful results through classical NLP techniques.
- DistilBERT's enhanced model with class balancing and weighted training demonstrates robust performance and generalization capability.

All stages, from data exploration to advanced model tuning and evaluation, are thoroughly covered and reproducible.


## Note on Results Discrepancy

During the initial phase, we encountered technical challenges that prevented us from completing all planned experiments before the report deadline. Rather than delay the project, we proceeded with our best available analysis while prioritizing a proper implementation.

The current code reflects this completed work, including the primary fine-tuned model that proved most effective. While the report discusses two model variants for comparison, we’ve focused the release on the more stable implementation for clarity. The methodology remains consistent throughout, and we’ve ensured the final code properly reflects our intended approach.
