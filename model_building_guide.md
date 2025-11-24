# Market Sentiment ML Model - Implementation Guide

## Overview
This guide explains the step-by-step process to build a model predicting S&P 500 movements based on Trump tweets.

## Model Architecture Decision

### Problem Formulation
**Binary Classification**: Predict if market will go UP (1) or DOWN (0) in the next 1-3 days

**Why Classification over Regression?**
- Stock returns are extremely noisy (many factors beyond tweets)
- Direction (up/down) is more actionable than exact percentage
- Easier to evaluate performance (accuracy, precision, recall)
- Less sensitive to outliers

### Models to Build

1. **Baseline: Sentiment-based Logistic Regression**
   - Extract sentiment scores from tweets (VADER for social media text)
   - Simple, interpretable baseline
   - Fast to train and evaluate

2. **Advanced: TF-IDF + XGBoost**
   - Bag-of-words representation of tweets
   - Captures specific keywords that may correlate with market movements
   - Handles non-linear patterns better

3. **Optional: Transformer-based (FinBERT)**
   - Pre-trained on financial text
   - Best for capturing nuanced sentiment
   - Requires more computational resources

## Implementation Steps

### Step 1: EDA (Exploratory Data Analysis)
- Analyze distribution of returns
- Check class balance (up vs down days)
- Identify if we need stratification or class weighting

### Step 2: Text Preprocessing
- Remove URLs, mentions, special characters
- Lowercase normalization
- Keep exclamation marks (signal emotion)

### Step 3: Create Target Variables
- Binary labels: 1 if return > 0, else 0
- Create for 1-day, 2-day, 3-day returns

### Step 4: Train/Validation/Test Split
- **CRITICAL**: Use time-based split (not random!)
- Train: First 60% of time period
- Validation: Next 20%
- Test: Final 20%
- Prevents data leakage (can't use future to predict past)

### Step 5: Feature Engineering
- Sentiment scores (compound, positive, negative, neutral)
- Tweet length
- Number of capital letters (indicates shouting)
- Presence of financial keywords

### Step 6: Model Training
- Train baseline and advanced models
- Tune hyperparameters on validation set
- Evaluate on test set

### Step 7: Model Evaluation
- Accuracy
- Precision/Recall (especially important if classes are imbalanced)
- F1 Score
- Confusion Matrix
- ROC-AUC

## Expected Challenges

1. **Class Imbalance**: Markets tend to go up more than down (bull market period)
   - Solution: Use class weights or stratified sampling

2. **Weak Signal**: Tweets may have minimal predictive power
   - Solution: Set realistic expectations, focus on statistical significance

3. **Temporal Dependency**: Market has momentum/trends
   - Solution: Could add previous day's return as feature (advanced)

## Success Criteria

- **Beat Random Baseline**: >50% accuracy on balanced classes
- **Beat Market Baseline**: Better than always predicting "UP"
- **Statistical Significance**: Precision/Recall improvements are meaningful

## Next Steps After Model Building

1. Feature importance analysis - which words matter most?
2. Error analysis - which tweets are hardest to predict?
3. Consider ensemble methods combining multiple models
