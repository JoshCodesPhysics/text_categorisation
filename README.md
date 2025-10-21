# text_categorisation
Building a model to categorise text snippets

# Set up
- Build a virtual environment with `python3.12 -m venv .venv`, source with `source venv/bin/activate` and install dependencies with `pip install -r requirements.txt`

# Dataset

- Used huggingface dair emotion dataset with text snippets labeled with the associated emotion.
- Plotted label distribution to check imbalance, and used `class_weight = "balanced"` in the logistic regression model, which weights data from a certain class inversely proportional to the count of that class (n_samples / (n_classes * np.bincount(specific_class))) to ensure training is fair across categories.
- Double checked row text length between dataset splits to ensure mean and median text lengths are roughly similar (I think context windows matter for this kind of thing, especially without using a RoPE)

# Tfid Vectorizer
- Word count per row and word frequency across whole dataset converted to vector, combined with vocabulary index vector
- Training, validation and test datasets come pre-stratified and equally balanced from hugging face
- Using english dataset so remove english "stop" (common, non-emotional) words
- Convert everything to lowercase for consistency
- Embed one and two-grams (word chunks). Two grams might be good for adjectivised emotions, like 'bitter despair'.
- Remove words that appear very frequently (and, an, the - irrelevant to emotional context) or very infrequently (typos for example).
- Logarithmically scale weighting of frequently-appearing words that have not been removed so that they do not dominate.
- Number of features and frequency cutoffs are geared towards the smaller dataset that we are working with here

# Logistic Regression Model
- Using the Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm to minimise the model loss (approximate the Hessian with a few gradient evaluations) - good for small datasets and is the default solver for sklearn LR
- Multinomial - a multidimensional surface -> polynomial for each class combined to produce logit values that can be softmaxed into probability-like values (sum to one) for each class. Can use top-k choice or just greedy choice of highest value. The parameters of each polynomial are optimised for the most accurate predictions 
- L2 regularisation for overfitting prevention (sum of squares term that keep most weights small and heavily penalises particularly large weights - generalisation)
- C parameter = 0.5 -> Low level of trust in training dataset. Data row - feature ratio is quite low, need more data if we want more trust. Use aggressive regularisation to prevent overfitting to small dataset.
- Use sufficient training iterations for loss convergence, and use all CPU cores for speed

## Trained and validated, calculated F1 macro score and accuracy overall and per class, and plotted confusion matrix, feature correlations, feature-target correlations. F1 and accuracy 88, 90 %
- Good scores for Joy, Sadness and Anger. Some confusion of joy <-> love, fear <-> surprise, sadness <-> fear.
- Test subset would be used for final model evaluation after hyperparameter tuning and cross-validation checks

## Tested pre-trained BERT model, and then fine-tuned SetFit model for zero-shot classification to compare scores
- Pre trained BERT model performed poorly without finetuning - ~39% accuracy. Perhaps larger or more specialised model would do better.
- Fine-tuned SetFit model -> Comparitive accuracy to TFID and logistic regression approach: 89%
- May improve with model complexity and increasing number of finetuning steps

# Given more time
- Optimise model, increase datasets and evaluate on final test set
- Compare models with uncertainties on their performance metrics in mind (statistical significance)
- Use a random baseline classifier model and combine all into an AUC plot
- Look at precision/recall scores
- Start building out pipeline architecture (containerisation, model persistence)
- Visualise model coefficients with SHAP
- Plot each model metrics in bar chart