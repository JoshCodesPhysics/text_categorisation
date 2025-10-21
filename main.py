from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from transformers import pipeline
from setfit import SetFitModel, Trainer, TrainingArguments


def plot_classes(dataset, classes, plotname, filename):
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=dataset["label"], hue=dataset["label"], palette="Set2", legend=False)
    plt.title(f'{plotname} Distribution')
    plt.xlabel('Emotion Categories')
    plt.ylabel('Count')

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_text_count_histogram(dataset, plotname, filename):
    
    text_lengths = [len(text.split()) for text in dataset["text"]]
    
    plt.figure(figsize=(10, 6))
    plt.hist(text_lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(f'{plotname} Text Length Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    mean_length = np.mean(text_lengths)
    median_length = np.median(text_lengths)
    plt.axvline(mean_length, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_length:.1f}')
    plt.axvline(median_length, color='orange', linestyle='--', alpha=0.7, label=f'Median: {median_length:.1f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# From: https://www.kaggle.com/aashita/word-clouds-of-various-shapes
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 
                   title = None, title_size=40):
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='black',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=800, 
                    height=400,
                    mask = mask)

    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    plt.imshow(wordcloud);
    plt.title(title, fontdict={'size': title_size, 'color': 'black',
    'verticalalignment': 'bottom'})

    plt.axis('off');
    plt.tight_layout()
    os.makedirs("wordcloud_plot", exist_ok=True)
    plt.savefig("wordcloud_plot/wordcloud.png")
    plt.close()

def get_conf_matrix_and_class_report(val_dataset, validation_predictions, classes):
    cm = confusion_matrix(val_dataset["label"], validation_predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Emotion Classification')
    plt.xlabel('Predicted Emotion')
    plt.ylabel('Actual Emotion')
    plt.tight_layout()
    os.makedirs("confusion_matrix", exist_ok=True)
    plt.savefig("confusion_matrix/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Classification Report:")
    print(classification_report(val_dataset["label"], validation_predictions, target_names=classes))

def plot_correlation_matrix(correlation_matrix, sample_size, feature_names):
    plt.figure(figsize=(20, 16))  # Larger figure for readability
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask upper triangle
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                xticklabels=feature_names, yticklabels=feature_names)
    
    plt.title(f'Feature Correlation Matrix (Top {sample_size} Features)', fontsize=16)
    plt.xlabel('Features (Words/Bigrams)', fontsize=12)
    plt.ylabel('Features (Words/Bigrams)', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    os.makedirs("correlation_plot", exist_ok = True)
    plt.savefig("correlation_plot/feature_correlations.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_emotion_correlation(feature_emotion_coef, feature_names, emotion_classes):
    plt.figure(figsize=(12, 16))
    
    sns.heatmap(feature_emotion_coef, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=False, linewidths=0.5, cbar_kws={"shrink": .8},
                xticklabels=emotion_classes, yticklabels=feature_names)
    
    plt.title('Feature-to-Emotion Correlations (Logistic Regression Coefficients)', fontsize=14)
    plt.xlabel('Emotion Classes', fontsize=12)
    plt.ylabel('Features (Words/Bigrams)', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    os.makedirs("correlation_plot", exist_ok = True)
    plt.savefig("correlation_plot/feature_emotion_correlations.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    train_dataset = load_dataset("dair-ai/emotion", "split", split="train")
    val_dataset = load_dataset("dair-ai/emotion", "split", split="validation") 
    test_dataset = load_dataset("dair-ai/emotion", "split", split="test")
    classes = test_dataset.features["label"].names
    # train_dataset = get_templated_dataset(candidate_labels = classes, sample_size=test_dataset.num_rows)

    print(f"{train_dataset[0]=}, {classes=}")

    # Some EDA
    os.makedirs("class_plots", exist_ok=True)
    plot_classes(train_dataset, classes, "Training", "class_plots/training_countplot.png")
    plot_classes(val_dataset, classes, "Validation", "class_plots/validation_countplot.png")
    plot_classes(test_dataset, classes, "Test", "class_plots/test_countplot.png")
    
    os.makedirs("text_length_plots", exist_ok=True)
    plot_text_count_histogram(train_dataset, "Training", "text_length_plots/training_text_count_histogram.png")
    plot_text_count_histogram(val_dataset, "Validation", "text_length_plots/validation_text_count_histogram.png")
    plot_text_count_histogram(test_dataset, "Test", "text_length_plots/test_text_count_histogram.png")

    plot_wordcloud(train_dataset["text"], title="Word Cloud of Emotions")

    # ----------------- TFID vectorizer + logits approach ------------------

    text_transformer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), lowercase=True, max_features=10000, min_df=1, max_df=0.9, sublinear_tf=True)

    # Build vocabulary from training data only. New words in test data ignored.
    vectorised_train_dataset = text_transformer.fit_transform(train_dataset["text"])
    vectorised_validate_dataset = text_transformer.transform(val_dataset["text"])
    vectorised_test_dataset = text_transformer.transform(test_dataset["text"])

    print(f"{vectorised_train_dataset.shape=}, {vectorised_validate_dataset.shape=}, {vectorised_test_dataset.shape=}")
    model = LogisticRegression(class_weight='balanced', solver='lbfgs', penalty = "l2", C = 0.5, max_iter = 2000, random_state = 42, n_jobs=4)

    model.fit(vectorised_train_dataset, train_dataset["label"])

    sample_size = 30

    feature_importance = np.abs(model.coef_).mean(axis=0)
    top_features_idx = np.argsort(feature_importance)[-sample_size:]

    feature_matrix = vectorised_train_dataset[:, top_features_idx].toarray()
    correlation_matrix = np.corrcoef(feature_matrix.T)
    
    feature_names = text_transformer.get_feature_names_out()
    top_feature_names = [feature_names[i] for i in top_features_idx]

    feature_emotion_coef = model.coef_[:, top_features_idx].T

    plot_correlation_matrix(correlation_matrix, sample_size, top_feature_names)
    plot_feature_emotion_correlation(feature_emotion_coef, top_feature_names, classes)

    validation_predictions = model.predict(vectorised_validate_dataset)
    f1 = f1_score(validation_predictions, val_dataset["label"], average = "macro")
    accuracy = accuracy_score(validation_predictions, val_dataset["label"])

    print(f"TFID + Logits metrics: {f1=}, {accuracy=}")
    get_conf_matrix_and_class_report(val_dataset, validation_predictions, classes)

    # -------------- Using pretrained BART model for zero shot classification to compare to simple TFID + Logit -----------

    pipe = pipeline("zero-shot-classification", model = "facebook/bart-large-mnli", device=0)

    zeroshot_predictions = pipe(list(val_dataset["text"]), batch_size=16, candidate_labels=classes)
    # Get largest value from prediction set
    # pipe returns dictionary containing each row text sequence under sequence: ..., labels: [sadness, surprise]..., scores: [0.94..., ...]
    # print("Label, text: ", [f"{zeroshot_predictions[i]['labels'][0]}: {val_dataset['text'][i]}" for i in range(len(zeroshot_predictions))][:10])
    zeroshot_predictions = [classes.index(prediction["labels"][0]) for prediction in zeroshot_predictions]
    # print(f"{zeroshot_predictions=}, {val_dataset['label']=}")

    zeroshot_f1 = f1_score(zeroshot_predictions, val_dataset["label"], average = "macro")
    zeroshot_accuracy = accuracy_score(zeroshot_predictions, val_dataset["label"])

    print(f"Pretrained (not finetuned) metrics: {zeroshot_f1=}, {zeroshot_accuracy=}")
    
    # -------------- Finetuning 'small' pretrained LLM for zero shot classification to compare to BART and simple TFID + Logit -----------
    
    # Create smaller datasets since training takes a very long time
    small_train_size = len(train_dataset) // 10  
    small_val_size = len(val_dataset) // 10
    
    small_train_dataset = train_dataset.select(range(small_train_size))
    small_val_dataset = val_dataset.select(range(small_val_size))
    
    model = SetFitModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        use_differentiable_head=True,
        head_params={"out_features": len(classes)}
    )

    args = TrainingArguments(
        batch_size=16,
        num_epochs=1,
        # num_iterations=1000,
        max_steps=10000, 
    )

    trainer = Trainer(
        model=model,
        train_dataset=small_train_dataset,
        eval_dataset=small_val_dataset,
        args=args
    )

    trainer.train()
    zeroshot_metrics = trainer.evaluate()
    print(f"Finetuned model metrics: {zeroshot_metrics}")

    model.save_pretrained("finetuned_all_mini_lm_l6_v2")
    # model = SetFitModel.from_pretrained("finetuned_all_mini_lm_l6_v2")
