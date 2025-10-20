from datasets import load_dataset
# from setfit import get_templated_dataset
# import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os
from wordcloud import WordCloud, STOPWORDS


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
    

if __name__ == "__main__":
    # Load all splits of the dataset
    train_dataset = load_dataset("dair-ai/emotion", "split", split="train")
    val_dataset = load_dataset("dair-ai/emotion", "split", split="validation") 
    test_dataset = load_dataset("dair-ai/emotion", "split", split="test")
    classes = test_dataset.features["label"].names
    # train_dataset = get_templated_dataset(candidate_labels = classes, sample_size=test_dataset.num_rows)

    print(f"{train_dataset[0]=}, {classes=}")

    os.makedirs("class_plots", exist_ok=True)
    plot_classes(train_dataset, classes, "Training", "class_plots/training_countplot.png")
    plot_classes(val_dataset, classes, "Validation", "class_plots/validation_countplot.png")
    plot_classes(test_dataset, classes, "Test", "class_plots/test_countplot.png")
    
    # Plot text count histograms
    os.makedirs("text_length_plots", exist_ok=True)
    plot_text_count_histogram(train_dataset, "Training", "text_length_plots/training_text_count_histogram.png")
    plot_text_count_histogram(val_dataset, "Validation", "text_length_plots/validation_text_count_histogram.png")
    plot_text_count_histogram(test_dataset, "Test", "text_length_plots/test_text_count_histogram.png")

    plot_wordcloud(train_dataset["text"], title="Word Cloud of Emotions")

    text_transformer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), lowercase=True, max_features=10000, min_df=1, max_df=0.9, sublinear_tf=True)

    # Build vocabulary from training data only. New words in test data ignored.
    vectorised_train_dataset = text_transformer.fit_transform(train_dataset["text"])
    vectorised_validate_dataset = text_transformer.transform(val_dataset["text"])
    vectorised_test_dataset = text_transformer.transform(test_dataset["text"])

    print(f"{vectorised_train_dataset.shape=}, {vectorised_validate_dataset.shape=}, {vectorised_test_dataset.shape=}")

    model = LogisticRegression(class_weight='balanced', solver='lbfgs', multi_class="multinomial", penalty = "l2", C = 0.5, max_iter = 2000, random_state = 42, n_jobs=4)