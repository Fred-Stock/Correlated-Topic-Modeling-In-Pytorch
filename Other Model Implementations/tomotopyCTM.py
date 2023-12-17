import pandas as pd
import regex as re
import spacy
import nltk
import tomotopy as tp
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import logging
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from typing import List

logger = logging.getLogger(__name__)

# Define the preprocessing functions as done in data_preprocessing
def basic_processing(text: List[str]) -> List[str]:
    logger.info("Starting pre-processing of the documents")
    text = [sent.replace("From:", "") for sent in text]
    text = [sent.replace("Organization:", "") for sent in text]
    text = [sent.replace("Subject:", "") for sent in text]
    text = [sent.replace("Summary:", "") for sent in text]
    text = [sent.replace("Keywords:", "") for sent in text]
    text = [sent.replace("Distribution:", "") for sent in text]
    text = [sent.replace("Lines:", "") for sent in text]
    text = [sent.replace("Nntp-Posting-Host:", "") for sent in text]

    # Remove the email id (@)
    text = [re.sub(r"\S*@\S*\s?", " ", sent) for sent in text]

    # Remove any new line characters
    text = [re.sub(r"\s+", " ", sent) for sent in text]

    # Remove single quote marks
    text = [re.sub(r"\'", " ", sent) for sent in text]

    # Remove digits
    text = [re.sub(r"\d", " ", sent) for sent in text]

    return text


def sent_to_words(sentences: List[str]) -> List[List[str]]:
    logger.info("Starting tokenization of the documents")
    for sentence in sentences:
        yield (
            simple_preprocess(str(sentence), deacc=True)
        )  # deacc=True removes punctuations


def remove_stopwords(texts: List[str],
                     stop_words: List[str]) -> List[List[str]]:
    logger.info("Starting removal of stop words")
    return [
        [word for word in simple_preprocess(str(doc)) if word not in stop_words]
        for doc in texts
    ]

def lemmatization(texts: List[str], nlp, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]) -> List[List[str]]:
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def remove_words_less_than_length_3(texts: List[List[str]]) -> List[List[str]]:
    logger.info("Starting removal of words with length less than 3")
    for document in texts:
        for word in document:
            if len(word) < 3:
                document.remove(word)
    return texts
    
def nltk_initialization() -> None:
    nltk.download("stopwords")

# Similar to preprocessing used in our CTM Model implementation
def pre_processing(documents):
    if not stopwords:
        nltk_initialization()

    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    processed_data = basic_processing(documents)
    processed_data = list(sent_to_words(processed_data))
    processed_data = remove_stopwords(processed_data, stop_words)
    processed_data = lemmatization(processed_data, nlp)
    processed_data = remove_words_less_than_length_3(processed_data)

    return processed_data


def runModel(data, num_doc, num_topics):
    # Run preprocessing and init model
    documents = data.head(num_doc).content
    tokenized_data = pre_processing(documents)
    model = tp.CTModel(k=num_topics, min_cf=3)

    for doc in tokenized_data:
        model.add_doc(doc)

    for i in range(0, 100, 10):
        model.train(10)

    # Extract topics and dist
    topic_distributions = np.array([doc.get_topic_dist() for doc in model.docs])

    # Extract top 5 topics
    for k in range(model.k):
        print('Topic #{}'.format(k), model.get_topic_words(k, top_n=5))

    return topic_distributions, model.get_topic_words(k, top_n=5)

# Prints Correlation Matrix
def plot_correlation_matrix(topic_distributions, topic_words):
    corr_matrix = np.corrcoef(topic_distributions, rowvar=False)
    topic_labels = [words[0] for words in topic_words]

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                xticklabels=topic_labels, yticklabels=topic_labels)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    news_data = pd.read_csv("../newsgroups_data.csv")
    news_data = news_data.drop(columns=["Unnamed: 0"])

    topic_distributions, topic_words = runModel(news_data, 32, 5)
    plot_correlation_matrix(topic_distributions, topic_words)