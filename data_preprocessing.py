import regex as re
import spacy

# NLTK packages
import nltk
from nltk.corpus import stopwords

from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import CountVectorizer

# Packages to make life easier
from typing import List
import logging

logger = logging.getLogger(__name__)


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


def lemmatization(
    texts: List[str], nlp,
    allowed_postags: List[str] = ["NOUN", "ADJ", "VERB", "ADV"]
) -> List[List[str]]:
    logger.info("Starting lemmatization of the documents")
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            " ".join(
                [
                    token.lemma_ if token.lemma_ not in ["-PRON-"] else ""
                    for token in doc
                    if token.pos_ in allowed_postags
                ]
            )
        )
    return texts_out


def remove_words_less_than_length_3(text: List[List[str]]) -> List[List[str]]:
    logger.info("Starting removal of words with length less than 3")
    for document in text:
        for word in document:
            if len(word) < 3:
                document.remove(word)
    return text


def pre_processing(documents) -> List[List[str]]:
    if stopwords:
        stop_words = stopwords.words("english")
        stop_words.extend(["from", "subject", "re", "edu", "use"])
    else:
        nltk_initialization()
        stop_words = stopwords.words("english")
        stop_words.extend(["from", "subject", "re", "edu", "use"])

    # Load the spacy (en_core_web_sm): small English pipeline
    # trained on written web text (blogs, news, comments), that includes
    # vocabulary, syntax and entities.
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    logger.info("Pre-processing the documents")
    # Pre-process the documents
    processed_data = basic_processing(documents)

    # Tokenization
    processed_data = list(sent_to_words(processed_data))

    # Remove the stop words
    processed_data = remove_stopwords(processed_data, stop_words)

    # Lemmatization
    processed_data = lemmatization(
        processed_data, nlp, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]
    )

    # Implement CountVectorizer
    vectorizer = CountVectorizer(
        analyzer="word",  # tokenizes the documents
        min_df=3,  # minimum number of documents a token must appear in
        token_pattern="[a-zA-Z0-9]{3,}",  # num chars > 3
        max_features=20000,  # max number of unique words
    )

    X = vectorizer.fit_transform(processed_data)
    print(f"Printing vocabulary:\n{vectorizer.get_feature_names_out()[:10]}")
    print("Shape of the bag-of-words model:", X.shape)

    vocab_size = X.shape[1]

    return X.toarray(), vocab_size


def nltk_initialization() -> None:
    nltk.download("stopwords")
