import logging
import pandas as pd
from typing import List
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from data_preprocessing import basic_processing, sent_to_words, remove_stopwords, lemmatization, remove_words_less_than_length_3

logger = logging.getLogger(__name__)

def pre_processing(documents) -> List[List[str]]:
    
    # Initialize the NLTK functionalities: [stopwords, wordnet, punkt, averaged_perceptron_tagger]
    if stopwords:
        stop_words = stopwords.words('english')
        stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    else:
        nltk_initialization()
        stop_words = stopwords.words('english')
        stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
        
    # Load the spacy (en_core_web_sm): small English pipeline trained on written web text (blogs, news, comments), 
    # that includes vocabulary, syntax and entities.
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    
    logger.info('Pre-processing the documents')
    # Pre-process the documents
    processed_data = basic_processing(documents)
    
    # Tokenization
    processed_data = list(sent_to_words(processed_data))
    
    # Remove the stop words
    processed_data = remove_stopwords(processed_data, stop_words)
    # print('\nExample document after removing stop words:\n', processed_data[:1])
    
    # Lemmatization
    processed_data = lemmatization(processed_data, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    # print('\nExample document after lemmatization:\n', processed_data[:1])
    
    # Remove words less than length 3
    # processed_data = remove_words_less_than_length_3(processed_data)
    # print('\nExample document after removing words less than length 3:\n', processed_data[:1])
    
    # Implement CountVectorizer
    vectorizer = CountVectorizer(analyzer = 'word', # tokenizes the documents
                                 min_df = 3, # minimum number of documents a token must appear in
                                 token_pattern = '[a-zA-Z0-9]{3,}', # num chars > 3
                                 max_features = 50000 # max number of unique words
                                 )
    
    X = vectorizer.fit_transform(processed_data)
    print('Printing some vocabulary:', vectorizer.get_feature_names_out()[:10])
    print('Shape of the bag-of-words model:', X.shape)
    
    vocab_size = X.shape[1]
    
    return X.toarray(), vocab_size

def nltk_initialization() -> None:
    nltk.download('stopwords')

if __name__ == '__main__':
    news_data = pd.read_csv('newsgroups_data.csv')
    print('Shape of dataset:', news_data.shape)
    
    news_data = news_data.drop(columns=['Unnamed: 0'])
    
    documents = news_data.content
    target_labels = news_data.target
    target_names = news_data.target_names
    
    final_data = pre_processing(documents)