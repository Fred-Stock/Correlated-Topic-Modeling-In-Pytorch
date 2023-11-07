# Description: This file is used to test the LDA model.

import pandas as pd
from data_preprocessing import pre_processing
from gensim.models import CoherenceModel# spaCy for preprocessing
from gensim.models import ldamodel
from pprint import pprint
import pyLDAvis
import pyLDAvis.gensim
from visualizing_wordcloud import visualize_wordcloud

def model_training(topic_num, corpus, id2word):
    """
    This function trains the LDA model.
    """
    print('Starting model training')
    lda_model = ldamodel.LdaModel(corpus = corpus,
                                id2word = id2word,
                                num_topics = topic_num, # How many topics we want the model to generate
                                random_state = 100,
                                update_every = 1, # Number of documents passed for probability update
                                chunksize = 100, # Number of documents to be used in a training chunk
                                passes = 20, # Number of epochs
                                per_word_topics = True # Compute list of topics, most likely topic for each word
                                )
    
    return lda_model


def performance_metrics(model, corpus, texts, id2word, coherence_type = 'u_mass'):
    print('Starting performance metrics')
    # Compute Perplexity
    perplexity = lda_model.log_perplexity(corpus)

    # Compute Coherence Score
    # calculate coherence of the model using a pipeline
    # segmentation, probability estimation, confirmation measure, aggregation
    coherence_model_lda = CoherenceModel(model = model,
                                        texts = texts,
                                        dictionary = id2word,
                                        coherence = coherence_type)
    coherence_lda = coherence_model_lda.get_coherence()
    
    return perplexity, coherence_lda

if __name__ == '__main__':
    
    news_data = pd.read_csv("newsgroups_data.csv")
    print("Shape of dataset:", news_data.shape)

    news_data = news_data.drop(columns=["Unnamed: 0"])

    documents = news_data.content
    target_labels = news_data.target
    target_names = news_data.target_names

    id2word, texts, corpus = pre_processing(documents, model_type = 'lda')
    
    print(f'Printing human understandable representation of bag-of-words\n \
          {[[(id2word[id], freq) for id, freq in text] for text in corpus[:1]]}'
          )
    
    num_topics = 10
    lda_model = model_training(num_topics, corpus, id2word)
    
    doc_topics_dist = lda_model.get_document_topics(bow = corpus[:5])

    print('Printing the topics and their probabilities for the first 5 documents')
    for idx, vals in zip(range(1, 6), doc_topics_dist):
        print(f"Document {idx+1} - (Topic Number, probability or contribution) are {vals}")
        
    print('Printing the topics and their word composition')
    pprint(lda_model.print_topics())
    
    perplexity, coherence = performance_metrics(lda_model, corpus, texts, id2word, coherence_type = 'u_mass')
    
    print(f'Perplexity: {perplexity}')
    print(f'Coherence Score: {coherence}')
    
    visualize_wordcloud(lda_model)
    
    
    
    