import re
import numpy as np
import tensorflow as tf

from langdetect import detect_langs

# check if an item in dataframe is written in english with probability above a certain treshold
def is_foreign(item,treshold=0.7):
    
    languages = detect_langs(item)
    
    for lang in languages:
        if lang.lang == 'en' and lang.prob > treshold:
            return False
    
    return True

# clean the text and use lemmatizer
def clean_text_lemmatize(item,lemmatizer,stopwords):
    
    # remove latex equations
    item = re.sub('\$+.*?\$+','',item)
    
    # tokenize and remove punctuation
    item = re.findall('[a-zA-Z0-9]+',item)
    
    # lowecase everything
    item = [word.lower() for word in item]
    
    # remove english stopwords
    item = [word for word in item if word not in stopwords]
    
    # lemmatize the words
    item = [lemmatizer.lemmatize(word) for word in item]
    
    return item

# Maps the words into the text to the corresponding index in the word2vec model
#
# NOTE: Since we later pad with 0's, and the index 0 is associated with a particular
#       word in our w2v model, we shift the index of every word by 1.
#
def hashing_trick(text,w2v):
    return np.array([w2v.vocab[word].index+1 for word in text if word in w2v.vocab])

# Maps each sentence in dataframe into a padded sequence of integer (pad is done with 0)
def hashed_padded_sequences(df_text,w2v):
    
    hashed_text = df_text.apply(hashing_trick,args=(w2v,))
    
    max_length = hashed_text.apply(len).max()
    hashed_padded_text = hashed_text.apply(lambda arr : np.pad(arr,(0,max_length-arr.size)))
    
    return hashed_padded_text

# Creates embedding layer from w2v model
#
# NOTE: padding 0's are sent to the 0 vector
#
def get_keras_embedding(w2v,masking=False):

    # get size of the vocabulary and embedding dimension
    vocab_size, k = w2v.vectors.shape
    
    # Embedding weights, each row is the vector associated to a specific word in the model
    # NOTE: The 0-th row is mapped into the 0 vector (this is the padding)
    embedding_weigths = np.vstack((np.zeros(k,dtype=np.float),w2v.vectors))

    # Create the Keras layer (it won't be trainable)
    embedding_layer = tf.keras.layers.Embedding(vocab_size+1,
                                                k,
                                                weights=[embedding_weigths],
                                                trainable=False,
                                                mask_zero=masking
                                               )
    
    return embedding_layer