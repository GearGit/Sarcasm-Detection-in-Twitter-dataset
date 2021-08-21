import keras
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os

max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')


def lstm_headline(sentence):
    sentence = tokenizer.texts_to_sequences(sentence)
    sentence = pad_sequences(sentence, maxlen=29, dtype='int32', value=0)

    loaded_model = keras.models.load_model("models/lstm-headline")

    sentiment = loaded_model.predict(sentence, batch_size=1, verbose=2)[0]
    print(sentiment)
    os.system('cls')
    if np.argmax(sentiment) == 0:
        return 'Non-Sarcastic', sentiment
    elif np.argmax(sentiment) == 1:
        return 'Sarcastic', sentiment


def lstm_tweets(sentence):
    sentence = tokenizer.texts_to_sequences(sentence)
    sentence = pad_sequences(sentence, maxlen=29, dtype='int32', value=0)

    loaded_model = keras.models.load_model("models/lstm-tweets")

    sentiment = loaded_model.predict(sentence, batch_size=1, verbose=2)[0]
    os.system('cls')
    if np.argmax(sentiment) == 0:
        return 'Non-Sarcastic', sentiment
    elif np.argmax(sentiment) == 1:
        return 'Sarcastic', sentiment
