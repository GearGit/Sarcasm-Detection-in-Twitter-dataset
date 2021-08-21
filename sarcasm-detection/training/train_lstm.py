import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
# %matplotlib inline
import os
import re

df = pd.read_json('Sarcasm_Headlines_Dataset.json')
df.head()

sns.countplot(df.is_sarcastic)

plt.xlabel('Label')
plt.title('Sarcasm vs Non-sarcasm')

df['headline'] = df['headline'].apply(lambda x: x.lower())
df['headline'] = df['headline'].apply(
    (lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

for idx, row in df.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_fatures = 2000

tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(df['headline'].values)

X = tokenizer.texts_to_sequences(df['headline'].values)
X = pad_sequences(X)

Y = pd.get_dummies(df['is_sarcastic']).values
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

batch_size = 32
model.fit(X_train, Y_train, epochs=20, batch_size=batch_size, verbose=2)

validation_size = 1500

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]

score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)

print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

headline = ["God, you are the best boss EVER! Have I ever told you how much I love this job? I  wish I could live here! Somebody get me a tent, I never wanna leave!"]

headline = tokenizer.texts_to_sequences(headline)
headline = pad_sequences(headline, maxlen=29, dtype='int32', value=0)

sentiment = model.predict(headline, batch_size=1, verbose=2)[0]
if(np.argmax(sentiment) == 0):
    print("Non-sarcastic")
elif (np.argmax(sentiment) == 1):
    print("Sarcasm")
