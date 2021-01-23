import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import config

train_df = pd.read_csv(os.path.join(config.INPUT_PATH, 'train.csv'))
test_df = pd.read_csv(os.path.join(config.INPUT_PATH, 'test.csv'))

#    print(val_df)

training_sentences = train_df["tweet"].values
testing_sentences = test_df["tweet"].values

tokenizer = Tokenizer(num_words = config.vocab_size, oov_token = config.oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen = config.max_length, truncating = config.trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen = config.max_length, truncating = config.trunc_type)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(config.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences = True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

num_epochs = 3
model.fit(padded,
          train_df["label"].values,
          epochs = num_epochs,
          verbose = 1)

labels = model.predict(testing_padded)
labels = [0 if label<0.5 else 1 for label in labels]

print("Output :")
print(labels)

test_df["label"] = labels

test_df.to_csv(os.path.join(config.INPUT_PATH, "predictions.csv"), columns = ["id", "label"], index = False)
