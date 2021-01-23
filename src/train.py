import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import config

def run(fold, df):
    train_df = df[df["kfold"] != fold]
    val_df = df[df["kfold"] == fold]
    
#    print(val_df)

    training_sentences = train_df["tweet"].values
    testing_sentences = val_df["tweet"].values
    
    tokenizer = Tokenizer(num_words = config.vocab_size, oov_token = config.oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded = pad_sequences(sequences, maxlen = config.max_length, truncating = config.trunc_type)
    
    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen = config.max_length, truncating = config.trunc_type)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(config.vocab_size, config.embedding_dim, input_length = config.max_length),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])
    
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.summary()
    
    num_epochs = 10
    model.fit(padded,
              train_df["label"].values,
              epochs = num_epochs,
              validation_data = (testing_padded, val_df["label"].values),
              verbose = 1)

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(config.INPUT_PATH, r'train_folds.csv'))

    run(0, df)
    run(1, df)
    run(2, df)
    run(3, df)
    run(4, df)