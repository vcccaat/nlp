# CNN with adam
import numpy as np
import string
import pandas as pd
import nltk
import keras

from sklearn import random_projection
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords

from keras.models import Model
from keras.layers import Embedding, Dense, Dropout, Conv2D, MaxPool2D, Concatenate, Input, Reshape, Flatten
from keras.optimizers import SGD
from keras import metrics

stop_words = set(stopwords.words('english') + list(string.punctuation))


# -------------- Helper Functions --------------
def tokenize(text):
    tokens = []
    for word in nltk.word_tokenize(text):
        word = word.lower()
        if word not in stop_words and not word.isnumeric():
            if re.search('[a-zA-Z]',word):
                tokens.append(stemmer.stem(word))
    return tokens


def get_sequence(data, seq_length, vocab_dict):
    '''
    :param data: a list of words, type: list
    :param seq_length: the length of sequences,, type: int
    :param vocab_dict: a dict from words to indices, type: dict
    return a dense sequence matrix whose elements are indices of words,
    '''
    data_matrix = np.zeros((len(data), seq_length), dtype=int)
    for i, doc in enumerate(data):
        for j, word in enumerate(doc):
            if j == seq_length:
                break
            word_idx = vocab_dict.get(word, 1) # 1 means the unknown word
            data_matrix[i, j] = word_idx
    return data_matrix


def read_data(file_name, input_length, vocab=None):
    """
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    """
    df = pd.read_csv(file_name)
    df['words'] = df['text'].apply(tokenize)

    if vocab is None:
        vocab = set()
        for i in range(len(df)):
            for word in df.iloc[i]['words']:
                vocab.add(word)
    vocab_dict = dict()
    vocab_dict['<pad>'] = 0 # 0 means the padding signal
    vocab_dict['<unk>'] = 1 # 1 means the unknown word
    vocab_size = 2
    for v in vocab:
        vocab_dict[v] = vocab_size
        vocab_size += 1

    data_matrix = get_sequence(df['words'], input_length, vocab_dict)

    return df['user_id'], df['stars'].apply(int)-1, data_matrix, vocab
# ----------------- End of Helper Functions-----------------


def load_data(input_length):
    # Load training data and vocab
    train_id_list, train_data_label, train_data_matrix, vocab = read_data("drive/My Drive/train.csv", input_length)
    K = max(train_data_label)+1  # labels begin with 0

    # Load testing data
    test_id_list, _, test_data_matrix, _ = read_data("drive/My Drive/valid.csv", input_length, vocab=vocab)
    test_data_label = pd.read_csv("drive/My Drive/valid.csv")['stars'] - 1
    
    print("Vocabulary Size:", len(vocab))
    print("Training Set Size:", len(train_id_list))
    print("Test Set Size:", len(test_id_list))
    print("Training Set Shape:", train_data_matrix.shape)
    print("Testing Set Shape:", test_data_matrix.shape)

    # Converts a class vector to binary class matrix.
    train_data_label = keras.utils.to_categorical(train_data_label, num_classes=K)
    test_data_label = keras.utils.to_categorical(test_data_label, num_classes=K)
    return train_data_matrix, train_data_label, test_data_matrix, test_data_label, vocab


if __name__ == '__main__':
    # Hyperparameters
    input_length = 300
    embedding_size = 100
    hidden_size = 100
    batch_size = 100
    dropout_rate = 0.5
    filters = 100
    kernel_sizes = [3, 4, 5]
    padding = 'valid'
    activation = 'relu'
    strides = 1
    pool_size = 2
    learning_rate = 0.1
    total_epoch = 10

    train_data_matrix, train_data_label, test_data_matrix, test_data_label, vocab = load_data(input_length)

    # Data shape
    N = train_data_matrix.shape[0]
    K = train_data_label.shape[1]

    input_size = len(vocab) + 2
    output_size = K

    # New model
    x = Input(shape=(input_length, ))
    # print('x',x)

    # embedding layer and dropout
    # YOUR CODE HERE
    e = Embedding(input_dim=input_size, output_dim=embedding_size, input_length=input_length)(x)
    e_d = Dropout(dropout_rate)(e)
    # print('e_d',e_d)

    # construct the sequence tensor for CNN
    # YOUR CODE HERE
    e_d = Reshape((input_length, embedding_size, 1))(e_d)
    # print('new e_d',e_d)

    # CNN layers
    conv_blocks = []
    for kernel_size in kernel_sizes:
        conv = Conv2D(filters=filters, kernel_size=(kernel_size, embedding_size), padding=padding, activation=activation, strides=(strides, strides))(e_d)
        maxpooling = MaxPool2D(pool_size=((input_length-kernel_size)//strides+1, 1))(conv)
        faltten = Flatten()(maxpooling)
        conv_blocks.append(faltten)

    # concatenate CNN results
    c = Concatenate()(conv_blocks) if len(kernel_sizes) > 1 else conv_blocks[0]
    c_d = Dropout(dropout_rate)(c)


    # dense layer
    d = Dense(hidden_size, activation=activation)(c_d)

    # output layer
    y = Dense(output_size, activation='softmax')(d)

    # build your own model
    model = Model(x, y)
    

    # SGD optimizer with momentum
    optimizer = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # training
    model.fit(train_data_matrix, train_data_label,  validation_data=(test_data_matrix, test_data_label), epochs=total_epoch, batch_size=batch_size)
    # testing
    train_score = model.evaluate(train_data_matrix, train_data_label, batch_size=batch_size)
    test_score = model.evaluate(test_data_matrix, test_data_label, batch_size=batch_size)

    print('Training Loss: {}\n Training Accuracy: {}\n'
          'Testng Loss: {}\n Testing accuracy: {}'.format(
              train_score[0], train_score[1],
              test_score[0], test_score[1]))
