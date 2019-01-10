import numpy as np
from keras.layers import Convolution1D, MaxPooling1D, Flatten, Dense
from keras.models import Sequential
from keras.models import model_from_json
from nltk import word_tokenize
import gensim
from gensim.models import Word2Vec

class CNNCategorizer:
    def __init__(self, word2vec_model, train_dict, n_gram, vecsize, nb_filters, max_num_words_at_sentence):
        self.word2vec_model = word2vec_model
        self.train_dict = train_dict
        self.n_gram = n_gram
        self.vecsize = vecsize
        self.nb_filters = nb_filters
        self.max_num_words_at_sentence = max_num_words_at_sentence
        self.trained = False

    def prepare_trainingdata(self):
        category_labels = self.train_dict.keys()
        category_label_to_index = dict(zip(category_labels, range(len(category_labels))))

        # tokenize
        sentences = []
        indices = []
        for category in category_labels:
            for sentence in self.train_dict[category]:
                indices.append(
                    [1 if category_label_to_index[category] == i else 0 for i in range(len(category_labels)) ]
                )
                sentences.append(word_tokenize(sentence))
        indices = np.array(indices, dtype=np.int)
        
        # store embedded vectors
        train_embedvec = np.zeros(shape=(len(sentences), self.max_num_words_at_sentence, self.vecsize))
        for i in range(len(sentences)):
            for j in range(min(self.max_num_words_at_sentence, len(sentences[i]))):
                train_embedvec[i, j] = self.word_to_embedvec(sentences[i][j])
        
        return category_labels, train_embedvec, indices


    def train(self):
        # prepare data for training 
        self.category_labels, train_embedvec, indices = self.prepare_trainingdata()

        # build CNN
        model = Sequential()
        model.add(Convolution1D(nb_filter=self.nb_filters,
                                filter_length=self.n_gram,
                                border_mode='valid',
                                activation='relu',
                                input_shape=(self.max_num_words_at_sentence, self.vecsize)))
        model.add(MaxPooling1D(pool_length=self.max_num_words_at_sentence - self.n_gram + 1))
        model.add(Flatten())
        model.add(Dense(len(self.category_labels), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        # train the model
        model.fit(train_embedvec, indices)

        # flags
        self.model = model
        self.trained = True

    def savemodel(self, nameprefix):
        if not self.trained:
            raise Exception('Model not trained')

        # save model configuration and weights
        with open('{}.json'.format(nameprefix), 'w') as model_file:
            model_file.write(self.model.to_json())
        self.model.save_weights('{}.h5'.format(nameprefix))

        # save category labels
        with open('{}_category_labels.txt'.format(nameprefix), 'w') as labelfile:
            labelfile.write('\n'.join(self.category_labels))

    def loadmodel(self, nameprefix):
        # load model configuration and weights
        self.model = model_from_json(open('{}.json'.format(nameprefix), 'rb').read())
        self.model.load_weights('{}.h5'.format(nameprefix))
    
        # extract category labels
        with open('{}_category_labels.txt'.format(nameprefix), 'r') as labelfile:
            self.category_labels = list(map(lambda line: line.strip(), labelfile.readlines()))

        self.trained = True

    def word_to_embedvec(self, word):
        return self.word2vec_model[word] if word in self.word2vec_model else np.zeros(self.vecsize)

    def sentence_to_embedvec(self, shorttext):
        tokens = word_tokenize(shorttext)
        matrix = np.zeros((self.max_num_words_at_sentence, self.vecsize))
        for i in range(min(self.max_num_words_at_sentence, len(tokens))):
            matrix[i] = self.word_to_embedvec(tokens[i])
        return matrix

    def score(self, shorttext):
        if not self.trained:
            raise Exception('Model not trained')

        matrix = np.array([self.sentence_to_embedvec(shorttext)])

        predictions = self.model.predict(matrix)

        scoredict = {}
        for idx, classlabel in zip(range(len(self.category_labels)), self.category_labels):
            scoredict[classlabel] = predictions[0][idx]
        return scoredict
