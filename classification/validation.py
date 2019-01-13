import gensim
from gensim.models import Word2Vec
from collections import defaultdict
import numpy as np
from lib.DataExtractor import DataExtractor
from lib.CNNCategorizer import CNNCategorizer


def main():

    # configuration 
    word2vec_model_path = "resources/word2vec/en.bin"
    trainfile = "data/data.csv"    

    print("Loading nltk dependencies...")
    import nltk
    nltk.download('punkt')
    
    print("Loading w2v model...")
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
    #word2vec_model = gensim.models.Word2Vec.load(word2vec_model_path)
    
    # extract dataset
    train_dict = DataExtractor(trainfile).csv_to_dict()

    print("Instantiating classifier...")
    classifier = CNNCategorizer(word2vec_model, train_dict, n_gram=2, vecsize=300, nb_filters=1200, max_num_words_at_sentence=15)

    print("Evaluation in progress...")
    scores = classifier.evaluate()

    print("===================\nMean accuracy: {0:.2f}".format(np.mean(scores)) + " (+/- {0:.2f})".format(np.std(scores)))

    print("Done!")
    
if __name__ == '__main__':
    main()