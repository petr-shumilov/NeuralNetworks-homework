import gensim
from gensim.models import Word2Vec
from collections import defaultdict
from lib.DataExtractor import DataExtractor
from lib.CNNCategorizer import CNNCategorizer


def main():

    # configuration 
    word2vec_model_path = "resources/word2vec/en.bin"
    output_nameprefix = "model/ver0_1"
    input_nameprefix = "model/ver0_1"
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


    try:
        print("Loading the model...")
        classifier.loadmodel(input_nameprefix)

    except FileNotFoundError:
        print("Couldn't find the model. Train...")
        classifier.train()

        print("Saving models...")
        classifier.savemodel(output_nameprefix)

        print("Training successfully finished")

    finally: 
        # Interactive mode
        while True:
            sentence = input('sentence> ')
            if len(sentence) > 0:
                for label, score in sorted(classifier.score(sentence).items(), key=lambda s: s[1], reverse=True):
                    print(label, ' : ', score)
            else:
                break


if __name__ == '__main__':
    main()