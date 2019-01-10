# NeuralNetworks-homework
## CNNClassificator
### Description
This is attempt to train the convolutional neural network model for classsification small texts (sentence level). We are using Keras lib for building simple CNN and Gensim word2vec as a tool for vectorizing. For training the model was decided to use a small dataset (`classification/data/data.csv`) for avoid to spend a lot of time. As a result example of work:
```
sentence> stone age
history  :  0.71295136
physics  :  0.123437874
mathematics  :  0.08566385
theology  :  0.077946864
sentence> god
theology  :  0.40092126
history  :  0.35237652
physics  :  0.16437434
mathematics  :  0.08232781
sentence> analysis
mathematics  :  0.44903654
physics  :  0.2362535
history  :  0.17560464
theology  :  0.13910529
sentence> war      
history  :  0.50802124
theology  :  0.1795608
physics  :  0.1578736
mathematics  :  0.15454437
```
### Start
```bash
$ docker build -t nn .
$ docker run --name nn -v $(pwd)/classification:/app/classification:rw --rm --entrypoint /bin/bash -it nn

# in container tty
python3.6 main.py
```
### Word2Vec model
This package require pre-trained gensim word2vec model. You can download it following by [link](https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download)
Below the instruction how to extract the model
```bash
$ cp /your/path/GoogleNews-vectors-negative300.bin.gz $REPO_ROOT/classification/resources/word2vec/
$ cd $REPO_ROOT/classification/resources/word2vec/

$ gunzip file.gz
$ mv GoogleNews-vectors-negative300.bin/data en.bin
```

