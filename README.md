# Abstract classification via word2vec embedding

### Goal

The aim of this study is to understand whether a word2vec embedding coupled with a neural network for classification can achieve better performance than simpler embeddings like bag-of-words or tf-idf vectorizers. Our study is divided in two main parts,


- PART 1: We train a word2vec embedder using ~20,000 abstracts collected via the Nature metadata dataset, filtered by the keyword *field theory*.


- PART 2: We consider a restricted dataset, produced in a previous study (see the [link](https://github.com/carlosparaciari/abstract-classification-supervised)) where the abstract have labels from 9 possible classes, and we train different neural network architectures for classification.

### Methods

We use a dataset that we have extracted, analysed, and cleanded in a different stduy (see link above). The dataset is obtained via the Nature metadata API. From previous analysis, we know that ensemble learners and linear methods achieve an accuracy of around 73-77% on this dataset, with observations from 3 of the 9 classes being particulary difficult to correctly assign. The word embedding used in the previous study is a tf-idf vectorizer.

In the first part of the project, we train the gensim word2vec model to embed words commonly appearing in scientific papers, and particularly in papers concerning *quantum field theory*. We decide to train the embedder instead of using a pre-trained model since abstracts of papers published in scientific journals ofter contains lots of jargon.

The second stage of this project consists in training different architectures of neural networks for classifying the abstracts. Specifically, we first use a Convolutional Neural Network (CNN) searching for specific features involving a given number of words in the text. Later, we use a Recursive Neural Network (RNN) to obtian information from the whole abstract.

### Results

For what concerns the word2vec embedding we trained, we can only judge its performance by demanding similar words to a given set of words, that we consider meaningful for the task. While the model we trained is by no means general enough to be used in production, it seems to be sufficient for our pourposes. Indeed, almost all the word associations made by the model seem to be reasonable.

For example, the model suggests that the most similar words to "eth" are "eigenstate" and "thermalization", that indeed are the first two terms in the acronim. For "ads cft", the model associate "holography", that indded is a primary feature of the AdS-CFT correspondence.

We then train abstract classifiers using different Neural Network architectures, see below for a detailed description of the architectures. Interestingly, the two architectures used perform very differently, with the CNN achieving an accuracy ~ 81% (thus outperforming all previous methods used on this dataset), and the RNN obtaining an accuracy of ~ 74% (close to the one achived by Gradient Boosting in the previous study).

Furthermore, we find that the RNN requires an higher number of weights (equivalently, hidden nodes/layers) than the CNN, and the training takes considerably more time.

### Remarks

There are multiple aspects that could be interesting studying.

- For the word2vec embedding, we trained the model using CBOW (continuous bag-of-words), but an alternative could be Skip-Gram. In order to achive better embeddings, one should also probably use a much bigger corpus than the one used here.


- We could also try to use pre-trained embeddings, although these are (reasonably) trained on non-scientific texts, and therefore would struggle with the jargon used in scientific jounal abstracts. We could use the embedding as a baseline.


- The two NN architectures used here are quite simple, and it might be interesting to use more advanced one (e.g. Bidirectional Encoder Representations from Transformers, BERT)

### Sections

- Word2vec embedding using gensim

- Abstract classifiers using TensorFlow/Keras
    - Convolutional Neural Network
    - Recursive Neural Network
    
- Model evaluation (Mc Nemar Test)