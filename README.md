# quora_question_similarity
A deep learning based naive implementation of the quora similarity question problem.

# Approach
The approach used here relies on using word2vec embeddings from spacy which we use to train a siamese network to make final predictions on whether two questions are similar or not. This is achieved by training a joint embedding neural network model which tries to minimize the distance between two similar questions while at the same time trying to maximize the distance between two unrelated questions.

# How to run
The project provides a basic trainer file(python/example_trainer.py) which is provided as a proof of concept on how the model training workflow looks like.

If you want to test whether two questions(strings) are similar or not, i also provide a ready to use (python/question_similarity.py) which uses a more robust model which has been trained on the 80/20 split of the entire 400k dataset with 25 iterations of the training data.

example usage: python python/question_similarity.py "What are some examples of products that can be make from crude oil?" "What are some of the products made from crude oil?"

PS: ensure that the two questions are provided within quotes to ensure that they are correctly parsed.
I also provide the tfidf model that was learned as part of the full training to ensure that the results can be reproduced as well as to be used for the prediction stage.


# Results
The simple siamese network with very basic tfidf pre-processing was able to achieve a accuracy of 79.8% on a random 80/20 split of the quora data.

# Future works
## ML Improvements
- preprocess data to get rid of punctuation marks
- try converting common words (What, which, where, how, etc.) to lowercase as currently the tfidf model associates differents weights with uppercase and lowercase instances
- investigate getting rid of stop words as well as converting apostrophe words to their normal forms
- investigate augmenting the training data by randomly mixing two unrelated questions as not-similar.
- run a parameter sweep on the various NN architecural params to find a optimal solution

## System Improvements
- build a standalone executable of the python predictor with all packages included to facilitate easier predictions

# System Requirements
- You will need keras 2.1.6, numpy 1.13.3, tensorflow 1.9.0 and pandas 0.18.0
