"""
Example file on how to run the NN trainer with a small(40k) training data set
"""
from data_utils import *
from model import *
from preprocess import *
from tfidf_utils import *

df = get_data_frame("/Users/parvoberoi/Desktop/quora-similar-questions/train.csv")
generate_length_statistics(df, ["question1", "question2"])
df = filter_rows_with_character_limit(df, 10)
df = encode_questions(df, ["question1", "question2"])
df = df[:40000]
question_left = "question1"
question_right = "question2"
df = generate_feature_vectors(df, question_left, question_right)
(X_train, Y_train, X_test, Y_test) = get_train_test_data(df)
model = train(X_train, Y_train, X_test, Y_test, epochs=1, iterations=25)
