import numpy as np
from data_utils import *
from tfidf_utils import *
import pickle
import os

def print_num(num):
	print("Number is %d" % num)


def get_model_formatted_data(df, q1_feat_col_name, q2_feat_col_name, label_col):
    X = np.zeros([df.shape[0], 2, 384])
    Y = np.zeros([df.shape[0]]) 

    q1 = [a[None,:] for a in list(df[q1_feat_col_name].values)]
    q1_feats = np.concatenate(q1, axis=0)

    q2 = [a[None,:] for a in list(df[q2_feat_col_name].values)]
    q2_feats = np.concatenate(q2, axis=0)

    X[:,0,:] = q1_feats[:,0,:]   # temporary hack to deal with initial incorrect shape
    X[:,1,:] = q2_feats[:,0,:]   # temporary hack to deal with initial incorrect shape
    Y = df[label_col].values
    return X, Y

# csv_data_path = "/Users/parvoberoi/Desktop/quora-similar-questions/train.csv"
def get_data_frame(csv_data_path):
	return read_csv_data(csv_data_path)
	
def preprocess_data(df, columns_to_process=["question1", "question2"]):
	# encode string questions as unicode
	# TODO(parvoberoi): pre-process to get rid of punctuation marks
	df = encode_questions(df, columns_to_process)
	column_stats = generate_length_statistics(df, columns_to_process)
	df = filter_rows_with_character_limit(df, filter_threshold=10, )

	return df


def generate_feature_vectors(df, question_left="question1", question_right="question2", lowercase=False):
    text_corpus = list(df[question_left]) + list(df[question_right])
    word2tfidf = generate_tfidf_weights(text_corpus, lowercase)

    # generate tokenized features for questions
    df[question_left + "_feats"] = get_column_feature_vector(df, question_left, word2tfidf)
    df[question_right + "_feats"] = get_column_feature_vector(df, question_right, word2tfidf)
    return df



def get_train_test_data(df, question_left="question1_feats", question_right="question2_feats", label_col="is_duplicate"):
    # split into training and test set
    (train_df, test_df, num_train, num_test) = split_train_test(df, train_percentage=0.8)

    # get data formatted for model train and eval
    X_train, Y_train = get_model_formatted_data(train_df, question_left, question_right, label_col)
    X_test, Y_test = get_model_formatted_data(test_df, question_left, question_right, label_col)
    return (X_train, Y_train, X_test, Y_test)

 
