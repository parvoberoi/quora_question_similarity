import numpy as np
# from data_utils import *
# from tfidf_utils import *
import pickle
import os
import spacy

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
def get_data_frame(data_path):
    df = pd.read_csv(data_path, sep=',',header=0)
    return df

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
    nlp = spacy.load('en')
    df[question_left + "_feats"] = get_column_feature_vector(df, question_left, word2tfidf, nlp)
    df[question_right + "_feats"] = get_column_feature_vector(df, question_right, word2tfidf, nlp)
    return df



def split_train_test(df, train_percentage=0.8):
    if (train_percentage >= 1):
        raise Exception(
            "Percentage of training data cannot be greater than 1, select a value between 0.1 to 0.99. Ideally 0.8"
        )

    # shuffle df
    df = df.reindex(np.random.permutation(df.index))

    train_data, test_data = train_test_split(df, test_size=(1 - train_percentage))
    num_train = train_data.shape[0]
    num_test = test_data.shape[0]
    return (train_data, test_data, num_train, num_test)

def get_train_test_data(df, question_left_feats="question1_feats", question_right_feats="question2_feats", label_col="is_duplicate", train_percentage=0.8):
    # split into training and test set
    (train_df, test_df, num_train, num_test) = split_train_test(df, train_percentage)

    # get data formatted for model train and eval
    X_train, Y_train = get_model_formatted_data(train_df, question_left_feats, question_right_feats, label_col)
    X_test, Y_test = get_model_formatted_data(test_df, question_left_feats, question_right_feats, label_col)
    return (X_train, Y_train, X_test, Y_test)


