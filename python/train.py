import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# column_list_to_encode = ["question1", "question2"]
def encode_questions(df, column_list_to_encode):
    # encode questions to unicode
    for column in column_list_to_encode:
        try:
            if type(df[column][0]) == 'unicode':
                continue
        except:
            pass
        df[column] = df[column].apply(lambda x: unicode(str(x),"utf-8"))
    return df

# column_list_to_compute = ["question1", "question2"]
# find min, max length of questions in the dataset
def generate_length_statistics(df, column_list_to_compute):
    def get_length_column_name(column):
        return column + "_length"
    for column in column_list_to_compute:
        df[get_length_column_name(column)] = df[column].apply(str).apply(len)

    column_stats = {column: {} for column in column_list_to_compute}
    for column in column_list_to_compute:
        column_stats[column]["max"] = df[get_length_column_name(column)].max()
        column_stats[column]["min"] = df[get_length_column_name(column)].min()

    return column_stats

def filter_rows_with_character_limit(df, filter_threshold, filter_column_1="question1_length", filter_column_2="question2_length"):
    # ignore all rows with question size less than 10 characters 
    if (filter_column_1 not in df.columns) or (filter_column_2 not in df.columns):
        return df
    return df[(df[filter_column_1] > filter_threshold) & (df[filter_column_2] > filter_threshold)]
