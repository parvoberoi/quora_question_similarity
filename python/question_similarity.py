from keras.models import model_from_json
from preprocess import *
from tfidf_utils import get_text_vector

import os
import pickle
import sys

def predict_similarity(question_left, question_right):
	cwd = os.getcwd()

	# load final model
	model = model_from_json(open(os.path.join(cwd,  "models/final_model.model")).read())
	model.load_weights(os.path.join(cwd, "models/final_model.model") + '.weights')

	# load stored corpus TFIDF model for translating inputs
	vectorizer = pickle.load(open(os.path.join(cwd, "models/final_tfidf.pickle"), "rb"))
	word2tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
	
	question_1 = unicode(question_left)
	question_2 = unicode(question_right)

	question_1_vector = get_text_vector(question_1, word2tfidf)
	question_2_vector = get_text_vector(question_2, word2tfidf)

	prediction = model.predict([question_1_vector, question_2_vector])
	return prediction[0][0] < 0.5


if __name__ == "__main__":
	questions = sys.argv[1:]
	if len(questions) != 2:
		raise Exception("Incorrect number of sentences provided. Please enclose input questions in quotes as two separate arguments")
	

	if(predict_similarity(questions[0], questions[1])):
		print("Yes")
	else:
		print("No")
