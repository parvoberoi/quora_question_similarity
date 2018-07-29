import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


# text_corpus = list(df['question1']) + list(df['question2'])
def get_wordTFIDFModel(text_corpus, lowercase=False):
    tfidf = TfidfVectorizer(lowercase=lowercase)
    tfidf.fit_transform(text_corpus)

    return tfidf

def generate_tfidf_weights(text_corpus, lowercase, dump_model=False):
    # generate tfidf model
    tfidf_model = get_wordTFIDFModel(text_corpus, lowercase=False)
    if dump_model:
        dump_tfidf_model(tfidf_model)

    return dict(zip(tfidf_model.get_feature_names(), tfidf_model.idf_))


def dump_tfidf_model(tfidf_model, extension="models/tfidf.pickle"):
    pickle.dump(
        tfidf_model,
        open(os.path.join(os.getcwd(), extension), "wb")
    )


def get_text_vector(text, word2TFIDF, nlp=None):
    if not nlp:
        nlp = spacy.load('en')
    # tokenize
    doc = nlp(text)
    word = doc[0].vector
    word_vec = np.zeros([1, len(word)])
    for word in doc:
        # word2vec
        vec = word.vector
        # fetch tfidf score if token not found ignore
        idf = word2TFIDF.get(str(word), 0)
        # compute final vec
        word_vec += vec * idf
    mean_vec = np.divide(word_vec, len(doc))
    return mean_vec

def get_column_feature_vector(df, column_name, word2TFIDF, nlp=None):
    if not nlp:
        nlp = spacy.load('en')
    feature_vector = []
    for text in list(df[column_name]):
        mean_vec = get_text_vector(text, word2TFIDF, nlp)
        feature_vector.append(mean_vec)
    return feature_vector
