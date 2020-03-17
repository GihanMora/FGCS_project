import spacy

import gensim
import json

from gensim.utils import simple_preprocess
from rake_nltk import Rake, Metric

from wordcloud import STOPWORDS
stop_words = list(STOPWORDS)+["one","going","go","things","will","know","really","said","say","see","talk","think","time","help","thing","want","day","work"]

def sent_to_words(sentences):#split sentences to words and remove punctuations

    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def remove_stopwords(texts):#remove stopwords to do more effective extraction
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):#lemmatize words to get core word

    nlp = spacy.load('en', disable=['parser', 'ner'])
    nlp.max_length = 150000000
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def run_rake_model(posts,rake_limit):
    # from nltk.corpus import stopwords
    # stop_words = stopwords.words('english')
    # stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    # data_words = list(sent_to_words(posts))
    # data_words_nostops = remove_stopwords(data_words)
    # data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ'])
    # print(data_lemmatized)
    # all_tokens = [j for i in data_lemmatized for j in i]
    # combined_text = " ".join(all_tokens)

    combined_text = " ".join(posts)

    # text = ["RAKE short for Rapid Automatic Keyword Extraction algorithm, " \
    #        "is a domain independent keyword extraction algorithm which tries " \
    #        "to determine key phrases in a body of text by analyzing the frequency " \
    #        "of word appearance and its co-occurance with other words in the text."]

    r = Rake(max_length=3,min_length=1,ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO)
    # print('lemmatized',data_lemmatized)
    # total_data = []
    # for each in data_lemmatized:
    #     total_data+=each
    # print(total_data)
    # cleaned_text = " ".join(total_data)
    # print('cleaned',cleaned_text)
    # print('combined',text)
    r.extract_keywords_from_text(combined_text)
     # To get keyword phrases ranked highest to lowest.
    res = r.get_ranked_phrases_with_scores()
    res_words = r.get_ranked_phrases()
    # print(res)
    # print(res_words)
    return res_words[:100]

# run_rake_model("F://Armitage_project/crawl_n_depth/extracted_json_files/www.axcelerate.com.au_0_data.json",50)