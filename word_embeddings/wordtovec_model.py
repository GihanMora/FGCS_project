# Python program to generate word vectors using Word2Vec

# importing all necessary modules
import csv
import json
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.cluster import KMeans

warnings.filterwarnings(action='ignore')
import matplotlib.pyplot as plt
from gensim.utils import simple_preprocess
from nltk import BigramCollocationFinder, BigramAssocMeasures, TrigramCollocationFinder, TrigramAssocMeasures
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import gensim
import spacy

import gensim
import json

from gensim.utils import simple_preprocess
from rake_nltk import Rake, Metric
from gensim.models import Word2Vec
from wordcloud import STOPWORDS
stop_words = list(STOPWORDS)+["one","going","go","things","will","know","really","said","say","see","talk","think","time","help","thing","want","day","work"]
def get_ngrams(tokens, n ):
    n_grams = list(ngrams(tokens, n))
    print(n_grams)
    return [ ' '.join(grams) for grams in n_grams]
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


def extract_nearest_tokens(posts,category,model_a):
    data_words = list(sent_to_words(posts))
    data_words_nostops = remove_stopwords(data_words)
    data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ'])
    tri_g=[]
    for each in data_lemmatized:
        tr = get_ngrams(each, 3)
        tri_g.append(tr)
    #
    print('tri g',tri_g)
    all_tokens = [j for i in data_lemmatized for j in i]
    n_g = get_ngrams(all_tokens, 3)
    print(n_g)
    # combined_text = " ".join(all_tokens)
    # print("all tokens",all_tokens)
    # print(category.split(" "))

    # tri_grams = []
    # data = data_words
    # for k in data:
    #     for p in get_ngrams(k, 3):
    #         tri_grams.append(p)
    # print(sent_tokenize(f[:710000]))
    # print('tok',sent_tokenize(combined_text))
    # iterate through each sentence in the file

    # for i in sent_tokenize(combined_text):
    #     temp = []
    #     # print("i",i)
    #     # tokenize the sentence into words
    #     print('tokenized',word_tokenize(i))
    #     for j in word_tokenize(i):
    #         if ((j.lower() not in stop_words) and (j.lower()).isalpha()):
    #             temp.append(j.lower())
    #
    #     data.append(temp)
    # print('data',data)
    try:
        if(model_a=='CBOW'):
            model = gensim.models.Word2Vec(tri_g , min_count=1,size=100, window=5)

            # cbow_results = model1.most_similar(positive=category.split(" "), topn=100)
            seed = ['physical','symptom'#Physical symptoms
            , 'anxiety','head'
            , 'panic','attack'
            , 'muscle','twitch'
            , 'hand'
            , 'pain'
            , 'brain'
            , 'body'
            , 'coronary'
            , 'untamed',
            
            'disorder'
            , 'nerve',
            
            'syndrome'

            , 'chronic','pain'
            , 'palpitation'
            , 'cardiology']
            #
            # similarity = model.wv.n_similarity(['muscle'], ['fun'])
            # print(similarity)
            cbow_results = model.most_similar(positive=['physical','pain','body'],negative=['fear','doctor','people','day','much'], topn=10)
            print("cbow", cbow_results)
            #


            # return cbow_results



        # Create Skip Gram model
        if(model == 'SKIP'):
            model2 = gensim.models.Word2Vec(data_lemmatized, min_count=1, size=100,window=5, sg=1)
            skip_results = model2.most_similar(positive=category.split(" "), topn=100)
            print("skip", skip_results)
            return skip_results


    except SyntaxError:
        print("error",KeyError)
        return []



df = pd.read_csv("../Data/ANXIETY_all_posts.csv", encoding='utf8')
# print(df['post'][0])
anxiety_post_set = []
for each_p in df['post']:
    anxiety_post_set.append(each_p)

extract_nearest_tokens(anxiety_post_set,"anxiety","CBOW")



# path_f = "F://Armitage_project//crawl_n_depth//extracted_json_files//1727_Smith_Bros_Group_data.json"
# path_f = "F://Armitage_project//crawl_n_depth//extracted_json_files//544_Aged_Care_Channel_data.json"
# sort_on_relevance(path_f, "plant hire search site construction","CBOW")
# with open('improve_res.csv', 'w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Wordcloud original results', 'Sorted by relavence to industry using CBOW model', 'Sorted by relavence to industry using SKIP-GRAM model', 'Get top 100 similar tokens for industry using CBOW', 'Get top 100 similar tokens for industry using SKIP-GRAM'])
#
#     for each_res in range(len(word_cloud_results)):
#         writer.writerow([word_cloud_results[each_res],filtered_results_cbow[each_res],filtered_results_skip[each_res],cbow_results[each_res][0],skip_results[each_res][0]])