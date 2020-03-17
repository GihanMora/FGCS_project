import json
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
# spacy for lemmatization
import spacy
from lda import guidedlda
# Enable logging for gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import numpy as np
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
# NLTK Stop words
# from nltk.corpus import stopwords
# stop_words = stopwords.words('english')
# stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import STOPWORDS

# print(stop_words)
stop_words = list(STOPWORDS)+["one","going","go","things","will","know","really","said","say","see","talk","think","time","help","thing","want","day","work"]
# print(stop_words)


def sent_to_words(sentences):#split sentences to words and remove punctuations
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def make_bigrams(texts,bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts,bigram_mod,trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def remove_stopwords(texts):#remove stopwords to do more effective extraction
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]



def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):#lemmatize words to get core word
    """https://spacy.io/api/annotation"""
    nlp = spacy.load('en', disable=['parser', 'ner'])
    nlp.max_length = 150000000
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def run_guided_lda_model(posts,number_of_topics):#this will extract paragraph and header text from given json file and extract the topics from that
    print("guided lda model started")
    data_words = list(sent_to_words(posts))
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    print('remove_punctuations...')
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)
    print("words_list_no stop", len(data_words_nostops))
    data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)

    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    print('data_lemmatized...')
    all_tokens = [j for i in data_lemmatized for j in i]
    combined_text = " ".join(all_tokens)
    doc_list=[combined_text]

    token_vectorizer = CountVectorizer(doc_list,stop_words=stop_words)
    try:
        X= token_vectorizer.fit_transform(doc_list)

    # print('X',X)
    # print('voc',token_vectorizer.vocabulary_)
    # print('shape',X.shape)
        tf_feature_names = token_vectorizer.get_feature_names()
        word2id = dict((v, idx) for idx, v in enumerate(tf_feature_names))
    except ValueError:
        print("Vocabulary is empty")
        return "No enough Data/Vocabulary is empty"

    # print('w2id',word2id)


    seed_topic_list = [
        ['symptoms','health','disorder','care','palpitation']#health conditions
, ['medication','doctor','clinical','psychologist','cardiology','therapy']  #medical context
, ['fear','feeling','scared','worry','scary','pain']  #feelings
, ['social','family','friend','people','child']  #social context
, ['good', 'bad','constant ','high','extreme','general','panic','normal']  #conditions
, ['head','body','hand','brain']  #physical
, ['advice', 'help', 'support']  # support
]
# #bi
#     seed_topic_list = [
#         [ 'health', 'disorder', 'care']  # health conditions
#         , ['medication', 'doctor', 'cardiology', 'therapy']  # medical context
#         , ['fear', 'feeling', 'scared', 'worry', 'scary', 'pain']  # feelings
#         , ['social', 'family', 'friend', 'people', 'child']  # social context
#         , ['good', 'bad', 'constant ', 'high', 'extreme', 'general', 'panic', 'normal']  # conditions
#         , ['head', 'body', 'hand', 'brain']  # physical
#         , ['advice', 'help', 'support']  # support
#     ]
# #self
#     seed_topic_list = [
#         ['health', 'disorder', 'care']  # health conditions
#         , ['medication', 'doctor', 'psychologist', 'cardiology', 'therapy']  # medical context
#         , ['fear', 'feeling', 'scared', 'worry', 'scary', 'pain']  # feelings
#         , ['social', 'family', 'friend', 'people', 'child']  # social context
#         , ['good', 'bad', 'constant ', 'high', 'extreme', 'general', 'panic', 'normal']  # conditions
#         , ['head', 'body', 'hand', 'brain']  # physical
#         , ['advice', 'help', 'support']  # support
#     ]
    number_of_topics=7

    model = guidedlda.GuidedLDA(n_topics=number_of_topics, n_iter=100, random_state=7, refresh=20)

    seed_topics = {}

    for t_id, st in enumerate(seed_topic_list):
        for word in st:
            if(word in word2id.keys()):
                seed_topics[word2id[word]] = t_id
            else:
                try:
                    word2id[word]=str(int(list(word2id.keys())[-1])+1)
                    seed_topics[word2id[word]] = t_id
                except ValueError:
                    pass

    # print('st',seed_topics)
    # seed_topics = {3543: 0, 656: 0, 3940: 0, 5907: 1, 5690: 1, 1329: 1, 7364: 2, 4496: 2}
    model.fit(X, seed_topics=seed_topics, seed_confidence=0.35)

    n_top_words = 15
    topic_word = model.topic_word_
    # print(topic_word)
    topics_set = []
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(tf_feature_names)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        # print(topic_words)
        # print(list(topic_words))
        words = [w for w in topic_words]
        topics_set.append(words)
        # print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    print("final result")
    print(topics_set)
    # words_list = {'Topic_' + str(i): [word for word, prob in model.show_topic(i, topn=10)] for i in
    #               range(0, model.num_topics)}

    return topics_set


#To run this scrpit individually use following line and run the script
# topics = run_lda_model(path to the json object,number_of_topics)
# print(topics)
# run_guided_lda_model("F://Armitage_project/crawl_n_depth/extracted_json_files/www.axcelerate.com.au_0_data.json",3)