import gensim

import spacy
import pytextrank
import json

# example text
from gensim.utils import simple_preprocess
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
def run_textrank_model(posts,phrase_limit,summery_limit):  # this will extract paragraph and header text from given json file and extract the topics from that

    # data_words = list(sent_to_words(posts))
    # data_words_nostops = remove_stopwords(data_words)
    # data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ'])
    # print(data_lemmatized)
    # all_tokens = [j for i in data_lemmatized for j in i]
    # combined_text = " ".join(all_tokens)
    combined_text = " ".join(posts)
    # combined_text = h_p_data
    # print(combined_text)
    print("running textrank model")
    # load a spaCy model, depending on language, scale, etc.
    nlp = spacy.load("en_core_web_sm")

    # add PyTextRank to the spaCy pipeline
    tr = pytextrank.TextRank()
    nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)
    nlp.max_length = 150000000
    doc = nlp(combined_text)

    # examine the top-ranked phrases in the document
    tr_results = []
    tr_words = []
    for p in doc._.phrases[:phrase_limit]:
        print("{:.4f} {:5d}  {}".format(p.rank, p.count, p.text))
        tr_results.append([p.rank,p.count, p.text])
        tr_words.append(p.text)
        # print(p.chunks)
    # summery_res = []
    # for sent in doc._.textrank.summary(limit_sentences=summery_limit):
    #     print(sent)
    #     summery_res.append(str(sent))
    # print(tr_results)
    # print(summery_res)
    return tr_words
    # print(summery_res)
    # data[0]['textrank_resutls'] = tr_results  # dump the extracted topics back to the json file
    # data[0]['textrank_summery__resutls'] = summery_res
    # with open(path_to_json, 'w') as outfile:
    #     json.dump(data, outfile)

# run_textrank_model("F://Armitage_project//crawl_n_depth//extracted_json_files//0_www.sureway.com.au_data.json",50,5)