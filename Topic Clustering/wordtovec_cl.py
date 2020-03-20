# Python program to generate word vectors using Word2Vec

# importing all necessary modules
import csv
import json
import pandas as pd


from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
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
    # print(n_grams)
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
    data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN','VERB'])
    #
    # print('lemma',data_lemmatized)
    # all_tokens = [j for i in data_lemmatized for j in i]
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
            # word_list = ['anxiety symptoms', 'bad anxiety', 'other anxiety', '’s anxiety', 'physical anxiety symptoms', 'health anxiety', 'anxiety attacks', 'health anxiety issues', 'bad anxiety driving', 'anxiety issues', 'anxiety medication', 'anxiety problems', 'anxiety feelings', 'other anxiety disorders', 'constant anxiety attacks', 'constant anxiety', 'anxiety head', 'daily anxiety attacks', 'anxiety levels', 'real anxiety', 'social anxiety', 'high anxiety', 'extreme anxiety', 'severe anxiety', 'general anxiety', 'pre anxiety', 'clinical anxiety', 'chronic anxiety', 'incontinence anxiety', 'anxiety amiee2893', 'good things', 'climate change anxiety', 'bad time', 'new things', 'generalized anxiety disorder', 'other things', 'dinner prep anxiety', 'many things', 'little things', 'going bust', 'few things', 'many different things', 'more time', 'big time', 'health type things', 'things', 'little everyday things', 'family time', 'anxious times', 'most things']+['anxiety', 'good', 'people', 'life', 'bad', 'year', 'symptom', 'fear', 'lot', 'new', 'much', 'thank', 'week', 'job', 'great', 'thought', 'little', 'many', 'way', 'doctor', 'anxious', 'month', 'mind', 'hope', 'family', 'able', 'body', 'medication', 'big', 'psychologist', 'today', 'post', 'panic', 'friend', 'right', 'place', 'panic attack', 'problem', 'word', 'support', 'first', 'experience', 'issue', 'different', 'bit', 'feeling', 'last', 'situation', 'phone', 'wrong', 'long', 'person', 'moment', 'worry', 'old', 'scared', 'well', 'normal', 'stress', 'night', 'sure', 'muscle', 'thread', 'hand', 'hard', 'care', 'food', 'pain', 'brain', 'whole', 'difficult', 'appointment', 'head', 'advice', 'enough', 'home', 'part', 'point', 'question', 'twitch', 'wasn', 'helpful', 'stuff', 'similar', 'eye', 'mental health', 'small', 'end', 'constant', 'important', 'test', 'partner', 'control', 'sorry', 'step', 'full', 'use', 'scary', 'couple', 'doesn']+['anxieti', 'time', 'thing', 'day', 'exact symptom', 'year', 'lot', 'life', 'peopl', 'fear', 'sever week', 'way', 'mind', 'thank', 'good', 'profession help', 'job', 'work', 'other', 'muscl twitch', 'month', 'medic', 'doctor', 'abl', 'whole bodi weak', 'scare', 'famili issu', 'thought', 'best', 'post', 'anxiou', 'alon', 'psychologist', 'panic attack', 'right word', 'care', 'stress', 'depress', 'friend', 'panic', 'worri', 'one', 'sure', 'today', 'bit', 'bad', 'wors enemi', 'sorri', 'problem', 'forum']+['panic_attack', 'mental_health', 'health_anxiety', 'symptom_anxiety', 'anxiety_depression', 'anxiety_symptom', 'residential_care', 'chest_pain', 'few_day', 'muscle_twitch', 'physical_symptom', 'few_week', 'first_time', 'anxiety_bad', 'climate_change', 'heart_attack', 'many_people', 'thank_reply', 'anxiety_anxiety', 'care_facility', 'day_day', 'intrusive_thought', 'last_few', 'last_year', 'long_time', 'other_people', 'anxiety_attack', 'beyondblue_topic', 'few_year', 'good_luck', 'good_thing', 'heart_palpitation', 'last_week', 'new_job', 'next_week', 'same_thing', 'side_effect', 'thing_anxiety', 'blood_test', 'lot_people', 'time_day', 'anxiety_fear', 'bad_anxiety', 'depression_anxiety', 'everyday_thing', 'few_month', 'irrational_fear', 'kind_thought', 'left_hand', 'many_year', 'most_people', 'next_day', 'other_thing', 'people_anxiety', 'same_symptom', 'alone_time', 'anxiety_disorder', 'anxiety_medication', 'beyond_blue', 'little_bit', 'nervous_system', 'other_day', 'other_symptom', 'peace_mind', 'people_same', 'time_anxiety', 'time_time', 'time_year', 'anxiety_anxious', 'anxiety_many', 'anxiety_med', 'anxiety_new', 'anxiety_panic', 'anxiety_physical', 'anxiety_year', 'daily_basis', 'first_step', 'good_idea', 'health_issue', 'last_month', 'medical_professional', 'medication_anxiety', 'most_day', 'new_phone', 'next_time', 'only_thing', 'own_thread', 'paw_print', 'right_leg', 'thing_life', 'welcome_forum', 'anxiety_level', 'anxiety_lot', 'anxiety_thing', 'anxiety_time', 'anxious_thought', 'arm_leg', 'baby_step', 'big_fear', 'blue_forum']+['residential_care_facility', 'physical_symptom_anxiety', 'anxiety_physical_symptom', 'somatic_nervous_system', 'beyond_blue_forum', 'massive_panic_attack', 'mental_health_plan', 'mental_health_professional', 'anxiety_beyondblue_topic', 'anxiety_panic_attack', 'big_life_change', 'first_time_life', 'health_anxiety_bad', 'health_plan_psychologist', 'heart_attack_year', 'mental_health_issue', 'month_little_twitch', 'nerve_conduction_study', 'panic_attack_car', 'panic_attack_night', 'peace_mind_anxiety', 'positive_self_talk', 'topic_anxiety_beyondblue', 'www_beyondblue_org', 'a_website_www', 'actual_heart_attack', 'answer_question_irrational', 'anti_anxiety_med', 'anxiety_attack_traffic', 'anxiety_fear_worry', 'anxiety_home_safe', 'anxiety_many_form', 'anxiety_muscle_twitch', 'anxiety_new_job', 'anxiety_th_month', 'anxiety_whole_life', 'appointment_residential_care', 'available_phone_a', 'bad_panic_attack', 'bed_spare_room', 'beyondblue_org_professional', 'beyondblue_topic_anxiety', 'beyondblue_topic_year', 'care_facility_psychologist', 'care_good_way', 'carpal_tunnel_syndrome', 'child_heart_healthy', 'climate_change_anxiety', 'counsellor_support_service', 'deep_breath_hand', 'despair_young_strong', 'different_petrol_station', 'direction_help_area', 'doesn_hour_same', 'due_prolonged_stress', 'exact_same_symptom', 'extreme_muscle_body', 'fear_long_period', 'few_week_good', 'full_life_functional', 'full_panic_mode', 'gp_test_result', 'group_muscle_twitch', 'hand_foot_weird', 'hand_night_time', 'hand_whole_time', 'health_anxiety_time', 'health_counsellor_support', 'health_doesn_hour', 'heart_defect_surgery', 'heart_palpitation_anxiety', 'hour_same_challenge', 'huge_weight_shoulder', 'irrational_fear_same', 'kitten_old_self', 'life_change_anxiety', 'life_difficult_moment', 'life_functional_human', 'little_bit_more', 'little_twitch_fear', 'long_period_sort', 'lot_people_same', 'low_self_esteem', 'many_people_time', 'med_next_week', 'medication_terrible_side', 'mental_health_attention', 'mental_health_counsellor', 'mental_health_doesn', 'muscle_twitch_cramp', 'new_phone_paw', 'new_vacuum_cleaner', 'night_panic_attack', 'normal_normal_normal', 'old_self_sufficient', 'org_professional_mental', 'own_thread_topic', 'panic_attack_most', 'past_few_week', 'period_sort_trauma']
            # list_of_lists = []
            # for each_w in word_list:
            #     list_of_lists.append([each_w])
            tri_g = []
            for each in data_lemmatized:
                tr = get_ngrams(each, 3)
                tri_g.append(tr)
            # model = gensim.models.Word2Vec(tri_g , min_count=1,size=20, window=5)
            # X = model.wv.syn0
            # train_names= list(model.wv.vocab.keys())
            # print(X)

            max_epochs = 100
            vec_size = 20
            alpha = 0.025

            model = Word2Vec(size=vec_size,
                            alpha=alpha,
                            min_alpha=0.00025,
                            min_count=1,
                            )
            # print("tagged",tagged_data[0])
            model.build_vocab(tri_g)

            for epoch in range(max_epochs):
                print('iteration {0}'.format(epoch))
                model.train(tri_g,
                            total_examples=model.corpus_count,
                            epochs=model.iter)
                # decrease the learning rate
                model.alpha -= 0.0002
                # fix the learning rate, no decay
                model.min_alpha = model.alpha

            X = model.wv.syn0
            train_names = list(model.wv.vocab.keys())


            # kmeans = KMeans(n_clusters=8, random_state=0).fit(X)
            # print(kmeans.labels_)
            print(train_names)
            # print(len(kmeans.labels_))
            # # print('kmeans_lab',kmeans.labels_)
            # print(len(train_names))
            # craete a dictionary to get cluster data
            # clusters = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [],
            #             13: [], 14: [], 15: [], 16: [], 17: [], 18: [], 19: []}
            # for i in range(len(kmeans.labels_)):
            #     # print(i)
            #     # print(kmeans.labels_[i])
            #     # print(kmeans.labels_[i],train_names[i])
            #     clusters[kmeans.labels_[i]].append(train_names[i])
            # # print(clusters)
            # for each_i in clusters:
            #     tag_set = []
            #     for k in clusters[each_i]:
            #         # print(k)
            #         tag_set.append(k)
            #     print(each_i)
            #     print(tag_set[:20])
            #     print('*********************')
            # cbow_results = model.most_similar(positive=['husband avoid situation','family issue exposure','sharing head husband','avoidance husband force','cbt family issue','life marriage symptom'], topn=30)
            # cbow_results = model.most_similar(
            #     positive=['body weakness muscle','medication pain headache','leg anxiety struggle','stomach pain vomit',
            # 'pain headache pain'], topn=50)#good
            seeds = ['safety plan recommend','people suggest psychologist','seem forum help',
         'reach kid helpline','motivate start work','support depression journey','call support line',
         'people listen support','trust relationship support','mindfulness support group','ask want understand'
,'push direction help']


            # seeds = ['reconfirm doctor', 'ask rest', 'suggestion thought',
            #          'point advice', 'thank reply', 'support service', 'message support',
            #          'encourage reach', 'give idea', 'seek counselling', 'explain rid']


            tok=[]
            for each in seeds:
                cbow_results = model.most_similar(positive=[each], topn=10)  # good
                cbow_results = [i[0] for i in cbow_results]
                print(cbow_results)
                tok.extend(cbow_results)
            # print([i[0] for i in cbow_results1])
            print(tok)
            # cbow_results2 = model.most_similar(positive=seeds, topn=110)  # good
            # print(cbow_results2)
            # cbow_results2 = [i[0] for i in cbow_results2]
            # print(cbow_results2)
            # all_tokens = [j for i in tri_g for j in i]
            # for each in all_tokens:
            #     # print(each)
            #     seed = 'support'
            #     if(seed in each):
            #         print(each)
            # # print(cbow_results2)
            # cbow_results3 = model.most_similar(positive=['family rift hype'], topn=10)  # good
            # # print(cbow_results2)
            # cbow_results3 = [i[0] for i in cbow_results3]
            # cbow_results4 = model.most_similar(positive=['family self cry'], topn=10)  # good
            # cbow_results4 = [i[0] for i in cbow_results4]
            # # print(cbow_results3)
            # cbow_results5 = model.most_similar(positive=['family depression feeling'], topn=10)  # good
            # # print(cbow_results4)
            # cbow_results5 = [i[0] for i in cbow_results5]
            # cbow_results6 = model.most_similar(positive=['family plea tv'], topn=10)  # good
            # # print(cbow_results5)
            # cbow_results6 = [i[0] for i in cbow_results6]
            # cbow_results7 = model.most_similar(positive=['child health issue'], topn=10)  # good
            # # print(cbow_results6)
            # cbow_results7 = [i[0] for i in cbow_results7]
            # cbow_results8 = model.most_similar(positive=['child pressure dad'], topn=10)  # good
            # # print(cbow_results6)
            # cbow_results8 = [i[0] for i in cbow_results8]
            # cbow_results9 = model.most_similar(positive=['esteem criticism parent'], topn=10)  # good
            # # print(cbow_results6)
            # cbow_results9 = [i[0] for i in cbow_results9]
            # cbow_results10 = model.most_similar(positive=['finance wife illness'], topn=10)  # good
            # # print(cbow_results6)
            # cbow_results10 = [i[0] for i in cbow_results10]
            # cbow_results11 = model.most_similar(positive=['mother stress support'], topn=10)  # good
            # # print(cbow_results6)
            # cbow_results11 = [i[0] for i in cbow_results11]
            # cbow_results12 = model.most_similar(positive=['life family fiancee'], topn=10)  # good
            # # print(cbow_results6)
            # cbow_results12 = [i[0] for i in cbow_results12]
            # cbow_results13 = model.most_similar(positive=['boyfriend blame enemy'], topn=10)  # good
            # # print(cbow_results6)
            # cbow_results13 = [i[0] for i in cbow_results13]
            # cbow_results_all =cbow_results1+cbow_results2+cbow_results3+cbow_results4+cbow_results5+cbow_results6+cbow_results7+cbow_results8+cbow_results9+cbow_results10+cbow_results11+cbow_results12
            # print(cbow_results_all)

            # ['medication', 'doctor', 'psychologist', 'appointment',  # Treatments/ mentions of doctor, therapist
            #  'clinical', 'gloval structures need', 'contact cardiology offices', 'nice paramedics advised',
            #  'monitor physiological symptoms', 'medical professional', 'residential care facility',
            #  'mental health plan',
            #  'heart_defect_surgery', 'period sort trauma', 'latest depression med', 'safety planning',
            #  'mental health counsellor',
            #  'mental health plan', 'therapy'],


            #seed = ['physical symptoms'#Physical symptoms
            # , 'anxiety','head'
            # , 'panic','attack'
            # , 'muscle','twitch'
            # , 'hand'
            # , 'pain'
            # , 'brain'
            # , 'body'
            # , 'coronary','angiograms'
            # , 'untamed','beasts','sometimes'
            # , 'distorting','stifling'
            # , 'obsessive','compulsive','disorder'
            # , 'nerve','conduction','studies'
            # , 'compressed','ulnar','nerve'
            # , 'carpal','tunnel','syndrome'
            # , 'benign','fasciculation','syndrome'
            # , 'ventricular','ectopic','beats'
            # , 'chronic','pain'
            # , 'palpitation'
            # , 'cardiology']
            #
            # similarity = model1.wv.n_similarity(seed, 'medication')
            # print(similarity)
            # print("cbow", cbow_results)
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




df = pd.read_csv("../Data/Self_Harm_all_posts.csv", encoding='utf8')
# print(df['post'][0])
post_set = []
for each_p in df['post']:
    post_set.append(each_p)

extract_nearest_tokens(post_set,"anxiety","CBOW")
# path_f = "F://Armitage_project//crawl_n_depth//extracted_json_files//1727_Smith_Bros_Group_data.json"
# path_f = "F://Armitage_project//crawl_n_depth//extracted_json_files//544_Aged_Care_Channel_data.json"
# sort_on_relevance(path_f, "plant hire search site construction","CBOW")
# with open('improve_res.csv', 'w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Wordcloud original results', 'Sorted by relavence to industry using CBOW model', 'Sorted by relavence to industry using SKIP-GRAM model', 'Get top 100 similar tokens for industry using CBOW', 'Get top 100 similar tokens for industry using SKIP-GRAM'])
#
#     for each_res in range(len(word_cloud_results)):
#         writer.writerow([word_cloud_results[each_res],filtered_results_cbow[each_res],filtered_results_skip[each_res],cbow_results[each_res][0],skip_results[each_res][0]])



# ['treat seek reconfirm', 'reconfirm doctor seek', 'doctor seek see', 'doctor treat seek', 'tell doctor treat', 'seek see week', 'dentist tell doctor', 'see week process', 'fear dentist tell', 'week process hope', 'help fear dentist', 'capacity eat treat', 'eat treat advice', 'seek capacity eat', 'treat advice discharge']
# ['start ask rest', 'rest symptom describe', 'symptom describe muscle', 'symptom start ask', 'describe muscle twitch', 'request symptom start', 'muscle twitch let', 'refer request symptom', 'twitch let search', 'brain refer request', 'let search find', 'search find wake', 'request brain refer', 'find wake feeling', 'follow request brain']
# ['procrastinate suggestion thought', 'figure procrastinate suggestion', 'cleaner figure procrastinate', 'vacuum cleaner figure', 'assemble vacuum cleaner', 'begin assemble vacuum', 'take begin assemble', 'know take begin', 'find time help', 'down recovery reminder', 'help muscle relax', 'help read post', 'reminder help read', 'muscle relax remember', 'time help muscle']
# ['point advice give', 'thank understand point', 'advice give re', 'give re thought', 're thought process', 'thought process line', 'process line world', 'line world group', 'world group effort', 'way motivate business', 'must lead way', 'inspiration fear must', 'fear must lead', 'must lead inspiration', 'motivate business government']
# ['spot try let', 'reply spot try', 'try let stick', 'let stick spend', 'stick spend life', 'spend life try', 'life try anxiety', 'try anxiety situation', 'anxiety situation have', 'ask type question', 'question job group', 'position ask type', 'wonder question tell', 'question ask case', 'type question job']
# ['reach support service', 'service re phone', 're phone website', 'encourage reach support', 'phone website health', 'feel encourage reach', 'website health counsellor', 'health counsellor support', 'service phone website', 'counsellor support service', 'support service phone', 'support service give', 'may feel encourage', 'call support service', 'service give support']
# ['support option may', 'send message support', 'area send message', 'option may use', 'direction area send', 'may use feel', 'point direction area', 'danger call hope', 'area thought danger', 'thought danger call', 'direction area thought', 'hope morning get', 'call hope morning', 'support point direction', 'morning get post']
# ['reach support service', 'feel encourage reach', 'support service re', 'may feel encourage', 'service re phone', 're phone website', 'understand may feel', 'phone website health', 'support understand may', 'website health counsellor', 'health counsellor support', 'service phone website', 'counsellor support service', 'support service give', 'support service phone']
# ['give idea way', 'hope get give', 'bowl hope get', 'idea way make', 'distraction bowl hope', 'make date way', 'contain distraction bowl', 'way make date', 'date way manage', 'arrange contain distraction', 'way manage symptom', 'make date give', 'date give find', 'feel way make', 'friend arrange contain']
# ['seek counselling freind', 'course deppresion seek', 'imagine course deppresion', 'counselling freind work', 'expectation imagine course', 'other expectation imagine', 'freind work study', 'meet other expectation', 'work study may', 'study meet other', 'stress study meet', 'study may fall', 'perspective stress study', 'put perspective stress', 'may fall line']
# ['happen explain rid', 'rid tell ok', 'tell ok sound', 'share happen explain', 'ok sound start', 'sound start say', 'start say situation', 'say situation majority', 'situation majority folk', 'majority folk terrify', 'role month end', 'leave role month', 'choice leave role', 'end stress relief', 'month end stress']
