import csv
import pandas as pd
from word_embeddings.wordtovec_model import extract_nearest_tokens
from Topic_Modelling.lda_model import run_lda_model
from Key_Phrase_Extration.textrank import run_textrank_model
from Key_Phrase_Extration.wordcloud_model import run_wordcloud_model
from Key_Phrase_Extration.Rake_model import run_rake_model
from Key_Phrase_Extration.kpe_model import key_phrase_extract
from Topic_Modelling.guided_lda import run_guided_lda_model

df = pd.read_csv("../Data/ANXIETY_all_posts.csv", encoding='utf8')
# print(df['post'][0])
anxiety_post_set = []
for each_p in df['post']:
    anxiety_post_set.append(each_p)

df = pd.read_csv("../Data/BI-POLAR_all_posts.csv", encoding='utf8')
# print(df['post'][0])
BI_polar_post_set = []
for each_p in df['post']:
    BI_polar_post_set.append(each_p)

df = pd.read_csv("../Data/Self_Harm_all_posts.csv", encoding='utf8')
# print(df['post'][0])
self_harm_post_set = []
for each_p in df['post']:
    self_harm_post_set.append(each_p)

# print(post_set[0])
# print(run_lda_model(self_harm_post_set, 10))
# print(run_guided_lda_model(self_harm_post_set,7))
# a_tr=run_textrank_model(anxiety_post_set,50,5)
# b_tr=run_textrank_model(BI_polar_post_set,50,5)
# s_tr=run_textrank_model(self_halm_post_set,50,5)
# print("anxiety",a_tr)
# print("bi",b_tr
# print("self",s_tr)
# print(a_tr)
# print(b_tr)
# print(s_tr)
# a_wc=run_wordcloud_model(anxiety_post_set,'tri')
# b_wc=run_wordcloud_model(BI_polar_post_set,'tri')
# s_wc=run_wordcloud_model(self_harm_post_set,'tri')
# print("anxiety",a_wc)
# print("bi",b_wc)
# print("self",s_wc)

#
# a_r=run_rake_model(anxiety_post_set, 1)
# b_r=run_rake_model(BI_polar_post_set, 1)
# s_r=run_rake_model(self_halm_post_set, 1)
# print(a_r)
# print(b_r)
# print(s_r)

# print(len(self_harm_post_set))
# key_phrase_extract(anxiety_post_set,50)
# key_phrase_extract(BI_polar_post_set[:600],50)
# key_phrase_extract(self_harm_post_set,50)

a_nt_c=extract_nearest_tokens(anxiety_post_set,"anxiety","CBOW")
# a_nt_s=extract_nearest_tokens(anxiety_post_set,"anxiety","SKIP")
# b_n_t_c=extract_nearest_tokens(BI_polar_post_set,"bi polar","CBOW")
# b_n_t_s=extract_nearest_tokens(BI_polar_post_set,"bi polar","SKIP")
# s_n_t_c=extract_nearest_tokens(self_harm_post_set,"self harm","CBOW")
# s_n_t_s=extract_nearest_tokens(self_harm_post_set,"self harm","SKIP")
cbow_list = []
print(len(a_nt_c))
for each in a_nt_c:
    cbow_list.append(each[0])
# skip_list = []
# for each in s_n_t_s:
#     skip_list.append(each[0])
#
print(cbow_list)
# print(skip_list)
#
# b_n_t_c=extract_nearest_tokens(BI_polar_post_set,"bi polar","CBOW")
# b_n_t_s=extract_nearest_tokens(BI_polar_post_set,"bi polar","SKIP")
#
# s_n_t_c=extract_nearest_tokens(self_halm_post_set,"self harm","CBOW")
# s_n_t_s=extract_nearest_tokens(self_halm_post_set,"self harm","SKIP")

# with open('extracted_topics.csv', 'w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     writer.writerow(['class', 'textrank_results', 'wordcloud_results', 'rake_results', 'word_2_vec_cbow_results','word_2_vec_skip_results'])
#     writer.writerow(['Anxiety',str(a_tr),str(a_wc),str(a_r),str(a_nt_c),str(a_nt_s)])
#     writer.writerow(['BI_polar',str(b_tr),str(b_wc),str(b_r),str(b_n_t_c),str(b_n_t_s)])
#     writer.writerow(['Self_halm',str(s_tr),str(s_wc),str(s_r),str(s_n_t_c),str(s_n_t_s)])

# aa=[['differ day', 0.013029953071326277], ['time', 0.01239997000883317], ['cheer', 0.01198000607921091], ['good', 0.011305081243335329], ['thing', 0.009435652895953316], ['bipolar world', 0.008711689702572929], ['year', 0.00867455533214047], ['kaz', 0.008365060337299998], ['med', 0.007520606253290082], ['lot', 0.006313944790120656], ['bit', 0.006175819575079947], ['peopl', 0.0061422457967520005], ['obviou way', 0.0048631869459807514], ['hiya len', 0.004789461112400802], ['toni', 0.004715884745216755], ['week', 0.004710361768904502], ['new medic', 0.004574502838816458], ['life', 0.0043409658816642585], ['wife', 0.004142078941352779], ['mani other', 0.00395735886115016], ['friend', 0.00393075509241071], ['head today', 0.003844777239468351], ['folk', 0.0037288907042841326], ['quirki', 0.0037255974495435922], ['bipolar moment', 0.003575866176345481], ['depress', 0.0035673530023541686], ['love', 0.0035466797284372277], ['work', 0.0035386781406917885], ['hour', 0.003519495483029657], ['great', 0.003420807368061186], ['head', 0.0031865240213662066], ['mallow', 0.0029668590083396017], ['better', 0.0029557952942447325], ['thank', 0.0029298455942630124], ['world', 0.0029158009607936773], ['excercis', 0.0028701788828949556], ['happi', 0.0028561139047472363], ['psych', 0.002818095573110446], ['hug', 0.0027765098149163535], ['differ', 0.0027391368006993663], ['hard', 0.0027369442910731563], ['glad', 0.002736779745513768], ['garden', 0.0026458919797302037], ['one', 0.0025939427592352165], ['care', 0.0025517535585910175], ['bed', 0.0025139389284052805], ['morn', 0.0025062308690560722], ['lol', 0.002461360917676767], ['post kazz', 0.002430934938007044], ['food', 0.002360211722538707]]
# bb=[]
# for i in aa:
#     bb.append(i[0])
# print(bb)

# cc=[['time', 0.018167049845362764], ['thing', 0.014219108733371687], ['life', 0.012763464755191262], ['peopl', 0.01170450452691765], ['day', 0.008643802797842244], ['support', 0.008175641868823115], ['wonder friend', 0.007704903410955265], ['thought', 0.00719564373513736], ['help', 0.006966681130317672], ['good', 0.006650076670485478], ['year', 0.006258405092146888], ['way', 0.006009833694613773], ['anxieti', 0.005948532970118213], ['suicid', 0.005848565732306448], ['happi', 0.005695929087341416], ['sorri', 0.005665125637554641], ['great', 0.005434197802242821], ['other', 0.005299634570072587], ['hard', 0.005256710275055453], ['famili', 0.00503670566871703], ['thank', 0.005008391014763626], ['depress', 0.004951039920194495], ['medic', 0.004858444661659618], ['work', 0.004847284107612052], ['poor mental health', 0.004722453880138097], ['forum', 0.004612773062146908], ['lot', 0.004597267131306052], ['hug', 0.004545858835370988], ['sure', 0.004479518404444943], ['feel', 0.004424257325489025], ['moment', 0.004356407074522296], ['husband', 0.004108725195011762], ['abl', 0.003988168730362407], ['better day', 0.0038824405372521052], ['job', 0.0037184170793441197], ['pain', 0.003620888228756791], ['gross one', 0.0035958907456917835], ['sarah', 0.003333668454039994], ['much support', 0.0031902651327462258], ['nik', 0.0030491632576606018], ['glad', 0.003033206402931796], ['self help book', 0.003002629182327701], ['parent relationship', 0.0029914158685399417], ['place', 0.0029491879810409722], ['idea', 0.002945990605608425], ['psychologist', 0.002941682782228064], ['safe space', 0.0029392611065432485], ['today', 0.0029017986392120185], ['doctor', 0.0028776396471194843], ['person', 0.0028701695549613717]]
# dd=[]
# for i in cc:
#     dd.append(i[0])
# print(dd)
