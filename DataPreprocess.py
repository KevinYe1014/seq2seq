import pandas as pd
import numpy as np
import re, time, os
from nltk.corpus import stopwords
import pprint
from sklearn.externals import joblib


# region
# reviews = pd.read_csv('./data/Reviews.csv')
#
# print(reviews.shape) # (568454, 10)
# reviews_area1 = reviews.iloc[0, 0:9]
# print(reviews_area1, '\n', reviews.iloc[0, 9])
# '''
# Id                                            1
# ProductId                            B001E4KFG0
# UserId                           A3SGXH7AUHU8GW
# ProfileName                          delmartian
# HelpfulnessNumerator                          1
# HelpfulnessDenominator                        1
# Score                                         5
# Time                                 1303862400
# Summary                   Good Quality Dog Food
# Text
# I have bought several of the Vitality canned dog food products and have found them all to be of good quality. ...
# '''
#
#
# # # check for any nulls values
# null_sum = reviews.isnull().sum()
# print(null_sum)
# '''
# # Id                         0
# # ProductId                  0
# # UserId                     0
# # ProfileName               16
# # HelpfulnessNumerator       0
# # HelpfulnessDenominator     0
# # Score                      0
# # Time                       0
# # Summary                   27
# # Text                       0
# # '''
#
# # remove null values and unneeded features
# reviews = reviews.dropna()
# reviews = reviews.drop(['Id','ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator',
#                         'Score','Time'], axis=1)
# reviews = reviews.reset_index(drop=True)
# print(reviews.shape)
# #  (568411, 2)
# for i in range(5):
#     index = i + 1
#     summary = reviews.iloc[i, 0]
#     text = reviews.iloc[i, 1]
#     print('Review #{0}:\n Summary: {1}\n Text: {2}\n'.format(str(index), summary, text))
# '''
# Review #1:
#  Summary: Good Quality Dog Food
#  Text: I have bought several of the Vitality canned dog food products and have found ...
#
#  '''
#
# contractions = {
# "ain't": "am not",
# "aren't": "are not",
# "can't": "cannot",
# "can't've": "cannot have",
# "'cause": "because",
# "could've": "could have",
# "couldn't": "could not",
# "couldn't've": "could not have",
# "didn't": "did not",
# "doesn't": "does not",
# "don't": "do not",
# "hadn't": "had not",
# "hadn't've": "had not have",
# "hasn't": "has not",
# "haven't": "have not",
# "he'd": "he would",
# "he'd've": "he would have",
# "he'll": "he will",
# "he's": "he is",
# "how'd": "how did",
# "how'll": "how will",
# "how's": "how is",
# "i'd": "i would",
# "i'll": "i will",
# "i'm": "i am",
# "i've": "i have",
# "isn't": "is not",
# "it'd": "it would",
# "it'll": "it will",
# "it's": "it is",
# "let's": "let us",
# "ma'am": "madam",
# "mayn't": "may not",
# "might've": "might have",
# "mightn't": "might not",
# "must've": "must have",
# "mustn't": "must not",
# "needn't": "need not",
# "oughtn't": "ought not",
# "shan't": "shall not",
# "sha'n't": "shall not",
# "she'd": "she would",
# "she'll": "she will",
# "she's": "she is",
# "should've": "should have",
# "shouldn't": "should not",
# "that'd": "that would",
# "that's": "that is",
# "there'd": "there had",
# "there's": "there is",
# "they'd": "they would",
# "they'll": "they will",
# "they're": "they are",
# "they've": "they have",
# "wasn't": "was not",
# "we'd": "we would",
# "we'll": "we will",
# "we're": "we are",
# "we've": "we have",
# "weren't": "were not",
# "what'll": "what will",
# "what're": "what are",
# "what's": "what is",
# "what've": "what have",
# "where'd": "where did",
# "where's": "where is",
# "who'll": "who will",
# "who's": "who is",
# "won't": "will not",
# "wouldn't": "would not",
# "you'd": "you would",
# "you'll": "you will",
# "you're": "you are"
# }
#
# def clean_text(text, remove_stopwords = True):
#     """
#     Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings
#     :param text:
#     :param remove_stopwords:
#     :return:
#     """
#     # convert to lower
#     text = text.lower()
#     # replace contractions with their longer forms
#     if True:
#         text = text.split()
#         new_text = []
#         for word in text:
#             if word in contractions:
#                 new_text.append(contractions[word])
#             else:
#                 new_text.append(word)
#         text = ' '.join(new_text)
#         # formate words and remove unwanted characters
#
#
#     text = re.sub(r'https?:\/\//.*[\r\n]*', '', text, flags=re.MULTILINE)
#     text = re.sub(r'\<a href', ' ', text)
#     text = re.sub(r'&amp;', '', text)
#     text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
#     text = re.sub(r'<br />', ' ', text)
#     text = re.sub(r'\'', ' ', text)
#
#     # optionally, remove stop words
#     if remove_stopwords:
#         text = text.split()
#         stops = set(stopwords.words('english'))
#         text = [w for w in text if not w in stops]
#         text = ' '.join(text)
#     return text
#
# # 打印stopwords.words('english')
# # print(stopwords.words('english'))
# # 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'hers
# def to_csv(clean_texts, clean_summaries, i):
#     columns_name = ['Text', 'Summary']
#     df = pd.DataFrame(columns=columns_name)
#     df['Text'] = clean_texts
#     df['Summary'] = clean_summaries
#     df.to_csv('./data/Reviews_Clean_{}.csv'.format(i), index=False)
#
#
# def CleanAll(reviews_):
#     summarys = reviews_.Summary
#     texts = reviews_.Text
#     clean_summaries = []
#     clean_texts = []
#     for summary in summarys:
#         clean_summaries.append(clean_text(summary, remove_stopwords=False))
#     for text in texts:
#         clean_texts.append(clean_text(text))
#     return clean_summaries, clean_texts
#
# # clean the summaries and texts
# # 处理text时填太慢了，分两部分处理
# # 568411
# midIndex = (int)(len(reviews) * 0.5)
# clean_summaries, clean_texts = CleanAll(reviews[0: midIndex])
# to_csv(clean_texts, clean_summaries, str(1)) # 284205
# clean_summaries, clean_texts = CleanAll(reviews[midIndex: len(reviews)])
# to_csv(clean_texts, clean_summaries, str(2)) # 284206

#endregion

# Reviews_Clean_1 = pd.read_csv('./data/Reviews_Clean_1.csv')
# Reviews_Clean_2 = pd.read_csv('./data/Reviews_Clean_2.csv')
# print(Reviews_Clean_1.head())
# # 0  bought several vitality canned dog food produc...  good quality dog food
# # 1  product arrived labeled jumbo salted peanuts p...      not as advertised
# print(Reviews_Clean_2.head())
#
# Reviews_Clean = pd.concat([Reviews_Clean_1, Reviews_Clean_2])
# Reviews_Clean.to_csv('./data/Reviews_Clean.csv', index=False)


# 之前的
# clean_summaries = []
# for summary in reviews.Summary:
#     clean_summaries.append(clean_text(summary, remove_stopwords=False))
# print('summaries are complete. ')
#
# clean_texts = []
# for text in reviews.Text:
#     clean_texts.append(clean_text(text))
# print('Texts are complete. ')
# 由于每次读取很慢，所以阶段性保存数据

# reviews = pd.read_csv('./data/Reviews_Clean.csv')
# reviews = reviews.dropna()  # 进行一次筛选
# clean_texts = reviews.Text
# clean_summaries = reviews.Summary
#
#
# region 计算词向量表
#
# def count_words(count_dict, text):
#     '''count the number of orrurrences of each word in a set of text'''
#     for sentence in text:
#         try: # 会有nan
#             for word in sentence.split():
#                 if word not in count_dict:
#                     count_dict[word] = 1
#                 else:
#                     count_dict[word] += 1
#         except:
#             continue
#
# # # Find the number of times each word was used and the size of the vocabulary
# word_counts = {}
# count_words(word_counts, clean_summaries) # 'good': 25861
# count_words(word_counts, clean_texts)
# print('Size of vocabulary: ',len(word_counts)) # 145369
#
#
# # Load conceptnet Numberbath's (CN) embeddings, similar to GloVe, but probably better
# # heeps://github.com/commonsense/conceptnet-numberbatch
# embeddings_index = {}
# with open('./textSum/numberbatch-en-17.04b.txt', encoding='utf-8') as f:
#     # reach_for_sky 0.1360 -0.2528 -0.2735 -0.1900 -0.0702 -0.0633 -0.0563 0.0944 ...
#     for line in f:
#         values = line.split(' ')
#         word = values[0]
#         try:
#             embedding = np.asarray(values[1:], dtype='float32')
#             embeddings_index[word] = embedding
#         except:
#             continue
#             print(line)
#
#
# print('Word embeddings: ',len(embeddings_index))   # 418082 每个300维
#
# # Find the number of words that are missing from CN, and are used more than our threshold
# missing_words = 0
# threshold = 20
# for word, count in word_counts.items():
#     if count > threshold:
#         if word not in embeddings_index:
#             missing_words += 1
#
# missing_ratio = round(missing_words / len(word_counts), 4) * 100
# print('Number of words missing from CN: ', missing_words)
# print('Percent of words that are missing from vocabulary:{}% '.format(missing_ratio))
# # Number of words missing from CN:  4229
# # Percent of words that are missing from vocabulary:  2.91
#
# # limit the vocab that we will use to words that appear >= threshold or are in GloVe
# # dictionary to convert words to integers
# vocabs_to_int = {}
# value = 0
# for word, count in word_counts.items():
#     if count >= threshold or word in embeddings_index:
#         vocabs_to_int[word] = value
#         value += 1
# # Special tokens that will be added to our vocab
# codes = ['<UNK>', '<PAD>', '<EOS>', '<GO>']
#
# # Add codes to vocab
# for code in codes:
#     vocabs_to_int[code] = len(vocabs_to_int)
#
# # Dictionary to convert integers to words
# int_to_vocabs = {}
# for word, value in vocabs_to_int.items():
#     int_to_vocabs[value] = word
#
# usage_ratio = round(len(vocabs_to_int) / len(word_counts), 4) * 100
# print('Total number of unique words: ', len(word_counts))
# print('Number of words we will use: ', len(vocabs_to_int))
# print('Percent of words we will use: {}%'.format(usage_ratio))
# # Total number of unique words:  145369
# # Number of words we will use:  60450
# # Percent of words we will use: 41.58%
#
#
# # 导出vocabs_to_int 和 int_to_vocabs
# # joblib.dump(int_to_vocabs, './textSum/int_to_vocabs')
# # joblib.dump(vocabs_to_int, './textSum/vocabs_to_int')
#
# # Need to use 300 for embedding dimensions to match CN's Vectors
# embedding_dim = 300
# nb_words = len(vocabs_to_int)
#
# # Create matrix with default values of zero
# word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)
# for word, i in vocabs_to_int.items():
#     if word in embeddings_index:
#         word_embedding_matrix[i] = embeddings_index[word] # embeddings_index就是 good : [ ... ] 300维的
#     else:
#         # if word not in CN, create a random embedding for it
#         new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
#         embeddings_index[word] = new_embedding
#         word_embedding_matrix[i] = new_embedding
#
# # check if value matches len(vocab_to_int)
# print("Embedding matrix: ", len(word_embedding_matrix)) # 60450
#
# # 临时保存，保存本次实验中词汇以及对应的词向量 [60450, 300]
# # with open('./textSum/word_embedding_matrix.txt', 'w', encoding='utf-8') as f:
# #     for i in range(len(word_embedding_matrix)):
# #         lis = [str(word) for word in word_embedding_matrix[i]]
# #         lines = ' '.join(lis)
# #         lines += '\n'
# #         f.write(lines)
# # 临时保存改为joblib
# joblib.dump(word_embedding_matrix, './textSum/word_embedding_matrix')

# endregion

# word_embedding_matrix = []
# with open('./textSum/word_embedding_matrix.txt', 'r', encoding='utf-8') as f:
#     for line in f:
#         line = line.strip('\n')
#         word_embedding_matrix.append(line_list = line.split())


# 导入数据vocabs_to_int int_to_vocabs 和word_embedding_matrix

reviews = pd.read_csv('./data/Reviews_Clean.csv')
reviews = reviews.dropna()  # 进行一次筛选
clean_texts = reviews.Text
clean_summaries = reviews.Summary
vocabs_to_int = joblib.load('./textSum/vocabs_to_int')
int_to_vocabs = joblib.load('./textSum/int_to_vocabs')
word_embedding_matrix = joblib.load('./textSum/word_embedding_matrix')





def convert_to_ints(text, word_count, unk_count, eos=False):
    '''
    Convert words in text to an integer.
    If word is not in vocab_to_int, use UNK's integer.
    Total the number of words and UNKs.
    Add EOS token to the end of texts
    :param text:
    :param word_count:
    :param unk_count:
    :param eos:
    :return:
    '''
    ints = []
    for sentence in text:
        sentence_ints = []
        for word in sentence.split():
            word_count  += 1
            if word in vocabs_to_int:
                sentence_ints.append(vocabs_to_int[word])
            else:
                sentence_ints.append(vocabs_to_int['<UNK>'])
                unk_count += 1
        if eos:
            sentence_ints.append(vocabs_to_int['<EOS>'])
        ints.append(sentence_ints)
    return ints, word_count, unk_count

# Apply convert_to_ints to clean_summaries and clean_texts
word_count = 0
unk_count = 0
int_summaries, word_count, unk_count = convert_to_ints(clean_summaries, word_count, unk_count)
int_texts, word_count, unk_count = convert_to_ints(clean_texts, word_count, unk_count, eos=True)

unk_percent = round(unk_count / word_count, 4) *100
print('Total number of words in headlines: ', word_count)
print('Total number of UNKs in headlines: ', unk_count)
print('Percent of words that are UNK: {}%'.format(unk_percent))
# Total number of words in headlines:  26481762
# Total number of UNKs in headlines:  219797
# Percent of words that are UNK: 0.83

def create_lengths(text):
    '''Create a data frame of the sentence lengths from a text'''
    lengths = []
    for sentence in text:
        lengths.append(len(sentence))
    return pd.DataFrame(lengths, columns=['counts'])

lengths_summaries = create_lengths(int_summaries)
lengths_texts = create_lengths(int_texts)

print('Summaries: ')
print(lengths_summaries.describe())
print('Texts:')
print(lengths_texts.describe())
# Summaries:
#               counts
# count  568409.000000
# mean        4.181749
# std         2.657970
# min         0.000000
# 25%         2.000000
# 50%         4.000000
# 75%         5.000000
# max        48.000000

# Inspect the length of texts

print(np.percentile(lengths_texts.counts, 90))
print(np.percentile(lengths_texts.counts, 95))
print(np.percentile(lengths_texts.counts, 99))

#Inspect the length of summaries
print(np.percentile(lengths_summaries.counts, 90))
print(np.percentile(lengths_summaries.counts, 95))
print(np.percentile(lengths_summaries.counts, 99))
# 87.0
# 119.0
# 219.0
# 8.0
# 9.0
# 13.0



def unk_counter(sentence):
    '''Counts the number of time UNK appears in a sentence'''
    unk_count = 0
    for word in sentence:
        if word == vocabs_to_int['<UNK>']:
            unk_count += 1
    return unk_count

# Sort the summaries and texts by the length of texts, shorter to longest
# Limit the length of summaries and texts based on the min and max ranges
# Remove reviews that include too many UNKs
sorted_summaries = []
sorted_texts = []
max_text_length = 84
max_summary_length = 13
min_length = 2
unk_text_limit = 1
unk_summary_limit = 0

for length in range(min(lengths_texts.counts), max_text_length):
    for count, words  in enumerate(int_summaries):
        if (len(int_summaries[count]) >= min_length and
            len(int_summaries[count]) <= max_summary_length and
                len(int_texts[count]) >= min_length and
                unk_counter(int_summaries[count]) <= unk_summary_limit and
                unk_counter(int_texts[count]) <= unk_text_limit and
                length == len(int_texts[count])
        ):
            sorted_summaries.append(int_summaries[count])
            # sorted_texts.append(int_texts[count])
# compare lengths to ensure they match
print(len(sorted_summaries))
# print(len(sorted_texts))

# 保存数据
# joblib.dump(sorted_texts, './textSum/sorted_texts')
joblib.dump(sorted_summaries, './textSum/sorted_summaries')

# 加载数据