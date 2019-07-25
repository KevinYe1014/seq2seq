# # import re
# # # string = 'absc\nlll'
# # # print(string)
# # # a = r'\n'
# # # print(re.sub(a, '', string))
# # #
# # # a = 'hello word you see'
# # # print(a.split())
# # # #
# # # # from nltk.corpus import stopwords
# # # # print(stopwords.words('english'))
# # #
# # # p = re.compile('(\d)-(\d*)')
# # # m = p.match('1-22-3') # m.group() == m.group(0)
# # # print(m.group(), m.group(0), m.group(1), m.group(2), m.groups(), m.groupdict())
# # #
# # #
# # # line = "Cats are smarter than dogs"
# # # matchObj = re.match(r'(.*) are (.*) ', line, re.M | re.I)
# # #
# # # if matchObj:
# # #     print("matchObj.group() : ", matchObj.group())
# # #     print("matchObj.group(1) : ", matchObj.group(1))
# # #     print("matchObj.group(2) : ", matchObj.group(2))
# # # else:
# # #     print("No match!!")
# # #
# # # a = 'xxIxxjshdxxlovexxsffaxxpythonxx'
# # # print(re.findall('xx(.*?)xx', a))
# # #
# # # print(re.findall('啦*', 'f'))
# # # print(re.findall('匹配规则*', '这个字符串是否有匹配规则则则则'))
# # # print(re.split('(\W+)', ' runoob, runoob, runoob.') )
# # #
# # #
# # # test_str = 'adb defg'
# # # print(test_str)
# # # print(re.findall('/n', test_str))
# # #
# # # print(re.split(r'[\s\,]+', 'a,b, c  d,,e, f, , g , ,h'))
# # #
# # # print(re.split(r'[0-9]', 'rubeydh4i255dah00132diofaedhfioiou'))
# #
# # s = '12 34\n56 78\n90'
# # print(re.findall( r'^\d+', s))
# # print(re.findall(r'\A\d+', s))
# #
# # s = 'abc abcde bc bcd'
# # print(re.findall(r'\bbc\b', s))
# # print(re.findall(r'\sbc\s', s))
# # s = 'aaa bbb111 cc22cc 33dd'
# # print(re.findall(r'\b[a-z]+\d*\b', s))
# # s = '/* part 1 */ code /* part 2 */'
# # print(re.findall(r'//*.*?/*/', s))
# # # print(re.findall(r"(?<=//*).+?(?=/*/)", s))  1.3 No
# #
# # s = 'aaa111aaa,bbb222,333ccc,444ddd444,555eee666,fff777ggg'
# # print(re.findall(r'(?P<g1>[a-z]+)\d+(?P=g1)', s))
# # s = '2548647'
# # print(re.findall(r'^(0|[1-9][0-9]*)$', s))
#
#
# import pandas as pd
# import numpy as np
# from sklearn.externals import joblib
# # # cloumns_name = ['Text', 'Summary']
# # # df = pd.DataFrame( columns = cloumns_name)
# # # a = ['hello word', 'you are good', 'do you kown']
# # # b = ['hello', 'you', 'do']
# # # df['Text'] = a
# # # df['Summary'] = b
# # # df.to_csv(r'c:/users/yelei/desktop/d3.csv', index=False)
# # p1 = pd.read_csv(r'c:/users/yelei/desktop/d3.csv')
# # #
# # # df = pd.DataFrame( columns = cloumns_name)
# # # a1 = ['pretty world', 'bravo you said']
# # # b1 = ['pretty', 'bravo']
# # # df['Text'] = a1
# # # df['Summary'] = b1
# # #
# # # df.to_csv(r'c:/users/yelei/desktop/d4.csv', index=False)
# # p2 = pd.read_csv(r'c:/users/yelei/desktop/d4.csv')
# # p = pd.concat([p1, p2])
# # p.to_csv(r'c:/users/yelei/desktop/p.csv', index=False)
# # p = pd.read_csv(r'c:/users/yelei/desktop/p.csv')
# # print(p)
#
# def to_csv(clean_texts, clean_summaries, i):
#     columns_name = ['Text', 'Summary']
#     df = pd.DataFrame(columns=columns_name)
#     df['Text'] = clean_texts
#     df['Summary'] = clean_summaries
#     df.to_csv(r'c:/users/yelei/desktop/Reviews_Clean_{}.csv'.format(i), index=False)
#
#
# def CleanAll(reviews_):
#     summarys = reviews_.Summary
#     texts = reviews_.Text
#     clean_summaries = []
#     clean_texts = []
#     for summary in summarys:
#         clean_summaries.append(summary)
#     for text in texts:
#         clean_texts.append(text)
#     return clean_summaries, clean_texts

# clean the summaries and texts
# # 处理text时填太慢了，分两部分处理
# # 568411
# reviews = pd.read_csv(r'c:/users/yelei/desktop/p.csv')
# midIndex = (int)(len(reviews) * 0.5)
# clean_summaries, clean_texts = CleanAll(reviews[0: midIndex])
# to_csv(clean_texts, clean_summaries, str(1)) # 284205
# print(pd.read_csv(r'c:/users/yelei/desktop/Reviews_Clean_1.csv'))
# clean_summaries, clean_texts = CleanAll(reviews[midIndex: len(reviews)])
# to_csv(clean_texts, clean_summaries, str(2)) # 284206
# print(pd.read_csv(r'c:/users/yelei/desktop/Reviews_Clean_2.csv'))
# word_embedding_matrix = np.zeros((2, 3), dtype=np.float32)
# for i in range(word_embedding_matrix.shape[0]):
#     word_embedding_matrix[i] = np.random.uniform(-1.0, 1.0, 3)

# joblib.dump(word_embedding_matrix, r'c:/users/yelei/desktop/word_embedding_matrix')
# a = joblib.load(r'c:/users/yelei/desktop/word_embedding_matrix')
# print(type(a))

# print(str(word_embedding_matrix[0]))
# with open(r'c:/users/yelei/desktop/word_embedding_matrix.txt', 'w', encoding='utf-8') as f:
#     for i in range(word_embedding_matrix.shape[0]):
#         lis = [str(word) for word in word_embedding_matrix[i]]
#         lines = ' '.join(lis)
#         lines += '\n'
#         f.write(lines)
#

# with open(r'c:/users/yelei/desktop/word_embedding_matrix.txt', encoding='utf-8') as f:
#     for line in f:
#         line = line.strip('\n')
#         values = np.asarray(line.split(), dtype=np.float32)
#
# print(values)

# vocabs_to_int = {}
# vocabs = list('abcd')
# for i in range(4):
#     vocabs_to_int[vocabs[i]] = i
# vocabs_to_int = pd.DataFrame(vocabs_to_int)
# print(vocabs_to_int)

# lis = [[1, 2, 5], [4], [8, 70, 12, 52], [8, 70, 12 ], [55]]
# lis = sorted(lis, key=lambda x: len(x))
# print(lis)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
# # print(tf.__version__)
# a = tf.get_variable(initializer=tf.constant_initializer([[1, 2, 3], [4, 5, 6]]), name='a', shape=[2, 3])
# b = tf.reverse(a, [-1])
# init_op = tf.global_variables_initializer()
# with tf.Session() as sess :
#     sess.run(init_op)
#     print(sess.run(b))

mask = tf.sequence_mask([1, 2, 3], 5, dtype=tf.float32)
with tf.Session() as sess:
    print(sess.run(mask))