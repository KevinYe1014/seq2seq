import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.externals import joblib


# region 加载数据
reviews = pd.read_csv('./data/Reviews_Clean.csv')
reviews = reviews.dropna()  # 进行一次筛选
clean_texts = reviews.Text

vocabs_to_int = joblib.load('./textSum/vocabs_to_int')
int_to_vocabs = joblib.load('./textSum/int_to_vocabs')
# endregion



batch_size = 64

## 测试效果
def text_to_seq(text):
    '''Prepare the text for the model'''
    return [vocabs_to_int.get(word, vocabs_to_int['<UNK>']) for word in text.split()]

# Create your own or use one from the dataset
# input_sentence = 'I have never eaten an apple before, but this red one was nice
# I think that I Will try a green apple next time'
# text = text_to_seq(input_sentence)
random = np.random.randint(0, len((clean_texts)))
input_sentence = clean_texts[random]
text = text_to_seq(input_sentence)

checkpoint = './best_model.ckpt'

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    text_length = loaded_graph.get_tensor_by_name('text_length:0')
    summary_length = loaded_graph.get_tensor_by_name('summary_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    # Multiply by batch_size to match the model's input parameters
    answer_logits = sess.run(logits, {input_data: [text] * batch_size, summary_length: [np.random.randint(5, 8)],
                                      text_length: [len(text)] * batch_size, keep_prob: 1.0})[0]
# Remove the padding from the tweet
pad = vocabs_to_int['<PAD>']

print('Original Text: ', input_sentence)
print('\nText')
print(' Word Ids: {}'.format([i for i in text]))
print(' Input words: {}'.format(' '.join([int_to_vocabs[i] for i in text])))

print('\nSummary')
print(' Word Ids: {}'.format([i for i in answer_logits if i != pad]))
print(' Response Words: {}'.format(' '.join([int_to_vocabs[i] for i in answer_logits if i != pad])))
