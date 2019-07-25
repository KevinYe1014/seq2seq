import pandas as pd
import numpy as np
import tensorflow as tf
import re, time, os
from nltk.corpus import stopwords
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
import pprint
from sklearn.externals import joblib




# region 加载数据
reviews = pd.read_csv('./data/Reviews_Clean.csv')
reviews = reviews.dropna()  # 进行一次筛选
clean_texts = reviews.Text
clean_summaries = reviews.Summary


vocabs_to_int = joblib.load('./textSum/vocabs_to_int')
int_to_vocabs = joblib.load('./textSum/int_to_vocabs')
word_embedding_matrix = joblib.load('./textSum/word_embedding_matrix')


sorted_texts = joblib.load( './textSum/sorted_texts')
sorted_summaries = joblib.load('./textSum/sorted_summaries')

# endregion

def model_inputs():
    '''Create placeholder for inputs to the model'''
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32,  name='keep_prob')
    summary_length = tf.placeholder(tf.int32, (None,), name='summary_length')
    max_summary_length = tf.reduce_max(summary_length, name='max_dec_len')
    text_length = tf.placeholder(tf.int32, (None, ), name='text_length')

    return input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length

def process_encoding_input(target_data, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of batch'''
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)
    return dec_input

def enc_state_concat(enc_state):
    '''连接编码state输出'''
    encoder_state = []
    for i in range(num_layers):
        encoder_state.append(enc_state[0][i])  # forward
        encoder_state.append(enc_state[1][i])  # backward
    encoder_state = tuple(encoder_state)  # 2 tuple, 2 tuple(c & h), batch_size, hidden_size
    return encoder_state


# region encoder

def encoding_layerBi(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob):
    '''Create the encoding layer for bidirecotion'''
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
            cell_fw = tf.nn.rnn_cell.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_keep_prob=keep_prob)
            cell_bw = tf.nn.rnn_cell.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, input_keep_prob=keep_prob)

            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, rnn_inputs, sequence_length, dtype=tf.float32)

    # Join ouputs since we are using a bidirectional RNN
    # 将两个LSTM的输出连接为一个张量。
    enc_output = tf.concat(enc_output, 2)
    return enc_output, enc_state

def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob):
    '''Create the encoding layer '''
    def get_lstm_cell(rnn_size):
        lstm_cell_ = tf.nn.rnn_cell.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_, input_keep_prob= keep_prob)
        return lstm_cell
    cell = tf.nn.rnn_cell.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])
    enc_output, enc_state = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length =sequence_length, dtype= tf.float32 )
    return enc_output, enc_state

# endregion

# region train and infer decoder layer

def training_decoding_layer(dec_embed_input, summary_length, dec_cell, initial_state, output_layer,
                            vocab_size, max_summary_length):
    '''Create the training logits'''
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input, sequence_length=summary_length, time_major=False)
    # 问题：下面的initial_state我直接用encoder的输出放在这的，但是会报错：
    training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, training_helper, initial_state, output_layer)
    # training_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder, output_time_major=False,
    #                                                        impute_finished=True, maximum_iterations=max_summary_length)
    training_logits, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(training_decoder, output_time_major=False,
                                                              impute_finished=True,
                                                              maximum_iterations=max_summary_length)
    a = training_logits.rnn_output
    b = training_logits.sample_id

    return training_logits

def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state,
                             output_layer, max_summary_length, batch_size):
    '''Create the inference logits'''
    start_token = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, start_token, end_token)
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, inference_helper, initial_state, output_layer)
    inference_logits, _, _ =tf.contrib.seq2seq.dynamic_decode(inference_decoder, output_time_major=False,
                                                           impute_finished=True, maximum_iterations=max_summary_length)
    return inference_logits
# endregion

# region decoer layer

def decoding_layer(dec_embed_input, embeddings, enc_state, vocab_size, summary_length,
                   max_summary_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers):
    '''
    构造Decoder层
    :param dec_embed_input: decoder端的输入
    :param embeddings:
    :param enc_output:
    :param enc_state:
    :param vocab_size:
    :param text_length:
    :param summary_length:
    :param max_summary_length:
    :param rnn_size:
    :param vocab_to_int:
    :param keep_prob:
    :param batch_size:
    :param num_layer:
    :return:
    '''

    def get_decoder_cell(rnn_size):
        lstm = tf.nn.rnn_cell.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed = 2))
        dec_cell = tf.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob=keep_prob)
        return dec_cell
    cell = tf.nn.rnn_cell.MultiRNNCell([get_decoder_cell(rnn_size) for _ in range(num_layers)])
    # output全连接层
    output_layer = Dense(vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    with tf.variable_scope('decode'):
        training_logits = training_decoding_layer(dec_embed_input, summary_length, cell,
                                                      enc_state, output_layer, vocab_size, max_summary_length)
    with tf.variable_scope('decode', reuse=True):
        inference_logits = inference_decoding_layer(embeddings, vocabs_to_int['<GO>'], vocab_to_int['<EOS>'],
                                                        cell, enc_state, output_layer, max_summary_length,
                                                        batch_size)
        return training_logits, inference_logits



def decoding_layer_with_attention(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, text_length, summary_length,
                   max_summary_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layer):
    '''Create the decoding cell and attention for the training and inference decoding layers'''
    for layer in range(num_layer):
        with tf.variable_scope('decoder_{}'.format(layer)):
            lstm = tf.nn.rnn_cell.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            dec_cell = tf.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob=keep_prob)

    output_layer = Dense(vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    # 选择注意力权重的计算模型。BahdanauAttention是使用一个隐藏层的前馈神经网络。
    # text_length是一个维度为[batch_size]的张量，代表batch中每个句子的长度，
    # Attention需要根据这个信息把填充位置的注意力权重设置为0.
    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size, enc_output, text_length, normalize=False, name='BahdanauAttention')
    # 将解码器的循环神经网络和注意力一起封装成更高层的循环神经网络
    # AttentionWrapper类：用来组件rnn
    # cell和上述的所有类的实例，从而构建一个带有attention机制的Decoder
    attention_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attn_mech, rnn_size)
    # 接下来我们再看下 AttentionWrapperState 这个类，这个类其实比较简单，就是定义了 Attention 过程中可能需要保存的变量，如 cell_state、attention、time、alignments
    # 等内容，同时也便于后期的可视化呈现，代码实现如下：
    # initial_state = tf.contrib.seq2seq.AttentionWrapperState(enc_state[0], _zero_state_tensors(rnn_size, batch_size, tf.float32))
    # question : 这个地方一直有问题，导致程序走不通。所以目前不纠结这个了，就以看懂为准。
    # 概括问题如下：首先由于上面双向lstm输出，后面decoder输入，这两个连接连接不了？？？？

    with tf.variable_scope('decode'):
        training_logits = training_decoding_layer(dec_embed_input, summary_length, attention_cell,
                                                  enc_state, output_layer, vocab_size, max_summary_length)

    with tf.variable_scope('decode', reuse=True):
        inference_logits = inference_decoding_layer(embeddings, vocabs_to_int['<GO>'], vocab_to_int['<EOS>'],
                                                    attention_cell, enc_state, output_layer, max_summary_length, batch_size)

    return training_logits, inference_logits


# endregion

# region build model

def seq2seq_model(input_data, target_data, keep_prob, text_length, summary_length, max_summary_length,
                  vocab_size, rnn_size, num_layers, vocab_to_int, batch_size):
    '''Use the previous functions to create the training and inference logits'''

    # Use Nemberbatch's embeddings and the newly created ones as our embeddings
    embeddings = word_embedding_matrix
    enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)
    enc_output, enc_state = encoding_layer(rnn_size, text_length, num_layers, enc_embed_input, keep_prob)

    dec_input = process_encoding_input(target_data, vocab_to_int, batch_size)
    dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)


    training_logits, inference_logits = decoding_layer(dec_embed_input, embeddings,
                                                       enc_state, vocab_size,  summary_length,
                                                       max_summary_length, rnn_size, vocab_to_int, keep_prob,
                                                       batch_size, num_layers)
    return training_logits, inference_logits

# endregion

def pad_sentence_batch(sentence_batch):
    '''Pad sentences with <PAD> so that each sentence of a batch has the same length'''
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocabs_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(summaries, texts, batch_size):
    '''Batch summaries, texts, and the lengths of their sentences together'''
    for batch_i in range(len(texts) // batch_size):
        start_i = batch_i * batch_size
        summaries_batch = summaries[start_i : start_i + batch_size]
        texts_batch = texts[start_i : start_i+ batch_size]
        pad_summaries_batch = np.array(pad_sentence_batch(summaries_batch))
        pad_texts_batch = np.array(pad_sentence_batch(texts_batch))

        # Need the lengths for the _lengths parameters
        pad_summaries_lengths = []
        for summary in pad_summaries_batch:
            pad_summaries_lengths.append(len(summary))

        pad_texts_lengths = []
        for text in pad_texts_batch:
            pad_texts_lengths.append(len(text))

        yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths

# Set the Hyperparameters
epochs = 100
batch_size = 64
rnn_size = 256
num_layers = 2
learning_rate = 0.005
keep_probability = 0.75

# Build the graph
train_graph = tf.Graph()
# set the graph to default to ensure that it is ready for training
with train_graph.as_default():
    # Load the model inputs
    input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length = model_inputs()
    # Create the training and inference logits
    # tf.reverse调换tensor中元素位置，如果后面是[0] 则是最外面一维，越大越往内部延伸。 反之如果是[-1]则是最里面一层，越小越往外面延伸。
    # 在 Google 提出 Seq2Seq 的时候，提出了将输出的语句反序输入到 encoder中，这么做是为了在 encoder阶段的最后一个输出恰好就是 docoder阶段的第一个输入。
    # 在实现时，无论是在训练阶段还是测试阶段，都将句子反序输入，但是预测结果序列是正序而不是反序。
    training_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]), targets, keep_prob,
                                                      text_length, summary_length, max_summary_length, len(vocabs_to_int) + 1,
                                                      rnn_size, num_layers, vocabs_to_int, batch_size)
    # Create tensors for the training logits and inference logits
    training_logits = tf.identity(training_logits.rnn_output, 'logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

    # Create the weights for sequence_loss
    masks = tf.sequence_mask(summary_length, max_summary_length, dtype=tf.float32, name='masks')

    with tf.name_scope('optimization'):
        # loss function
        cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)

        # optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5, 5), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
    print('Graph is built. ')


## 训练网络
# Subset the data for training
start = 200000
end = start + 50000
sorted_summaries_short = sorted_summaries[start: end]
sorted_texts_short = sorted_texts[start: end]
print('The shortest text length: ', len(sorted_summaries_short[0]))
print('The longest text length: ', len(sorted_texts_short[-1]))

# Train the Model
learning_rate_decay = 0.95
min_learning_rate = 0.0005
display_step = 20 # Check training loss after every 20 batches
stop_early = 0
stop = 3 # If the update loss does not decrease in 3 consecutive update checks, stop training
per_epoch = 3 # Make 3 update checks per epoch
update_check = (len(sorted_texts_short) // batch_size // per_epoch) - 1

update_loss = 0
batch_loss = 0
summary_update_loss = []
# Record the update losses for saving improements in the model

checkpoint = 'best_model.ckpt'
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    # If we want to continue training a previous session
    for epoch_i in range(1, epochs + 1):
        update_loss = 0
        batch_loss = 0
        for batch_i, (summaries_batch, texts_batch, summary_lengths, texts_lengths) in enumerate(
            get_batches(sorted_summaries_short, sorted_texts_short, batch_size)):
            start_time = time.time()
            _, loss = sess.run([train_op, cost],
                               {input_data: texts_batch, targets: summaries_batch,
                                lr: learning_rate, summary_length: summary_lengths,
                                text_length: texts_lengths, keep_prob: keep_probability})
            batch_loss += loss
            update_loss += loss
            end_time = time.time()
            batch_time = end_time - start_time

            if batch_i % display_step == 0 and batch_i > 0:
                print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconde: {:>4.2f}'
                      .format(epoch_i, epochs, batch_i, len(sorted_texts_short) // batch_size,
                              batch_loss / display_step, batch_time * display_step))
                batch_loss = 0

            if batch_i % update_check == 0 and batch_i > 0:
                print('Average loss for this update: ', round(update_loss / update_check, 3))
                summary_update_loss.append(update_loss)

                # If the update loss is at a new minmum, save the model
                if update_loss <= min(summary_update_loss):
                    print('New Record!')
                    stop_early = 0
                    saver = tf.train.Saver()
                    saver.save(sess, checkpoint)
                else:
                    print('No Improvement. ')
                    stop_early += 1
                    if stop_early == stop:
                        break
                update_loss = 0

            # Reduce learning rate, but not below its minimum value
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            if stop_early == stop:
                print('Stoppint Training. ')
                break









