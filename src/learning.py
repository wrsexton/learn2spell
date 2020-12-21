import typing as T
import tensorflow as TF
import numpy as NP
import json as J
import warnings as W

from collections import Counter
from os import path

import utility as U
import spells as S

def createLookupTable(data: str) -> dict:
    """Build vocabulary for machine learning
    
    :param data: The string from which to build the lookup table.
    :return: A tuple containing two maps associating vocab and integer values"""
    counts = Counter(data)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab)}
    return vocab_to_int

def get_inputs():
    """Create TF Placeholders for input, targets, and learning rate.

    :return: Tuple (input, targets, learning rate)
    """
    inputs = TF.placeholder(TF.int32, [None, None], name='input')
    targets = TF.placeholder(TF.int32, [None, None], name='targets')
    learning_rate = TF.placeholder(TF.float32, name='learning_rate')
    return (inputs,targets,learning_rate)

def get_init_cell(batch_size, rnn_size):
    """Create an RNN Cell and initialize it.
    
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    n_lstm_layers = 2
    cell = TF.contrib.rnn.MultiRNNCell(
        [TF.contrib.rnn.BasicLSTMCell(rnn_size) for _ in range(n_lstm_layers)])
    initial_state = TF.identity(
        cell.zero_state(batch_size, TF.float32), name='initial_state')
    return (cell, initial_state)

def get_embed(input_data, vocab_size, embed_dim):
    """Create embedding for <input_data>.

    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    embedding = TF.Variable(
        TF.random_uniform([vocab_size, embed_dim], minval=-1.0, maxval=1.0))
    return TF.nn.embedding_lookup(embedding, input_data)


def build_rnn(cell, inputs):
    """Create a RNN using a RNN Cell
    
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    outputs, final_state = TF.nn.dynamic_rnn(cell, inputs, dtype=TF.float32)
    final_state = TF.identity(final_state, name='final_state')
    return (outputs, final_state)

def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """Build part of the neural network
    
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    seq_length = input_data.shape[1]
    embed = get_embed(input_data, vocab_size, embed_dim)
    lstm_outputs, final_state = build_rnn(cell, embed)
    Logits = TF.layers.dense(lstm_outputs, vocab_size, activation=None)
    return (Logits, final_state)

def get_batches(int_text, batch_size, seq_length):
    """Return batches of input and target

    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    # Number of words per batch and batches
    words_per_batch = batch_size * seq_length
    n_batches = len(int_text) // words_per_batch
    # Keep words to allow for a full batch
    int_text = NP.array(int_text[:words_per_batch * n_batches])
    # Shifted array for targets
    int_target = NP.append(int_text[1:], int_text[0])
    # Reshape into batch_size rows
    int_text = int_text.reshape((batch_size, -1))
    int_target = int_target.reshape((batch_size, -1))
    # Batch matrix
    batch_data = np.zeros((n_batches, 2, batch_size, seq_length), dtype=np.int32)
    # fill batch_data matrix
    batches_idx = 0
    for n in range(0, int_text.shape[1], seq_length):
        # features
        batch_data[batches_idx, 0, :, :] = int_text[:, n:n+seq_length]
        # targets
        batch_data[batches_idx, 1, :, :] = int_target[:, n:n+seq_length]
        # increment
        batches_idx += 1
    return batch_data

def main():
    # PREPROCESSING
    preprocessed_path = "preprocessed.p"
    if not path.exists(preprocessed_path):
        spell_list_json = U.loadJSONData("spell_data.json")
        descs = S.Spells.spellsToSpellKeys(spell_list_json, "desc")
        data_descs = "\n\n".join(["\n".join(d) for d in descs])
        U.preprocessAndSaveData(data_descs, createLookupTable, preprocessed_path)
    int_data, vocab_to_int, int_to_vocab, tokens = U.loadPickle(preprocessed_path)

    # BUILDING THE NEURAL NETWORK
    # Check for a GPU
    if not TF.test.gpu_device_name():
        W.warn('No GPU found. Please use a GPU to train the neural network.')
    else:
        print(f"Default GPU Device: {TF.test.gpu_device_name()}")

    # Number of Epochs
    num_epochs = 200
    # Batch Size
    batch_size = 800
    # RNN Size
    rnn_size = 512
    # Embedding Dimension Size
    embed_dim = 300
    # Sequence Length
    seq_length = 15
    # Learning Rate
    learning_rate = 0.005
    # Show stats for every n number of batches
    show_every_n_batches = 10

    save_dir = "./save"

    # Build Graph
    from tensorflow_addons import seq2seq

    train_graph = TF.Graph()
    with train_graph.as_default():
        vocab_size = len(int_to_vocab)
        input_text, targets, lr = get_inputs()
        input_data_shape = TF.shape(input_text)
        cell, initial_state = get_init_cell(input_data_shape[0],
                                            rnn_size)
        logits, final_state = build_nn(cell,
                                       rnn_size,
                                       input_text,
                                       vocab_size,
                                       embed_dim)
        # Probabilities for generating words
        probs = TF.nn.softmax(logits, name='probs')
        # Loss function
        cost = seq2seq.sequence_loss(
            logits,
            targets,
            TF.ones([input_data_shape[0], input_data_shape[1]]))
        # Optimizer
        optimizer = TF.train.AdamOptimizer(lr)
        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(TF.clip_by_value(grad, -1., 1.), var)
                            for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

    # Training
    batches = get_batches(int_text, batch_size, seq_length)
    with TF.Session(graph=train_graph) as sess:
        sess.run(TF.global_variables_initializer())
        for epoch_i in range(num_epochs):
            state = sess.run(initial_state, {input_text: batches[0][0]}) 
            for batch_i, (x, y) in enumerate(batches):
                feed = {
                    input_text: x,
                    targets: y,
                    initial_state: state,
                    lr: learning_rate}
                train_loss, state, _ = sess.run([cost, final_state, train_op], feed)
                
                # Show every <show_every_n_batches> batches
                if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                    print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                        epoch_i,
                        batch_i,
                        len(batches),
                        train_loss))
                    
        # Save Model
        saver = TF.train.Saver()
        saver.save(sess, save_dir)

    # Generate Spell Description
    gen_length = 200
    prime_word = 'The'

    loaded_graph = TF.Graph()
    with TF.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = TF.train.import_meta_graph(save_dir + '.meta')
        loader.restore(sess, save_dir)
        
        # Get Tensors from loaded model
        input_text, initial_state, final_state, probs = get_tensors(loaded_graph)
        
        # Sentences generation setup
        gen_sentences = [prime_word]
        prev_state = sess.run(initial_state, {input_text: np.array([[1]])})
        
        # Generate sentences
        for n in range(gen_length):
            # Dynamic Input
            dyn_input = [[vocab_to_int[word]
                          for word in gen_sentences[-seq_length:]]]
            dyn_seq_length = len(dyn_input[0])
            
            # Get Prediction
            probabilities, prev_state = sess.run(
                [probs, final_state],
                {input_text: dyn_input, initial_state: prev_state})
            
            pred_word = pick_word(probabilities[dyn_seq_length-1], int_to_vocab)
            
            gen_sentences.append(pred_word)

        # Remove tokens
        result = ' '.join(gen_sentences)
        for key, token in tokens.items():
            ending = ' ' if key in ['\n', '(', '"'] else ''
            result = result.replace(' ' + token.lower(), key)
        #result = result.replace('\n ', '\n')
        #result = result.replace('( ', '(')
            
        print(result)
