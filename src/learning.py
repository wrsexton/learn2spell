import typing as T
import tensorflow as TF
import numpy as NP
import json as J
import warnings as W

from os import path

import utility as U
import spells as S

def createLookupTables(data: str) -> T.Iterable[dict]:
    """TODO Document"""
    vocab = sorted(set(data))
    print(f"{len(vocab)} unique characters found")
    vocab_to_int = {u:i for i, u in enumerate(vocab)}
    int_to_vocab = NP.array(vocab)
    return vocab_to_int, int_to_vocab

def buildModel(vocab_size: int,
               embedding_dim: int,
               rnn_units: int,
               batch_size: int) -> TF.Model:
    """TODO Document"""
    model = TF.keras.Sequential([
        TF.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        TF.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        TF.keras.layers.Dense(vocab_size)
    ])
    return model

def generate_text(model: TF.Model,
                  start_string: str,
                  char2idx: dict,
                  idx2char: dict) -> str:
    """TODO Document"""
    # Evaluation step (generating text using the learned model)
    # Number of characters to generate
    num_generate = 1000
    
    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = TF.expand_dims(input_eval, 0)
    
    # Empty string to store our results
    text_generated = []
    
    # Low temperature results in more predictable text.
    # Higher temperature results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0
    
    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = TF.squeeze(predictions, 0)
        
        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = TF.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        
        # Pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = TF.expand_dims([predicted_id], 0)
        
        text_generated.append(idx2char[predicted_id])
        
    return (start_string + ''.join(text_generated))

def main():
    # PREPROCESSING
    preprocessed_path = "preprocessed.p"
    if not path.exists(preprocessed_path):
        spell_list_json = U.loadJSONData("spell_data.json")
        descs = S.Spells.spellsToSpellKeys(spell_list_json, "desc")
        data_descs = "\n\n".join(["\n".join(d) for d in descs])
        U.preprocessAndSaveData(data_descs, createLookupTables, preprocessed_path)
    int_data, vocab_to_int, int_to_vocab, tokens = U.loadPickle(preprocessed_path)

    # BUILDING THE NEURAL NETWORK
    # Check for a GPU
    if not TF.test.gpu_device_name():
        W.warn('No GPU found. Please use a GPU to train the neural network.')
    else:
        print(f"Default GPU Device: {TF.test.gpu_device_name()}")

    seq_length = 100
    examples_per_epoch = len(int_data)

    char_dataset = TF.data.Dataset.from_tensor_slices(NP.array(int_data))
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)
    for input_example, target_example in  dataset.take(1):
        print('Input data: ', repr(''.join(int_to_vocab[input_example.numpy()])))
        print('Target data:', repr(''.join(int_to_vocab[target_example.numpy()])))

    # Batch size
    BATCH_SIZE = 64

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    # Length of the vocabulary in chars
    vocab_size = len(sorted(set(int_data)))

    # The embedding dimension
    embedding_dim = 256

    # Number of RNN units
    rnn_units = 1024

    model = buildModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=BATCH_SIZE)

    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    print(model.summary())

    sampled_indices = TF.random.categorical(example_batch_predictions[0],
                                            num_samples=1)
    sampled_indices = TF.squeeze(sampled_indices,axis=-1).numpy()
    print(sampled_indices)
    print("Input: \n", repr("".join(int_to_vocab[input_example_batch[0]])))
    print()
    print("Next Char Predictions: \n", repr("".join(int_to_vocab[sampled_indices])))

    def loss(labels, logits):
        return TF.keras.losses.sparse_categorical_crossentropy(labels,
                                                               logits,
                                                               from_logits=True)

    example_batch_loss = loss(target_example_batch, example_batch_predictions)
    print("Prediction shape: ",
          example_batch_predictions.shape,
          " # (batch_size, sequence_length, vocab_size)")
    print("scalar_loss:      ", example_batch_loss.numpy().mean())
    model.compile(optimizer='adam', loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    if not path.exists(checkpoint_dir):
        # Name of the checkpoint files
        checkpoint_prefix = path.join(checkpoint_dir, "ckpt_{epoch}")
        
        checkpoint_callback = TF.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True)

        EPOCHS = 30
        
        history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
        print(history)

    model = buildModel(vocab_size,
                       embedding_dim,
                       rnn_units,
                       batch_size=1)
    model.load_weights(TF.train.latest_checkpoint(checkpoint_dir))
    model.build(TF.TensorShape([1, None]))
    print(model.summary())

    print("--- GENERATED SPELL DESCRIPTION BEGIN ---")
    print(generate_text(model, start_string=u"A ", char2idx=vocab_to_int, idx2char=int_to_vocab))
    print("--- GENERATED SPELL DESCRIPTION END ---")
