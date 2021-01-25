# Third party module imports
import dotenv as DE
DE.load_dotenv()
DE.load_dotenv(DE.find_dotenv('.env.defaults'))

import tensorflow as TF

# Modules included with python
import typing as T
import numpy as NP
import json as J
import pickle as P
import warnings as W
import os

# local module imports
import spells as S

# ENV VARS
EPOCHS = int(T.cast(str, os.getenv("L2S_EPOCHS")))
START_STR_DESCRIPTION = T.cast(str, os.getenv("L2S_START_STR_DESCRIPTION"))

print(f"EPOCHS set to: {EPOCHS}")
print(f"START_STR_DESCRIPTION set to: {START_STR_DESCRIPTION}")

def preprocessAndSaveData(data: str,
                          lookup_creation_func: T.Callable[[str],T.Iterable[dict]], 
                          filepath: str):
    """Preprocesses the provided data string using the provided callable,
    and writes the information to a local file at the path provided for later
    use.

    :param data: The string of data to be processed by lookup_creation_function
    :param lookup_creation_func: A function that will provide a map of characters
     to integers, and integers to characters (in that order) returned as a tuple.
    :param filepath: The location where this preprocessed data should be pickled.
    """
    vocab_to_int, int_to_vocab = lookup_creation_func(data)
    int_data = [vocab_to_int[word] for word in data]
    with open(filepath, "wb") as f:
        P.dump((int_data, vocab_to_int, int_to_vocab), f)

def loadPickle(filepath: str) -> T.Iterable[T.Any]:
    """A simple call to load a pickle file

    :param filepath: The location of the pickle file to load.

    :return: The contents of the pickle file as an object.
    """
    with open(filepath, "rb") as f:
        return P.load(f)

def createLookupTables(data: str) -> T.Iterable[dict]:
    """Processes the provided data string for machine learning.
    
    :param data: The string to be processed.

    :return: A tuple containing a dictionary of characters from the data mapped 
     to integers, and an inverse of that mapping - in that order."""
    vocab = sorted(set(data))
    print(f"{len(vocab)} unique characters found")
    vocab_to_int = {u:i for i, u in enumerate(vocab)}
    int_to_vocab = NP.array(vocab)
    return vocab_to_int, int_to_vocab

def buildModel(vocab_size: int,
               embedding_dim: int,
               rnn_units: int,
               batch_size: int) -> TF.keras.Model:
    """Creates a sequential model using Tensorflow.

    :vocab_size: Number of unique characters in the data set.
    :embedding_dim: The dimension of the dense embedding.
    :rnn_units: Dimensionality of the outer space of the GRU layer
    :batch_size: Number of training samples per batch

    :return: A Tenforflow Model object, based on the provided params.
    """
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

def generate_text(model: TF.keras.Model,
                  start_string: str,
                  char2idx: dict,
                  idx2char: dict) -> str:
    """Generate a string of text using the provided tensorflow model.

    :param model: The tensorflow model object from which to generate text.
    :param start_string: A string to seed the text generation.
    :param char2idx: A dictionary of characters mapped to integers, representing all
     of the possible characters the model can generate.
    :param idx2char: A dictionary that serves as the inverse of char2idx

    :return: The generated string, beginning with the seeded start_string
    """
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
        
        # using a categorical distribution to predict the character
        #  returned by the model
        predictions = predictions / temperature
        predicted_id = TF.random.categorical(
            predictions, num_samples=1)[-1,0].numpy()
        
        # Pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = TF.expand_dims([predicted_id], 0)
        
        text_generated.append(idx2char[predicted_id])
        
    return (start_string + ''.join(text_generated))

def main():
    # PREPROCESSING
    preprocessed_path = "preprocessed.p"
    if not os.path.exists(preprocessed_path):
        spell_list_json = S.Spells().loadJSONData("spell_data.json")
        descs = S.Spells.spellsToSpellKeys(spell_list_json, "desc")
        data_descs = "\n\n".join(["\n".join(d) for d in descs])
        preprocessAndSaveData(data_descs, createLookupTables, preprocessed_path)
    int_data, vocab_to_int, int_to_vocab = loadPickle(preprocessed_path)

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
    if not os.path.exists(checkpoint_dir):
        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
        
        checkpoint_callback = TF.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True)
        
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
    print(generate_text(model,
                        start_string=START_STR_DESCRIPTION,
                        char2idx=vocab_to_int,
                        idx2char=int_to_vocab))
    print("--- GENERATED SPELL DESCRIPTION END ---")


print(f"--- Begin main() ---")
main()
