# LEARNING CURVES: SEQUENCE-TO-SEQUENCE ABSTRACTERS

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

import tensorflow as tf

import sklearn.metrics
from sklearn.model_selection import train_test_split

import pydealer

import sys; sys.path.append('../')
import seq2seq_abstracter_models
import hand2hand
import utils

# region SETUP
utils.print_section("SET UP")

print(tf.config.list_physical_devices())
assert len(tf.config.list_physical_devices('GPU')) > 0 # check if GPU is being used


import wandb
wandb.login()

import logging
logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

project_name = 'card-sorting-abstracters-learning-curves'


def create_callbacks(monitor='loss'):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, mode='auto', restore_best_weights=True),
#         tf.keras.callbacks.ReduceLROnPlateau( monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto'),
        wandb.keras.WandbMetricsLogger(log_freq='epoch'),
        # wandb.keras.WandbModelCheckpoint(filepath='models/model_{epoch:02d}', monitor=monitor, mode='auto', save_freq='epoch')
#         wandb.keras.WandbCallback(
#             monitor=monitor, log_weights=log_weights, log_gradients=log_gradients, save_model=save_model, save_graph=True,
#             training_data=train_ds, validation_data=val_ds,
#             labels=class_names, predictions=64, compute_flops=True)
        ]
    return callbacks

from seq2seq_transformer import masked_loss, masked_accuracy

metrics = [masked_accuracy]

loss = masked_loss
opt = tf.keras.optimizers.Adam()

fit_kwargs = {'epochs': 50}


# ## Dataset

deck = hand2hand.Cards()
pydeck = pydealer.Deck()
pydeck.shuffle()


n = 10_000
batch_size = 128

hand_size = 7

BEGIN_HAND = 52 # token for 'beginning of hand'
END_HAND = 53 # token for 'end of hand'

hands = np.array(n*(hand_size+2)*[0]).reshape(n, hand_size+2)
hands_sorted = np.array(n*(hand_size+2)*[0]).reshape(n, hand_size+2)

for i in np.arange(n):
    hand = pydeck.deal(hand_size)
    if len(hand) < hand_size:
        #print('shuffling deck')
        pydeck = pydealer.Deck()
        pydeck.shuffle()
        hand = pydeck.deal(hand_size)
    source = list(deck.index_pyhand(hand))
    source.insert(0,BEGIN_HAND)
    source.append(END_HAND)
    hands[i] = np.array(source)
    deck.sort_pyhand(hand)
    target = list(deck.index_pyhand(hand))
    target.insert(0,BEGIN_HAND)
    target.append(END_HAND)
    hands_sorted[i] = np.array(target)



hands_train, hands_test, sorted_train, sorted_test = train_test_split(hands, hands_sorted, test_size=0.25)

source_train = hands_train
target_train = sorted_train[:,:-1]
labels_train = sorted_train[:,1:]

source_test = hands_test
target_test = sorted_test[:,:-1]
labels_test = sorted_test[:,1:]


def evaluate_seq2seq_model(model, print=False):
    n = len(source_test)
    output = np.zeros(n*(hand_size+2), dtype=int).reshape(n,hand_size+2)
    output[:,0] = BEGIN_HAND
    for i in range(hand_size+1):
        predictions = model((source_test, output[:, :-1]), training=False)
        predictions = predictions[:, i, :]
        predicted_id = tf.argmax(predictions, axis=-1)
        output[:,i+1] = predicted_id

    acc = (np.sum(output[:,1:] == labels_test))/np.prod(labels_test.shape)
    if print: print('per-card accuracy: %.2f%%' % (100*acc))

    return acc

def log_to_wandb(model):
    acc = evaluate_seq2seq_model(model, print=False)
    wandb.log({'test per-card accuracy': acc})


plt.style.use('seaborn-whitegrid')
def plot_learning_curve(train_sizes, accuracies):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(train_sizes, accuracies)
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Test Accuracy')
    return fig


max_train_size = len(hands_train)
train_size_step = 500
min_train_size = train_size_step
train_sizes = np.arange(min_train_size, max_train_size+1, step=train_size_step)

num_trials = 5 # num of trials per train set size

print(f'will evaluate learning curve for `train_sizes` from {min_train_size} to {max_train_size} in increments of {train_size_step}.')
print(f'will run {num_trials} trials for each of the {len(train_sizes)} training set sizes for a total of {num_trials * len(train_sizes)} trials')


def evaluate_learning_curves(create_model, group_name, 
    source_train=source_train, target_train=target_train, labels_train=labels_train,
    train_sizes=train_sizes, num_trials=num_trials):
    accuracies = []

    for train_size in tqdm(train_sizes, desc='train size'):

        train_size_accs = []
        for trial in trange(num_trials, desc='trial', leave=False):
            run = wandb.init(project=project_name, group=group_name, name=f'train size = {train_size}; trial = {trial}',
                            config={'train size': train_size, 'trial': trial})
            model = create_model()

            X_train = source_train[:train_size], target_train[:train_size]
            y_train = labels_train[:train_size]

            train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)

            history = model.fit(train_ds, verbose=0, callbacks=create_callbacks(), **fit_kwargs)

            acc = evaluate_seq2seq_model(model)
            log_to_wandb(model)
            wandb.finish(quiet=True)

            train_size_accs.append(acc)
            del model


        accuracies.append(np.mean(train_size_accs))


    return accuracies

# endregion


# region Standard Transformer
utils.print_section("STANDARD TRANSFORMER")


def create_model():
    transformer = seq2seq_abstracter_models.Transformer(
        num_layers=2, num_heads=2, dff=64, 
        input_vocab_size=54, target_vocab_size=54, embedding_dim=128)

    transformer.compile(loss=loss, optimizer=opt, metrics=metrics)
    transformer((source_train, target_train))

    return transformer


accuracies = evaluate_learning_curves(create_model, group_name='standard_transformer')
print(accuracies)

# endregion

# region Symbolic Abstracter
utils.print_section("SYMBOLIC ABSTRACTER")


def create_model():
    seq2seq_symbolic_abstracter = seq2seq_abstracter_models.Seq2SeqSymbolicAbstracter(
        num_layers=2, num_heads=2, dff=64,
        input_vocab_size=54, target_vocab_size=54, embedding_dim=128)

    seq2seq_symbolic_abstracter.compile(loss=loss, optimizer=opt, metrics=metrics)
    seq2seq_symbolic_abstracter((source_train, target_train))

    return seq2seq_symbolic_abstracter


accuracies = evaluate_learning_curves(create_model, group_name='symbolic_abstracter')
print(accuracies)

# endregion

# region Relational Abstracter
utils.print_section("RELATIONAL ABSTRACTER")

def create_model():
    seq2seq_relational_abstracter = seq2seq_abstracter_models.Seq2SeqRelationalAbstracter(
        num_layers=2, num_heads=2, dff=64, 
        input_vocab_size=54, target_vocab_size=54, embedding_dim=128)

    seq2seq_relational_abstracter.compile(loss=loss, optimizer=opt, metrics=metrics)
    seq2seq_relational_abstracter((source_train, target_train))

    return seq2seq_relational_abstracter


accuracies = evaluate_learning_curves(create_model, group_name='relational_abstracter')
print(accuracies)
# endregion


# region Sensory-Connected Abstracter
utils.print_section("SENSORY-CONNECTED ABSTRACTER")

# changing source so that it is of the same length as the target and labels
# removing the last element from the source. the last element is always the end-of-sequence token anyways.
# i think this is fine
source_train = hands_train[:,:-1]
source_test = hands_test[:,:-1]

def create_model():
    sensory_connected_abstracter = seq2seq_abstracter_models.Seq2SeqSensoryConnectedAbstracter(
        num_layers=2, num_heads=2, dff=64, 
        input_vocab_size=54, target_vocab_size=54, embedding_dim=128)

    sensory_connected_abstracter.compile(loss=loss, optimizer=opt, metrics=metrics)
    sensory_connected_abstracter((source_train, target_train))

    return sensory_connected_abstracter


accuracies = evaluate_learning_curves(create_model, group_name='sensory_connected_abstracter')
print(accuracies)

# endregion