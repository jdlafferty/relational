import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import argparse
import os
import datetime

import tensorflow as tf
import tensorflow_datasets as tfds

import models

import sys; sys.path.append('../'); sys.path.append('../..')
from transformer_modules import TeacherForcingAccuracy
import utils

# region SETUP

seed = 314159

# parse script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=('transformer', 'abstractor'))
parser.add_argument('--n_epochs', default=10, type=int, help='number of epochs to train each model for')
parser.add_argument('--train_size', default=-1, type=int, help='size of training set to take')
parser.add_argument('--early_stopping', default=False, type=bool, help='whether to use early stopping')
parser.add_argument('--wandb_project_name', type=str, help='W&B project name')
parser.add_argument('--run_name', default=None, type=str, help='run namee')
args = parser.parse_args()

utils.print_section("SET UP")

print(f'received the following arguments: {args}')

# check if GPU is being used
print(tf.config.list_physical_devices())
assert len(tf.config.list_physical_devices('GPU')) > 0

# set up W&B logging
os.environ["WANDB_SILENT"] = "true"
import wandb
wandb.login()

import logging
logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

wandb_project_name = args.wandb_project_name

timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
model_checkpoints_dir = f'model_checkpoints/{args.model}_{timestamp}'
os.mkdir(model_checkpoints_dir)

def create_callbacks(monitor='loss'):
    callbacks = [
#         tf.keras.callbacks.ReduceLROnPlateau( monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto'),
        wandb.keras.WandbMetricsLogger(log_freq='batch'),
        wandb.keras.WandbModelCheckpoint(filepath=model_checkpoints_dir, monitor="val_loss",
            verbose=0, save_best_only=False, save_weights_only=True, save_freq="epoch")
        ]

    if args.early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, mode='auto', restore_best_weights=True))

    return callbacks

metrics = [tf.keras.metrics.sparse_categorical_accuracy]


loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=None, name='sparse_categorical_crossentropy')
create_opt = lambda : tf.keras.optimizers.Adam(learning_rate=6e-4, beta_1=0.9, beta_2=0.995, epsilon=1e-9, clipvalue=0.1)

fit_kwargs = {'epochs': args.n_epochs}

#region Dataset
train_examples, val_examples = tfds.load(
    'math_dataset/algebra__linear_1d',
    split=['train', 'test'],
    as_supervised=True)

max_q_length = 60 # NOTE: this is for `algebera__linear_1d`
max_a_length = 4

start_char = '@'
eos_char = ';'

q_text_vectorizer = tf.keras.layers.TextVectorization(
    standardize=None,
    split='character',
    output_mode='int',
    output_sequence_length=max_q_length,
)

a_text_vectorizer = tf.keras.layers.TextVectorization(
    standardize=None,
    split='character',
    output_mode='int',
    output_sequence_length=max_a_length+2,
)

q_text_vectorizer.load_assets('vectorizer_vocab')
a_text_vectorizer.load_assets('vectorizer_vocab')

def prepend_start_token(q,a):
    source = q
    a = start_char + a + eos_char
    return q, a

source_len = max_q_length
target_len = max_a_length + 1 # max length + 2 (for start token and end token) - 1 ([:-1])
label_len = max_a_length + 1 # max length + 2 (for start token and end token) - 1 ([1:])

input_vocab_size = q_text_vectorizer.vocabulary_size()
target_vocab_size = a_text_vectorizer.vocabulary_size()

def vectorize_qa(q,a):
    return q_text_vectorizer(q), a_text_vectorizer(a)

def get_source_target_label(q,a):
    source = q
    target = a[:-1]
    label = a[1:]
    source = tf.ensure_shape(source, (source_len,))
    target = tf.ensure_shape(target, (target_len,))
    label = tf.ensure_shape(label, (label_len,))

    return (source, target), label

train_examples = train_examples.map(prepend_start_token).map(vectorize_qa).map(get_source_target_label)
val_examples = val_examples.map(prepend_start_token).map(vectorize_qa).map(get_source_target_label)

batch_size = 1024
buffer_size = 16_000
train_ds = train_examples.shuffle(buffer_size).take(args.train_size).cache().batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_examples.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
#endregion


#region build model
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=None)
create_opt = lambda : tf.keras.optimizers.Adam()
teacher_forcing_accuracy = TeacherForcingAccuracy(ignore_class=None)
if args.model == 'abstractor':
    model = models.create_abstractor(input_vocab_size, target_vocab_size)
elif args.model == 'transformer':
    model = models.create_transformer(input_vocab_size, target_vocab_size)

model.compile(loss=loss, optimizer=create_opt(), metrics=teacher_forcing_accuracy)
model(next(iter(train_ds.take(1)))[0]); # build model
print(model.summary())
#endregion

#region train model
run = wandb.init(project=wandb_project_name, group=args.model, name=args.run_name,
                config=vars(args))
history = model.fit(train_ds, validation_data=val_ds, epochs=args.n_epochs, callbacks=create_callbacks())
wandb.finish(quiet=True)
#endregion