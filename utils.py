import matplotlib.pyplot as plt
import sklearn.metrics
import numpy as np


def train_val_test_split(X, y, test_size, val_size):
    train_idx, test_idx = sklearn.model_selection.train_test_split(range(len(y)), test_size=test_size, stratify=y)

    X_test = np.array([X[i] for i in test_idx])
    y_test = np.array([y[i] for i in test_idx])

    y_train = [y[i] for i in train_idx]

    train_idx, val_idx = sklearn.model_selection.train_test_split(train_idx, test_size=val_size/(1-test_size), stratify=y_train)

    X_train = np.array([X[i] for i in train_idx])
    y_train = np.array([y[i] for i in train_idx])
    X_val = np.array([X[i] for i in val_idx])
    y_val = np.array([y[i] for i in val_idx])


    return X_train, y_train, X_val, y_val, X_test, y_test


def plot_confusion_matrices(y, pred):
    '''plot confusion matrices with each choice of normalization'''

    if len(np.shape(pred))==2:
        pred = pred[:, 1]
    if len(np.shape(y))==2:
        y = y[:,1]

    fig, axs = plt.subplots(figsize=(18,3.5), ncols=4)

    for ax, normalize in zip(axs, ('true', 'pred', 'all', None)):

        cm = sklearn.metrics.confusion_matrix(y, pred, normalize=normalize)
        sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=('non-set', 'set')).plot(ax=ax);
        ax.set_title(f'Normalization: {normalize}');

    for ax in axs[1:]:
        ax.set_ylabel('')

    return fig


def plot_history(history, plot_attrs, val=True, **plt_kwargs):
    '''plot given attributes from training history'''

    fig, axs = plt.subplots(ncols=len(plot_attrs), **plt_kwargs)

    if not all(plot_attr in history.history for plot_attr in plot_attrs):
        raise ValueError('not all `plot_attrs` are in the history object')

    for plot_attr, ax in zip(plot_attrs, axs):
        ax.plot(history.history[plot_attr], label=plot_attr)
        if val:
            ax.plot(history.history[f'val_{plot_attr}'], label=f'val_{plot_attr}')
        ax.set_ylabel(plot_attr)
        ax.set_xlabel('epoch')
        ax.legend(loc='upper right')

    return fig

def plot_roc_pr_curves(pred_probas, y, **kwargs):
    '''create subplots fig for ROC and PR curves'''

    if len(np.shape(pred_probas))==2:
        pred = pred_probas[:, 1]
    if len(np.shape(y))==2:
        y = y[:,1]

    subplot_kwargs = {'figsize': (8,3)}
    subplot_kwargs.update(kwargs)
    fig, (ax1, ax2) = plt.subplots(ncols=2, **subplot_kwargs)
    sklearn.metrics.RocCurveDisplay.from_predictions(y, pred, ax=ax1)
    sklearn.metrics.PrecisionRecallDisplay.from_predictions(y, pred, ax=ax2)
    return fig

def print_classification_report(model, X, y, **kwargs):

    if len(np.shape(y))==2:
        y = y[:,1]

    pred = np.argmax(model(X), axis=1)

    print(sklearn.metrics.classification_report(y, pred, **kwargs))
