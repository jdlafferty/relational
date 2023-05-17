import numpy as np
import matplotlib.pyplot as plt

def plot_symbolic_comparison(pretrained_n, pretrained_accs, scratch_n, scratch_accs):
    import scipy.stats
    scratch_accuracy = np.mean(scratch_accs, axis=1)
    scratch_acc_sem = scipy.stats.sem(scratch_accs, axis=1)
    pretrained_accuracy = np.mean(pretrained_accs, axis=1)
    pretrained_acc_sem = scipy.stats.sem(pretrained_accs, axis=1)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(scratch_n, scratch_accuracy, label='MLP with purely symbolic input')
    ax.fill_between(scratch_n, scratch_accuracy - scratch_acc_sem,
        scratch_accuracy + scratch_acc_sem, alpha=0.5)
    ax.plot(pretrained_n, pretrained_accuracy, label='Abstractor on images, pre-learned relations')
    ax.fill_between(pretrained_n, pretrained_accuracy - pretrained_acc_sem,
        pretrained_accuracy + pretrained_acc_sem, alpha=0.5)
    ax.set_xlabel('Training Set Size')
    ax.legend()
    ax.set_ylabel('SET Classification Accuracy')
    ax.set_title("Comparison to symbolic baseline for SET task")
    ax.grid(linestyle='dashed')
    fig.savefig('set_symbolic_vs_abstractor.pdf')
    plt.show()

dat = np.load('symbolic_vs_abstractor_run0.npz')
train_sizes = dat['train_sizes']
accs = dat['accs']
symbolic_train_sizes = dat['symbolic_train_sizes']
symbolic_accs = dat['symbolic_accs']
plot_symbolic_comparison(train_sizes, accs, symbolic_train_sizes, symbolic_accs)
