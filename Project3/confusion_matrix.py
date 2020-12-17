import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as CM

def plot_confusion_matrix(Z_true, Z_pred, normalize=True, ndecimals=2,
                          title="Confusion Matrix", savename=None):
    """
    Function for making and plotting the confusion matrix of a model using
    sklearn.metrics.confusion_matrix.
    Arguments:
        Z_true (array): true observations
        Z_pred (array): predictions
        normalize (bool, optional): whether to normalize confusion matrix,
                                    defaults to True
        title (str, optional): title of plot, defaults to "Confusion Matrix"
        savename (str, optional): plot is saved under this name if provided,
                                  defaults to None
    """
    c = CM(Z_true, Z_pred)

    if normalize is True:
        c = c/np.sum(c)

    fig, ax = plt.subplots(figsize= (5, 4.5))
    vmax = 1 if normalize else c.max()
    im = ax.matshow(c, vmin=0, vmax=vmax, cmap="autumn_r")
    plt.colorbar(im)
    s = "{:0." + str(ndecimals) + "f}" if normalize else "{:d}"
    for (i, j), z in np.ndenumerate(c):
        ax.text(j, i, s.format(z), ha="center", va="center",
                fontsize=16)
    ax.set_xlabel("Predicted value", fontsize=12)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("True value", fontsize=12)
    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(top=0.84)
    if savename is not None:
        plt.savefig(f"Figures/{savename}.png", dpi=300)
    plt.show()
