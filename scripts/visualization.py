import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import seaborn as sns


def batch_image_peak(batch_x, batch_y):
    fig, m_axs = plt.subplots(4, 4, figsize=(16, 16))
    for (c_x, c_y, c_ax) in zip(batch_x, batch_y, m_axs.flatten()):
        c_ax.imshow(c_x[:, :, 0], cmap='bone')
        if c_y == 1:
            c_ax.set_title('Pneumonia')
        else:
            c_ax.set_title('No Pneumonia')
        c_ax.axis('off')
    plt.show()


def batch_histogram_peak(batch_x, batch_y):
    fig, m_axs = plt.subplots(4, 4, figsize=(16, 16))
    for (c_x, c_y, c_ax) in zip(batch_x, batch_y, m_axs.flatten()):
        c_ax.hist(c_x[:, :, 0].ravel(), bins=256)
        if c_y == 1:
            c_ax.set_title('Pneumonia')
        else:
            c_ax.set_title('No Pneumonia')
    plt.show()


def confusion_matrix(labels, predictions):
    cf_matrix = sklearn.metrics.confusion_matrix(labels, predictions)
    group_counts = [f"{value}" for value in cf_matrix.flatten()]
    group_percentages = [f"{(100*value):.2f}%" for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f"{v1}\n {v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt='')
    plt.ylabel('Ground Truth')
    plt.xlabel('Predictions')


def plot_auc(true_y, predicted_y):
    ## Hint: can use scikit-learn's built in functions here like roc_curve

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(true_y, predicted_y)
    roc_auc = sklearn.metrics.auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc


def precision_recall(true_y, predicted_y):
    display = sklearn.metrics.PrecisionRecallDisplay.from_predictions(true_y, predicted_y, name="Estimator")
    _ = display.ax_.set_title("Precision-Recall curve")


def plot_history(history):
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')

    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')

    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    return


def plot_history_fine(history, history_fine):
    initial_epochs = history.epoch[-1]

    acc = history.history['binary_accuracy'] + history_fine.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy'] + history_fine.history['val_binary_accuracy']

    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs - 1, initial_epochs - 1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.plot([initial_epochs - 1, initial_epochs - 1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    return
