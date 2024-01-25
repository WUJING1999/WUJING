import numpy as np
from sklearn.metrics import classification_report, \
    confusion_matrix, \
    cohen_kappa_score, \
    precision_score, \
    accuracy_score
import itertools
import matplotlib.pyplot as plt
import seaborn as sns


def reports(model, X_test, y_test):
    Y_pred = model.predict(X_test)
    print('y_pred',Y_pred.shape)
    y_pred = np.argmax(Y_pred, axis=1)
    target_names = ['Cunninghamia lanceolata', 'Pinus massoniana', 'Pinus elliottii', 'Eucalyptus grandis x urophylla',
                    'Eucalyptus urophylla', 'Castanopsis hystrix', 'Mytilaria laosensis', 'Camellia oleifera',
                    'Other broadleaf forest', 'Road', 'Cutting blank', 'Building land']

    classification = classification_report(np.argmax(y_test, axis=1),
                                           y_pred, target_names=target_names, digits=6)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    acc_for_each_class = precision_score(np.argmax(y_test, axis=1), y_pred, average=None)
    average_accuracy = np.mean(acc_for_each_class)
    kappa = cohen_kappa_score(np.argmax(y_test, axis=1),
                              y_pred, )
    score = model.evaluate(X_test, y_test, batch_size=32)
    Test_Loss = score[0] * 100
    Test_accuracy = score[1] * 100
    overall_acc = accuracy_score(np.argmax(y_test, axis=1), y_pred)

    return classification, confusion, Test_Loss, Test_accuracy, kappa, acc_for_each_class, average_accuracy, overall_acc


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    # sns.set(font_scale=0.6)
    fmt = '.4f' if normalize else 'd'
    thresh = 2000
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center', verticalalignment='center',
                 color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
