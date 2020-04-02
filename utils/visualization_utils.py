import matplotlib.pyplot as plt
import sklearn.metrics.roc_curve

def plot_ROC_curve(y_test, y_pred_proba, title = 'GraphNet'):
    FPR, TPR, threshold = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(FPR, TPR)
    
    plt.figure()
    plt.figure(figsize=(5,5))
    plt.plot(FPR, TPR, color='b', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Sensitivity')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid()
#     plt.savefig(title + '.png', dpi = 500)
    plt.show()  