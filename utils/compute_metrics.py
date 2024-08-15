from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


def compute_metrics(gt,preds):
    
    print("recall :", recall_score(gt, preds, average="binary", pos_label="passport"))    
    print("precision :", precision_score(gt,preds, pos_label="passport"))
    f1_scr = f1_score(gt, preds, average='weighted')
    print("F1 score = ",f1_scr)
    print("ACCURACY score :",accuracy_score(gt,preds))
    
    confusion_mat = confusion_matrix(gt, preds)
    print("confusion_mat : \n",confusion_mat)
