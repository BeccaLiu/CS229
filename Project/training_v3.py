import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost
from sklearn import ensemble
from sklearn import svm
from sklearn import tree
from sklearn.metrics import mean_squared_error, confusion_matrix
# models
from sklearn.linear_model import LogisticRegression
# prep
from sklearn.model_selection import train_test_split, cross_val_score, KFold



# validation libraries

PCT_CHANGE_THRESHOLD = 0.02

def train(X, y, lm):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    lm.fit(X_train, y_train)

    try:
        print("train score: %.2f%%" % (lm.score(X_train, y_train) * 100))
        kfold = KFold(n_splits=10, shuffle=True)
        results = cross_val_score(lm, X_train, y_train, cv=kfold)
        print("10-fold cross_val score: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
        #print("test score is {}".format(lm.score(X_valid, y_valid)))
    except():
        print("no support for score")

    # y_cls_pred = lm.predict(X_valid)
    #
    # cnf_matrix = confusion_matrix(y_valid, y_cls_pred)
    # np.set_printoptions(precision=2)
    #
    # #Plot non-normalized confusion matrix
    # plot_confusion_matrix(cnf_matrix, normalize = True, classes=['Not Sustained', 'Sustained'])


def train_gradient_boost(X,y,lm):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    mse = mean_squared_error(y_valid, lm.predict(X_valid))
    print("MSE: %.4f" % mse)

    # #############################################################################
    # Plot training deviance

    # compute test set deviance
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_predict(X_valid)):
        test_score[i] = clf.loss_(y_valid, y_pred)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')

    # #############################################################################
    # Plot feature importance
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, X.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
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
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def get_corr(train_df):
    cor_column = train_df.columns.values
    corr = train_df.corr()
    corr = corr.round(3)
    corr.insert(0, "corr", cor_column)
    # corr.style.background_gradient()

    #corr.to_csv("correlation_matrix.csv", encoding='utf-8', index=False)

    sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns)
    plt.show()


if __name__ == "__main__":
    train_df = pd.read_csv('all_reduced.csv')
    train_df = train_df.loc[np.abs(train_df['Brent_Spot_Price_pct_change']) >= PCT_CHANGE_THRESHOLD]

    feature_cols = [col for col in train_df.columns if 'sustained' not in col]
    X = train_df[feature_cols]
    y = train_df['sustained_average_1_to_130_days_later']


    print("Sample size is", train_df.shape[0], "with threshold", PCT_CHANGE_THRESHOLD)
    print("sustained_average_1_to_130_days_later baseline is {}".format(sum(y == 1) / len(y)))
    print()

    #-- gradient Boosting
    print("-- Gradient Boosting")
    gb_params = {'n_estimators': 150, 'max_depth': 7, 'min_samples_split': 20, 'min_samples_leaf': 1, 'learning_rate': 0.01}
    gb_clf = ensemble.GradientBoostingClassifier(**gb_params)
    train(X, y, gb_clf)

    # -- logistic regression
    print("-- logistic regression")
    logistic_clf = LogisticRegression(solver='lbfgs', max_iter=1000, C=3)
    train(X, y, logistic_clf)

    #-- Decision tree
    print("-- Decision Tree")
    dt_clf = tree.DecisionTreeClassifier(max_depth = 8, min_samples_split = 24)
    train(X,y,dt_clf)

    #-- Bagging
    print("-- Decision Tree Bagging (Generalization of Random Forest)")
    bagging_clf = ensemble.BaggingClassifier(tree.DecisionTreeClassifier(max_depth = 8, min_samples_split = 12), n_estimators = 100, max_samples = 0.9, max_features = 0.9)
    train(X,y,bagging_clf)

    #-- Extremely Randomized Trees
    print("-- Extremely Randomized Trees")
    ertrees_clf = ensemble.ExtraTreesClassifier(n_estimators = 1000, max_depth = 12, min_samples_split = 16, max_features = 0.5)
    train(X,y,ertrees_clf)

    #-- xgboost
    print("-- xgboost")
    xgboost_params = {'n_estimators': 1000, 'max_depth': 6, 'gamma': 10,
              'learning_rate': 0.1}
    xgboost_clf = xgboost.XGBClassifier(**xgboost_params)
    train(X,y,xgboost_clf)

    #-- Hard Voting with bagging, xgboost, and logistic
    print("-- Voting Classifier (Hard) with bagging, xgboost, and logistic")
    hard_voting_clf = ensemble.VotingClassifier(estimators=[('bagging', bagging_clf),
    ('logistic', logistic_clf), ('xgboost', xgboost_clf)], voting='hard')
    train(X,y,hard_voting_clf)

    #-- Soft Voting with bagging, xgboost, and logistic
    print("-- Voting Classifier (Soft) with bagging, xgboost, and logistic")
    soft_voting_clf = ensemble.VotingClassifier(estimators=[('bagging', bagging_clf),
    ('logistic', logistic_clf), ('xgboost', xgboost_clf)], voting='soft')
    train(X,y,soft_voting_clf)

    #-- Hard Voting  with bagging, xgboost
    print("-- Voting Classifier (Hard) with bagging, xgboost")
    hard_voting_clf = ensemble.VotingClassifier(estimators=[('bagging', bagging_clf),
    ('xgboost', xgboost_clf)], voting='hard')
    train(X,y,hard_voting_clf)

    #-- Soft Voting  with bagging, xgboost
    print("-- Voting Classifier (Soft) with bagging, xgboost")
    soft_voting_clf = ensemble.VotingClassifier(estimators=[('bagging', bagging_clf),
    ('xgboost', xgboost_clf)], voting='soft')
    train(X,y,soft_voting_clf)

    #-- Hard Voting  with bagging, xgboost, extremely randomized trees
    print("-- Voting Classifier (Hard) with bagging, xgboost, ertrees")
    hard_voting_clf = ensemble.VotingClassifier(estimators=[('bagging', bagging_clf),
    ('xgboost', xgboost_clf), ('ertrees', ertrees_clf)], voting='hard')
    train(X,y,hard_voting_clf)

    #-- Soft Voting  with bagging, xgboost, extremely randomized trees
    print("-- Voting Classifier (Soft) with bagging, xgboost, ertrees")
    soft_voting_clf = ensemble.VotingClassifier(estimators=[('bagging', bagging_clf),
    ('xgboost', xgboost_clf), ('ertrees', ertrees_clf)], voting='soft')
    train(X,y,soft_voting_clf)
