import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn import metrics
# models

# prep
from sklearn.model_selection import train_test_split, GridSearchCV

PCT_CHANGE_THRESHOLD = 0.02


def train_gridient_boost(X, y, clf, y_type, plot_graph, cv):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    if cv:
        print(clf.best_params_)
        print(clf.best_estimator_)
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))

    predict_y = clf.predict(X_valid)
    mse = mean_squared_error(y_valid, predict_y)
    print("MSE: %.4f" % mse)
    # y_predprob = clf.predict_proba(X_train)[:, 1]
    # print("Accuracy : %.4g" % metrics.accuracy_score(y_valid, clf.predict(X_valid)))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))

    print("train score is {}".format(clf.score(X_train, y_train)))
    print("test score is {}".format(clf.score(X_valid, y_valid)))

    # #############################################################################
    # Plot training deviance
    if plot_graph:

        # compute test set deviance
        test_score = np.zeros((clf.n_estimators,), dtype=np.float64)

        for i, y_pred in enumerate(clf.staged_predict(X_valid)):
            test_score[i] = clf.loss_(y_valid, y_pred)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('Score ' + y_type)
        plt.plot(np.arange(clf.n_estimators) + 1, clf.train_score_, 'b-',
                 label='Training Set Score')
        plt.plot(np.arange(clf.n_estimators) + 1, test_score, 'r-',
                 label='Test Set Score')
        plt.legend(loc='upper right')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Score')

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


if __name__ == "__main__":
    train_df = pd.read_csv('all_reduced.csv')
    train_df = train_df.loc[np.abs(train_df['Brent_Spot_Price_pct_change']) >= PCT_CHANGE_THRESHOLD]

    feature_cols = [col for col in train_df.columns if 'sustained' not in col]
    X = train_df[feature_cols]
    y_cols = [col for col in train_df.columns if
              'sustained_average_1_to_130_days_later' in col or 'sustained_1_day_later_' in col]
    print(X.shape)
    for y_name in y_cols:
        y = train_df[y_name]

        print("-- gradient boosting")
        print(y_name)
        print("baseline is {}".format(sum(y == 1) / len(y)))

        params = {
            'n_estimators': 150,
            'max_depth': 8,
            'min_samples_split': 20,  # ~0.5-1% of total values  15 with threshold, 40 without threshold
            'min_samples_leaf': 2,  #  bigger will decrease test score
            'learning_rate': 0.01
            #'random_state':10
        }
        print(params)

        #normal test case
        gradient_boosting = ensemble.GradientBoostingClassifier(**params)
        train_gridient_boost(X, y, gradient_boosting, y_name, True, False)


        #10 fold
        kfold = KFold(n_splits=10, shuffle=True)
        results = cross_val_score(gradient_boosting, X, y, cv=kfold)
        print("10-fold cross_val score: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))



        #test parameter
        param_test1 = {'min_samples_split':range(20,100,10),'min_samples_leaf':range(1,5,1)}
        gsearch1 = GridSearchCV(estimator=ensemble.GradientBoostingClassifier(**params),param_grid=param_test1, cv=10)
        train_gridient_boost(X, y, gsearch1, y_name, False,True)
        print()

    # y = train_df['sustained_average_1_to_130_days_later']
    # print("-- gradient boosting")
    # print("sustained_average_1_to_130_days_later")
    # print("baseline is {}".format(sum(y == 1) / len(y)))
    # params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.01}
    # gradient_boosting = ensemble.GradientBoostingClassifier(**params)
    # print(gradient_boosting)
    # train_gridient_boost(X, y, gradient_boosting,"ustained_average_1_to_130_days_later")
    # print()
