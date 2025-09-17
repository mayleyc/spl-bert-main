import time

from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from src.models.SVM.vectorization import tfidf_processing


def run_svm_classifier(train_x, train_y, test_x, config):
    train_vectors, test_vectors = tfidf_processing(train_x, test_x, config)
    # Linear SVC is generally faster than SVM(kernel=linear)
    clf_svc = LinearSVC()
    multilabel_classifier = OneVsRestClassifier(clf_svc, n_jobs=6)
    # Create grid search setup and fit it
    grid_clf = GridSearchCV(multilabel_classifier, config['SVM_GRID_PARAMS'],
                            cv=config['gridsearchCV_SPLITS'],
                            scoring='f1_macro', verbose=10)
    grid_clf.fit(train_vectors, train_y)
    print(grid_clf.best_params_)
    if config["retrain"]:
        print("Retraining on entire training split with best parameters...")
        # Retrain on "whole" data (grid_search splits similarly to before), utilizing the best parameters only
        optimized_clf = OneVsRestClassifier(LinearSVC(C=grid_clf.best_params_["estimator__C"],
                                                      max_iter=grid_clf.best_params_["estimator__max_iter"]),
                                            n_jobs=6)
        optimized_clf.fit(train_vectors, train_y)
        infer_time_start: int = time.perf_counter_ns()
        y_pred = optimized_clf.predict(test_vectors)
        infer_time_end: int = time.perf_counter_ns()
        infer_time_sample = (infer_time_end - infer_time_start) / len(test_x)
    else:
        # Same as just using the best parameters w/o retraining
        infer_time_start: int = time.perf_counter_ns()
        y_pred = grid_clf.predict(test_vectors)
        infer_time_end: int = time.perf_counter_ns()
        infer_time_sample = (infer_time_end - infer_time_start) / len(test_x)
    return y_pred, infer_time_sample
