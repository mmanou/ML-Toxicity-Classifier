from typing import Optional

import sklearn.naive_bayes as skl_nb
import sklearn.dummy as skl_dummy
import sklearn.linear_model as skl_lm
import sklearn.ensemble as skl_ensemble
import lib_processing

TRAIN_MAX_INSTANCES: Optional[int] = None
TEST_MAX_INSTANCES: Optional[int] = None


def main() -> None:

    print("embedding")
    run_tests("dataset/train_embedding.csv", "dataset/dev_embedding.csv")
    print("tfidf")
    run_tests("dataset/train_tfidf.csv", "dataset/dev_tfidf.csv")


def run_tests(train_file: str, test_file: str) -> None:

    (train_instances_wid, train_actual_labels, train_feature_column_names)\
        = lib_processing.preprocess_labelled_data(train_file, TRAIN_MAX_INSTANCES)
    (test_instances_wid, test_actual_labels, test_feature_column_names)\
        = lib_processing.preprocess_labelled_data(test_file, TEST_MAX_INSTANCES)

    # remove identity columns
    train_instances_noid = lib_processing.remove_identity_columns(train_instances_wid)
    test_instances_noid = lib_processing.remove_identity_columns(test_instances_wid)

    print("-- zero-R baseline")
    baseline_model = skl_dummy.DummyClassifier(strategy="prior")
    baseline_model.fit(train_instances_noid, train_actual_labels)
    baseline_predictions: list[int] = baseline_model.predict(test_instances_noid)
    lib_processing.eval_labels(baseline_predictions, test_actual_labels)

    print("-- NB")
    nb_gaussian = skl_nb.GaussianNB()
    nb_gaussian.fit(train_instances_noid, train_actual_labels)
    nb_predictions: list[int] = nb_gaussian.predict(test_instances_noid)
    lib_processing.eval_labels(nb_predictions, test_actual_labels)

    print("-- sklearn logistic_regression, iter=300")
    logreg_default = skl_lm.LogisticRegression(max_iter=300)
    logreg_default.fit(train_instances_noid, train_actual_labels)
    logreg_default_predictions: list[int] = logreg_default.predict(test_instances_noid)
    lib_processing.eval_labels(logreg_default_predictions, test_actual_labels)

    print("-- SGD Classifier default")
    sgd_model = skl_lm.SGDClassifier(
        loss="log_loss",
        shuffle=False
    )
    sgd_model.fit(train_instances_noid, train_actual_labels)
    sgd_predictions: list[int] = sgd_model.predict(test_instances_noid)
    lib_processing.eval_labels(sgd_predictions, test_actual_labels)

    print("-- SGD Classifier lrate=0.1")
    sgd_model = skl_lm.SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=0.1,
        shuffle=False)
    sgd_model.fit(train_instances_noid, train_actual_labels)
    sgd_predictions: list[int] = sgd_model.predict(test_instances_noid)
    lib_processing.eval_labels(sgd_predictions, test_actual_labels)

    print("-- SGD Classifier lrate=0.01")
    sgd_model = skl_lm.SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=0.01,
        shuffle=False)
    sgd_model.fit(train_instances_noid, train_actual_labels)
    sgd_predictions: list[int] = sgd_model.predict(test_instances_noid)
    lib_processing.eval_labels(sgd_predictions, test_actual_labels)

    print("-- Ensemble Classifier")
    voting_model = skl_ensemble.VotingClassifier(estimators=[("sgd", sgd_model),
                                                             ("logreg_default", logreg_default),
                                                             ("nb_gaussian", nb_gaussian)],
                                                 voting="hard")
    voting_model.fit(train_instances_noid, train_actual_labels)
    voting_predictions = voting_model.predict(test_instances_noid)
    lib_processing.eval_labels(voting_predictions, test_actual_labels)


if __name__ == "__main__":
    main()