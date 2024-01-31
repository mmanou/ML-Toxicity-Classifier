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
    embedding()
    print("tfidf")
    tfidf()
    print("dim-reduction tfidf w/unlabelled")
    tfidf_ablated()
    print("dim-reduction tfidf wout/unlabelled")
    tfidf_ablated_no_unlabelled()
    print("default tests")
    missing_ensemble_test()


def embedding():
    (train_instances_wid, train_actual_labels, train_feature_column_names)\
        = lib_processing.preprocess_labelled_data("dataset/train_embedding.csv", TRAIN_MAX_INSTANCES)
    (test_instances_wid, test_actual_labels, test_feature_column_names)\
        = lib_processing.preprocess_labelled_data("dataset/dev_embedding.csv", TEST_MAX_INSTANCES)

    (unlabelled_instances, _)\
        = lib_processing.preprocess_unlabelled_data("dataset/unlabeled_embedding.csv", TRAIN_MAX_INSTANCES)

    train_instances_noid = lib_processing.remove_identity_columns(train_instances_wid)
    test_instances_noid = lib_processing.remove_identity_columns(test_instances_wid)

    train_instances_wid = []
    test_instances_wid = []

    print("-- zero-R baseline")
    baseline_model = skl_dummy.DummyClassifier(strategy="prior")
    baseline_model.fit(train_instances_noid, train_actual_labels)
    baseline_unlabelled_predictions: list[int] = baseline_model.predict(unlabelled_instances).tolist()

    baseline_model = skl_dummy.DummyClassifier(strategy="prior")
    baseline_model.fit(train_instances_noid + unlabelled_instances,
                       train_actual_labels + baseline_unlabelled_predictions)
    baseline_predictions: list[int] = baseline_model.predict(test_instances_noid)
    lib_processing.eval_labels(baseline_predictions, test_actual_labels)

    print("-- NB")
    nb_model = skl_nb.GaussianNB()
    nb_model.fit(train_instances_noid, train_actual_labels)
    nb_unlabelled_predictions: list[int] = nb_model.predict(unlabelled_instances).tolist()
    nb_model = skl_nb.GaussianNB()
    nb_model.fit(train_instances_noid + unlabelled_instances,
                 train_actual_labels + nb_unlabelled_predictions)
    nb_predictions: list[int] = nb_model.predict(test_instances_noid)
    lib_processing.eval_labels(nb_predictions, test_actual_labels)

    print("-- logistic regression")
    logreg_model = skl_lm.LogisticRegression(max_iter=300)
    logreg_model.fit(train_instances_noid, train_actual_labels)
    logreg_unlabelled_predictions: list[int] = logreg_model.predict(unlabelled_instances).tolist()
    logreg_model = skl_lm.LogisticRegression(max_iter=300)
    logreg_model.fit(train_instances_noid + unlabelled_instances,
                     train_actual_labels + logreg_unlabelled_predictions)
    logreg_predictions: list[int] = logreg_model.predict(test_instances_noid)
    lib_processing.eval_labels(logreg_predictions, test_actual_labels)

    print("-- sgd")
    sgd_model = skl_lm.SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=0.01,
        shuffle=False)
    sgd_model.fit(train_instances_noid, train_actual_labels)
    sgd_unlabelled_predictions: list[int] = sgd_model.predict(unlabelled_instances).tolist()
    sgd_model = skl_lm.SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=0.01,
        shuffle=False)
    sgd_model.fit(train_instances_noid + unlabelled_instances,
                  train_actual_labels + sgd_unlabelled_predictions)
    sgd_predictions: list[int] = sgd_model.predict(test_instances_noid)
    lib_processing.eval_labels(sgd_predictions, test_actual_labels)

    print("-- ensemble")
    ensemble_model = skl_ensemble.VotingClassifier(estimators=[("sgd", sgd_model),
                                                             ("logreg_default", logreg_model),
                                                             ("nb_gaussian", nb_model)],
                                                   voting="hard")
    ensemble_model.fit(train_instances_noid, train_actual_labels)
    ensemble_unlabelled_predictions: list[int] = ensemble_model.predict(unlabelled_instances).tolist()
    ensemble_model = skl_ensemble.VotingClassifier(estimators=[("sgd", sgd_model),
                                                             ("logreg_default", logreg_model),
                                                             ("nb_gaussian", nb_model)],
                                                   voting="hard")
    ensemble_model.fit(train_instances_noid + unlabelled_instances,
                       train_actual_labels + ensemble_unlabelled_predictions)
    ensemble_predictions: list[int] = ensemble_model.predict(test_instances_noid)
    lib_processing.eval_labels(ensemble_predictions, test_actual_labels)


def tfidf():
    (train_instances_wid, train_actual_labels, train_feature_column_names) \
        = lib_processing.preprocess_labelled_data("dataset/train_tfidf.csv", TRAIN_MAX_INSTANCES)
    (test_instances_wid, test_actual_labels, test_feature_column_names) \
        = lib_processing.preprocess_labelled_data("dataset/dev_tfidf.csv", TEST_MAX_INSTANCES)

    (unlabelled_instances, _) \
        = lib_processing.preprocess_unlabelled_data("dataset/unlabeled_tfidf.csv", TRAIN_MAX_INSTANCES)

    train_instances_noid = lib_processing.remove_identity_columns(train_instances_wid)
    test_instances_noid = lib_processing.remove_identity_columns(test_instances_wid)

    train_instances_wid = []
    test_instances_wid = []

    print("-- zero-R baseline")
    baseline_model = skl_dummy.DummyClassifier(strategy="prior")
    baseline_model.fit(train_instances_noid, train_actual_labels)
    baseline_unlabelled_predictions: list[int] = baseline_model.predict(unlabelled_instances).tolist()

    baseline_model = skl_dummy.DummyClassifier(strategy="prior")
    baseline_model.fit(train_instances_noid + unlabelled_instances,
                       train_actual_labels + baseline_unlabelled_predictions)
    baseline_predictions: list[int] = baseline_model.predict(test_instances_noid)
    lib_processing.eval_labels(baseline_predictions, test_actual_labels)

    print("-- NB")
    nb_model = skl_nb.GaussianNB()
    nb_model.fit(train_instances_noid, train_actual_labels)
    nb_unlabelled_predictions: list[int] = nb_model.predict(unlabelled_instances).tolist()
    nb_model = skl_nb.GaussianNB()
    nb_model.fit(train_instances_noid + unlabelled_instances,
                 train_actual_labels + nb_unlabelled_predictions)
    nb_predictions: list[int] = nb_model.predict(test_instances_noid)
    lib_processing.eval_labels(nb_predictions, test_actual_labels)

    print("-- logistic regression")
    logreg_model = skl_lm.LogisticRegression(max_iter=300)
    logreg_model.fit(train_instances_noid, train_actual_labels)
    logreg_unlabelled_predictions: list[int] = logreg_model.predict(unlabelled_instances).tolist()
    logreg_model = skl_lm.LogisticRegression(max_iter=300)
    logreg_model.fit(train_instances_noid + unlabelled_instances,
                     train_actual_labels + logreg_unlabelled_predictions)
    logreg_predictions: list[int] = logreg_model.predict(test_instances_noid)
    lib_processing.eval_labels(logreg_predictions, test_actual_labels)

    print("-- sgd")
    sgd_model = skl_lm.SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=0.01,
        shuffle=False)
    sgd_model.fit(train_instances_noid, train_actual_labels)
    sgd_unlabelled_predictions: list[int] = sgd_model.predict(unlabelled_instances).tolist()
    sgd_model = skl_lm.SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=0.01,
        shuffle=False)
    sgd_model.fit(train_instances_noid + unlabelled_instances,
                  train_actual_labels + sgd_unlabelled_predictions)
    sgd_predictions: list[int] = sgd_model.predict(test_instances_noid)
    lib_processing.eval_labels(sgd_predictions, test_actual_labels)

    print("-- ensemble")
    ensemble_model = skl_ensemble.VotingClassifier(estimators=[("sgd", sgd_model),
                                                               ("logreg_default", logreg_model),
                                                               ("nb_gaussian", nb_model)],
                                                   voting="hard")
    ensemble_model.fit(train_instances_noid, train_actual_labels)
    ensemble_unlabelled_predictions: list[int] = ensemble_model.predict(unlabelled_instances).tolist()
    ensemble_model = skl_ensemble.VotingClassifier(estimators=[("sgd", sgd_model),
                                                               ("logreg_default", logreg_model),
                                                               ("nb_gaussian", nb_model)],
                                                   voting="hard")
    ensemble_model.fit(train_instances_noid + unlabelled_instances,
                       train_actual_labels + ensemble_unlabelled_predictions)
    ensemble_predictions: list[int] = ensemble_model.predict(test_instances_noid)
    lib_processing.eval_labels(ensemble_predictions, test_actual_labels)


def tfidf_ablated():
    (unlabelled_instances, _)\
        = lib_processing.preprocess_unlabelled_data("dataset/unlabeled_tfidf.csv", TRAIN_MAX_INSTANCES)

    print(unlabelled_instances[0])

    unlabelled_instances = lib_processing.remove_ablation_columns(unlabelled_instances)
    print(len(unlabelled_instances[0]))

    (train_instances_wid, train_actual_labels, train_feature_column_names) \
        = lib_processing.preprocess_labelled_data("dataset/train_tfidf.csv", TRAIN_MAX_INSTANCES)
    (test_instances_wid, test_actual_labels, test_feature_column_names) \
        = lib_processing.preprocess_labelled_data("dataset/dev_tfidf.csv", TEST_MAX_INSTANCES)

    train_instances_noid = lib_processing.remove_identity_columns(train_instances_wid)
    test_instances_noid = lib_processing.remove_identity_columns(test_instances_wid)

    train_instances_wid = []
    test_instances_wid = []

    train_instances_noid = lib_processing.remove_ablation_columns(train_instances_noid)
    test_instances_noid = lib_processing.remove_ablation_columns(test_instances_noid)

    print("-- zero-R baseline")
    baseline_model = skl_dummy.DummyClassifier(strategy="prior")
    baseline_model.fit(train_instances_noid, train_actual_labels)
    baseline_unlabelled_predictions: list[int] = baseline_model.predict(unlabelled_instances).tolist()

    baseline_model = skl_dummy.DummyClassifier(strategy="prior")
    baseline_model.fit(train_instances_noid + unlabelled_instances,
                       train_actual_labels + baseline_unlabelled_predictions)
    baseline_predictions: list[int] = baseline_model.predict(test_instances_noid)
    lib_processing.eval_labels(baseline_predictions, test_actual_labels)

    print("-- NB")
    nb_model = skl_nb.GaussianNB()
    nb_model.fit(train_instances_noid, train_actual_labels)
    nb_unlabelled_predictions: list[int] = nb_model.predict(unlabelled_instances).tolist()
    nb_model = skl_nb.GaussianNB()
    nb_model.fit(train_instances_noid + unlabelled_instances,
                 train_actual_labels + nb_unlabelled_predictions)
    nb_predictions: list[int] = nb_model.predict(test_instances_noid)
    lib_processing.eval_labels(nb_predictions, test_actual_labels)

    print("-- logistic regression")
    logreg_model = skl_lm.LogisticRegression(max_iter=300)
    logreg_model.fit(train_instances_noid, train_actual_labels)
    logreg_unlabelled_predictions: list[int] = logreg_model.predict(unlabelled_instances).tolist()
    logreg_model = skl_lm.LogisticRegression(max_iter=300)
    logreg_model.fit(train_instances_noid + unlabelled_instances,
                     train_actual_labels + logreg_unlabelled_predictions)
    logreg_predictions: list[int] = logreg_model.predict(test_instances_noid)
    lib_processing.eval_labels(logreg_predictions, test_actual_labels)

    print("-- sgd")
    sgd_model = skl_lm.SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=0.01,
        shuffle=False)
    sgd_model.fit(train_instances_noid, train_actual_labels)
    sgd_unlabelled_predictions: list[int] = sgd_model.predict(unlabelled_instances).tolist()
    sgd_model = skl_lm.SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=0.01,
        shuffle=False)
    sgd_model.fit(train_instances_noid + unlabelled_instances,
                  train_actual_labels + sgd_unlabelled_predictions)
    sgd_predictions: list[int] = sgd_model.predict(test_instances_noid)
    lib_processing.eval_labels(sgd_predictions, test_actual_labels)

    print("-- ensemble")
    ensemble_model = skl_ensemble.VotingClassifier(estimators=[("sgd", sgd_model),
                                                               ("logreg_default", logreg_model),
                                                               ("nb_gaussian", nb_model)],
                                                   voting="hard")
    ensemble_model.fit(train_instances_noid, train_actual_labels)
    ensemble_unlabelled_predictions: list[int] = ensemble_model.predict(unlabelled_instances).tolist()
    ensemble_model = skl_ensemble.VotingClassifier(estimators=[("sgd", sgd_model),
                                                               ("logreg_default", logreg_model),
                                                               ("nb_gaussian", nb_model)],
                                                   voting="hard")
    ensemble_model.fit(train_instances_noid + unlabelled_instances,
                       train_actual_labels + ensemble_unlabelled_predictions)
    ensemble_predictions: list[int] = ensemble_model.predict(test_instances_noid)
    lib_processing.eval_labels(ensemble_predictions, test_actual_labels)


def missing_ensemble_test():
    (train_instances_wid, train_actual_labels, train_feature_column_names) \
        = lib_processing.preprocess_labelled_data("dataset/train_embedding.csv", TRAIN_MAX_INSTANCES)
    (test_instances_wid, test_actual_labels, test_feature_column_names) \
        = lib_processing.preprocess_labelled_data("dataset/dev_embedding.csv", TEST_MAX_INSTANCES)

    train_instances_noid = lib_processing.remove_identity_columns(train_instances_wid)
    test_instances_noid = lib_processing.remove_identity_columns(test_instances_wid)

    train_instances_wid = []
    test_instances_wid = []

    print("-- NB")
    nb_model = skl_nb.GaussianNB()
    nb_model.fit(train_instances_noid, train_actual_labels)
    nb_predictions: list[int] = nb_model.predict(test_instances_noid)
    lib_processing.eval_labels(nb_predictions, test_actual_labels)

    print("-- logistic regression")
    logreg_model = skl_lm.LogisticRegression(max_iter=300)
    logreg_model.fit(train_instances_noid, train_actual_labels)
    logreg_predictions: list[int] = logreg_model.predict(test_instances_noid)
    lib_processing.eval_labels(logreg_predictions, test_actual_labels)

    print("-- sgd")
    sgd_model = skl_lm.SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=0.01,
        shuffle=False)
    sgd_model.fit(train_instances_noid, train_actual_labels)
    sgd_predictions: list[int] = sgd_model.predict(test_instances_noid)
    lib_processing.eval_labels(sgd_predictions, test_actual_labels)

    print("-- ensemble")
    ensemble_model = skl_ensemble.VotingClassifier(estimators=[("sgd", sgd_model),
                                                               ("logreg_default", logreg_model),
                                                               ("nb_gaussian", nb_model)],
                                                   voting="hard")
    ensemble_model.fit(train_instances_noid, train_actual_labels)
    ensemble_predictions: list[int] = ensemble_model.predict(test_instances_noid)
    lib_processing.eval_labels(ensemble_predictions, test_actual_labels)


def tfidf_ablated_no_unlabelled():
    (train_instances_wid, train_actual_labels, train_feature_column_names) \
        = lib_processing.preprocess_labelled_data("dataset/train_tfidf.csv", TRAIN_MAX_INSTANCES)
    (test_instances_wid, test_actual_labels, test_feature_column_names) \
        = lib_processing.preprocess_labelled_data("dataset/dev_tfidf.csv", TEST_MAX_INSTANCES)

    train_instances_noid = lib_processing.remove_identity_columns(train_instances_wid)
    test_instances_noid = lib_processing.remove_identity_columns(test_instances_wid)

    train_instances_wid = []
    test_instances_wid = []

    train_instances_noid = lib_processing.remove_ablation_columns(train_instances_noid)
    test_instances_noid = lib_processing.remove_ablation_columns(test_instances_noid)

    print("-- NB")
    nb_model = skl_nb.GaussianNB()
    nb_model.fit(train_instances_noid, train_actual_labels)
    nb_predictions: list[int] = nb_model.predict(test_instances_noid)
    lib_processing.eval_labels(nb_predictions, test_actual_labels)

    print("-- logistic regression")
    logreg_model = skl_lm.LogisticRegression(max_iter=300)
    logreg_model.fit(train_instances_noid, train_actual_labels)
    logreg_predictions: list[int] = logreg_model.predict(test_instances_noid)
    lib_processing.eval_labels(logreg_predictions, test_actual_labels)

    print("-- sgd")
    sgd_model = skl_lm.SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=0.01,
        shuffle=False)
    sgd_model.fit(train_instances_noid, train_actual_labels)
    sgd_predictions: list[int] = sgd_model.predict(test_instances_noid)
    lib_processing.eval_labels(sgd_predictions, test_actual_labels)

    print("-- ensemble")
    ensemble_model = skl_ensemble.VotingClassifier(estimators=[("sgd", sgd_model),
                                                               ("logreg_default", logreg_model),
                                                               ("nb_gaussian", nb_model)],
                                                   voting="hard")
    ensemble_model.fit(train_instances_noid, train_actual_labels)
    ensemble_predictions: list[int] = ensemble_model.predict(test_instances_noid)
    lib_processing.eval_labels(ensemble_predictions, test_actual_labels)


if __name__ == "__main__":
    main()
