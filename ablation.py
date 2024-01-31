from typing import Optional
import sklearn.naive_bayes as skl_nb
import sklearn.linear_model as skl_lm

import lib_processing


TRAIN_MAX_INSTANCES: Optional[int] = None
TEST_MAX_INSTANCES: Optional[int] = None

LEFT_INDEX = 0
RIGHT_INDEX = 300


def main() -> None:
    print("tfidf ablation")
    ablation("dataset/train_tfidf.csv", "dataset/dev_tfidf.csv")


def ablation(train_file: str, test_file: str):
    (train_instances_wid, train_actual_labels, train_feature_column_names)\
        = lib_processing.preprocess_labelled_data(train_file, TRAIN_MAX_INSTANCES)
    (test_instances_wid, test_actual_labels, test_feature_column_names)\
        = lib_processing.preprocess_labelled_data(test_file, TEST_MAX_INSTANCES)

    train_instances_noid = lib_processing.remove_identity_columns(train_instances_wid)
    test_instances_noid = lib_processing.remove_identity_columns(test_instances_wid)

    print("baselines")
    print("-- naive bayes")
    nb_gaussian = skl_nb.GaussianNB()
    nb_gaussian.fit(train_instances_noid, train_actual_labels)
    nb_predictions: list[int] = nb_gaussian.predict(test_instances_noid)
    nb_eval = lib_processing.eval_labels(nb_predictions, test_actual_labels)

    print("-- sklearn logistic_regression, iter=300")
    logreg_default = skl_lm.LogisticRegression(max_iter=300)
    logreg_default.fit(train_instances_noid, train_actual_labels)
    logreg_default_predictions: list[int] = logreg_default.predict(test_instances_noid)
    logreg_eval = lib_processing.eval_labels(logreg_default_predictions, test_actual_labels)

    print("-- SGD Classifier lrate=0.01")
    sgd_model = skl_lm.SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=0.01,
        shuffle=False)
    sgd_model.fit(train_instances_noid, train_actual_labels)
    sgd_predictions: list[int] = sgd_model.predict(test_instances_noid)
    sgd_eval = lib_processing.eval_labels(sgd_predictions, test_actual_labels)

    print(f"{LEFT_INDEX} to {RIGHT_INDEX}")
    for feature_index in range(LEFT_INDEX, RIGHT_INDEX):  # modify to get specific instance ranges if running in parallel
        train_instances = []
        test_instances = []
        for inst in train_instances_noid:
            train_instances.append([x for i, x in enumerate(inst) if i != feature_index])
        for inst in test_instances_noid:
            test_instances.append([x for i, x in enumerate(inst) if i != feature_index])

        print(f"\nremove feature {feature_index}")
        print(f"naive bayes")
        nb_gaussian = skl_nb.GaussianNB()
        nb_gaussian.fit(train_instances, train_actual_labels)
        nb_predictions: list[int] = nb_gaussian.predict(test_instances)
        new_nb_eval = lib_processing.eval_labels(nb_predictions, test_actual_labels)
        if new_nb_eval["accuracy"] > nb_eval["accuracy"]:
            print("Improved Accuracy for Naive Bayes")
        if new_nb_eval["f1_score"] > nb_eval["f1_score"]:
            print("Improved f1 for Naive Bayes")
        print()

        print("-- logistic_regression, iter=300")
        logreg_default = skl_lm.LogisticRegression(max_iter=300)
        logreg_default.fit(train_instances, train_actual_labels)
        logreg_default_predictions: list[int] = logreg_default.predict(test_instances)
        new_logreg_eval = lib_processing.eval_labels(logreg_default_predictions, test_actual_labels)
        if new_logreg_eval["accuracy"] > logreg_eval["accuracy"]:
            print("Improved Accuracy for Logistic Regression")
        if new_logreg_eval["f1_score"] > logreg_eval["f1_score"]:
            print("Improved f1 for Logistic Regression")
        print()

        print("-- SGD Classifier lrate=0.01")
        sgd_model = skl_lm.SGDClassifier(
            loss="log_loss",
            learning_rate="constant",
            eta0=0.01,
            shuffle=False)
        sgd_model.fit(train_instances, train_actual_labels)
        sgd_predictions: list[int] = sgd_model.predict(test_instances)
        new_sgd_eval = lib_processing.eval_labels(sgd_predictions, test_actual_labels)
        if new_sgd_eval["accuracy"] > sgd_eval["accuracy"]:
            print("Improved Accuracy for SGD")
        if new_sgd_eval["f1_score"] > sgd_eval["f1_score"]:
            print("Improved f1 for SGD")
        print()


if __name__ == "__main__":
    main()