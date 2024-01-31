from typing import Optional

import sklearn.naive_bayes as skl_nb
import sklearn.linear_model as skl_lm
import sklearn.ensemble as skl_ensemble
import lib_processing
import sklearn.model_selection as skl_ms


MAX_INSTANCES: Optional[int] = None
TRAIN_MAX_INSTANCES: Optional[int] = None
TEST_MAX_INSTANCES: Optional[int] = None

CROSS_VAL_FOLDS: int = 5


def main() -> None:
    (instances_wid, actual_labels, feature_column_names) =\
        lib_processing.preprocess_labelled_data_multifile(["dataset/dev_embedding.csv", "dataset/train_embedding.csv"])

    instances_noid = lib_processing.remove_identity_columns(instances_wid)

    logreg_model = skl_lm.LogisticRegression(max_iter=300)
    nb_model = skl_nb.GaussianNB()
    sgd_model = skl_lm.SGDClassifier(
            loss="log_loss",
            learning_rate="constant",
            eta0=0.01,
            shuffle=False)
    voting_model = skl_ensemble.VotingClassifier(estimators=[("sgd", sgd_model),
                                                             ("logreg_default", logreg_model),
                                                             ("nb_gaussian", nb_model)],
                                                 voting="hard")

    print("start kfolds")
    train_sizes, train_scores, actual_scores = skl_ms.learning_curve(
            voting_model,  # NOTE: replace with desired model
            instances_noid,
            actual_labels,
            scoring="accuracy",
            n_jobs=-1,  # -1 = maximum cpus for multi-threading
            cv=skl_ms.StratifiedKFold(10)
    )

    print("train_sizes = ")
    print(train_sizes.tolist())
    print("train_scores = ")
    print(train_scores.tolist())
    print("actual_scores = ")
    print(actual_scores.tolist())
