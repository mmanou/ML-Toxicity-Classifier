from typing import Optional
from sklearn import cluster as skl_cluster

import lib_processing

TRAIN_MAX_INSTANCES: Optional[int] = None
TEST_MAX_INSTANCES: Optional[int] = None

MAX_K: int = 50


def main() -> None:
    elbow("dataset/unlabeled_tfidf.csv")
    elbow("dataset/unlabeled_embedding.csv")


def elbow(filename: str):
    (train_instances_wid, train_feature_column_names)\
        = lib_processing.preprocess_unlabelled_data(filename, TRAIN_MAX_INSTANCES)

    print(train_instances_wid[0])

    train_instances_noid = lib_processing.remove_identity_columns(train_instances_wid)

    for k in range(1, MAX_K):
        km_model = skl_cluster.KMeans(n_clusters=k)
        km_model.fit(train_instances_noid)
        print(f"k={k}, inertia={km_model.inertia_}")


if __name__ == "__main__":
    main()

