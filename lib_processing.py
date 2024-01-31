from typing import Optional

import sklearn.metrics as skl_metrics


def preprocess_labelled_data_multifile(filenames: list[str],
                                       max_num_instances: Optional[int] = None
                                       ) -> (list[list[int, float]],
                                             list[int],
                                             list[str]):
    instances: list = []
    labels: list = []
    new_cols: list = []
    for fn in filenames:
        (new_instances, new_labels, new_cols) = preprocess_labelled_data(fn, max_num_instances)
        instances += new_instances
        labels += new_labels

    return instances, labels, new_cols


def preprocess_labelled_data(filename: str,
                             max_num_instances: Optional[int] = None  # optional limit on num instances processed
                             ) -> (list[list[int, float]],
                                   list[int],
                                   list[str]):
    data = open(filename, 'r', encoding='utf-8-sig').readlines()

    instances: list[list[float]] = []
    actual_labels: list[int] = []

    feature_column_names = data[0].strip(" ").split(',')  # still includes ID and Toxicity
    if feature_column_names[0] != "ID":
        raise Exception(f"Expected column 0 to be 'ID', got {feature_column_names[0]}")
    if feature_column_names[1] != "Toxicity":
        raise Exception(f"Expected column 1 to be 'Toxicity', got {feature_column_names[1]}")

    feature_column_names = feature_column_names[2:]  # remove ID and toxicity

    if max_num_instances:  # limit exists; increment to account for header
        if max_num_instances < 1:
            raise Exception(f"Expected max_num_instances >= 0 or None, got {max_num_instances}")
        max_num_instances += 1

    for line in data[1:max_num_instances]:  # ignore header
        labelled_instance = line.strip().split(',')[1:]                  # remove ID
        actual_labels.append(int(labelled_instance[0]))                  # save Toxicity label
        instances.append([float(val) for val in labelled_instance[1:]])  # ignore Toxicity label

    return instances, actual_labels, feature_column_names


def preprocess_unlabelled_data(filename: str,
                               max_num_instances: Optional[int] = None  # optional limit on num instances processed
               ) -> (list[list[int, float]],
                     list[str]):
    data = open(filename, 'r', encoding='utf-8-sig').readlines()

    instances: list[list[float]] = []

    feature_column_names = data[0].strip(" ").split(',')  # still includes ID
    if feature_column_names[0] != "ID":
        raise Exception(f"Expected column 0 to be 'ID', got {feature_column_names[0]}")
    if "Toxicity" in feature_column_names[1:]:
        raise Exception("Expected unlabelled data, found 'Toxicity' column")

    feature_column_names = feature_column_names[2:]  # remove ID and toxicity

    if max_num_instances:  # limit exists; increment to account for header
        if max_num_instances < 1:
            raise Exception(f"Expected max_num_instances >= 0 or None, got {max_num_instances}")
        max_num_instances += 1

    for line in data[1:max_num_instances]:  # ignore header
        unlabelled_instance = line.strip().split(',')[1:]              # remove ID
        instances.append([float(val) for val in unlabelled_instance])

    return instances, feature_column_names


def preprocess_final_data(filename: str,
                             max_num_instances: Optional[int] = None  # optional limit on num instances processed
                             ) -> (list[list[int, float]],
                                   list[str]):
    data = open(filename, 'r', encoding='utf-8-sig').readlines()

    instances: list[list[float]] = []

    feature_column_names = data[0].strip(" ").split(',')  # still includes ID and Toxicity
    if feature_column_names[0] != "ID":
        raise Exception(f"Expected column 0 to be 'ID', got {feature_column_names[0]}")
    if feature_column_names[1] != "Toxicity":
        raise Exception(f"Expected column 1 to be 'Toxicity', got {feature_column_names[1]}")

    feature_column_names = feature_column_names[2:]  # remove ID and toxicity

    if max_num_instances:  # limit exists; increment to account for header
        if max_num_instances < 1:
            raise Exception(f"Expected max_num_instances >= 0 or None, got {max_num_instances}")
        max_num_instances += 1

    for line in data[1:max_num_instances]:  # ignore header
        labelled_instance = line.strip().split(',')[1:]                  # remove ID
        instances.append([float(val) for val in labelled_instance[1:]])  # ignore Toxicity label

    return instances, feature_column_names


def eval_labels(predicted_labels: list,
                actual_labels: list,
                ) -> dict:
    accuracy = skl_metrics.accuracy_score(actual_labels, predicted_labels)
    precision = skl_metrics.precision_score(actual_labels, predicted_labels)
    recall = skl_metrics.recall_score(actual_labels, predicted_labels)
    (tn, fp, fn, tp) = skl_metrics.confusion_matrix(actual_labels, predicted_labels).ravel()
    f1_score = skl_metrics.f1_score(actual_labels, predicted_labels)

    print(f"accuracy: {accuracy}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    # print(f"confusion matrix:")
    # print(f"true_pos: {tp}\ntrue_neg: {tn}\nfalse_pos: {fp}\nfalse_neg: {fn}")
    print(f"f1 score: {f1_score}\n")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score,
            "true_pos": tp, "true_neg": tn, "false_pos": fp, "false_neg": fn}


def custom_confusion_matrix(predicted_labels: list[int],
                            actual_labels: list[int],
                            target_label: int
                            ) -> (int, int, int, int):

    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

    for actual, predicted in zip(actual_labels, predicted_labels):
        if actual == target_label:
            if predicted == target_label:
                true_pos += 1
            else: # predicted != target_label
                false_neg += 1
        else:  # actual != target_label
            if predicted == target_label:
                false_pos += 1
            else: # predicted != target_label
                true_neg += 1


def filter_instances_by_feature(instances: list[list],
                                feature_name: str,
                                feature_names: list[str],
                                truth_value = 1
                                ) -> list:
    column_index:int = feature_names.index(feature_name)
    return [inst for inst in instances if inst[column_index] == truth_value]


def remove_identity_columns(instances: list[list[int, float]]
                            ) -> (list[list[float]],
                                  list[int],
                                  list[str]):
    return [line[24:] for line in instances]


def remove_ablation_columns(instances: list[list[int, float]]
                            ) -> (list[list[float]],
                                  list[int],
                                  list[str]):

    bad_indexes = [98, 118, 130, 135, 159, 174, 180, 185, 193, 207, 221, 256, 257, 280]
    result = []
    for line in instances:
        result.append([val for i, val in enumerate(line) if i not in bad_indexes])

    return result
