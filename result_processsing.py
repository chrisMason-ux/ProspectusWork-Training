import pandas as pd

OUTPUT_FILE = "complete_results.csv"


def count(dataframe):
    return dataframe.shape[0]


def correct(dataframe):
    correct_count = 0

    for index, row in dataframe.iterrows():
        if row['Label'] == row['pred_label']:
            correct_count += 1

    return correct_count


def accuracy(dataframe):
    return round(correct(dataframe) / count(dataframe), 4)


def true_prediction(dataframe):
    true_prediction_count = 0

    for index, row in dataframe.iterrows():
        if row['pred_label'] == 1:
            true_prediction_count += 1

    return true_prediction_count


def true_actual(dataframe):
    true_actual_count = 0

    for index, row in dataframe.iterrows():
        if row['Label'] == 1:
            true_actual_count += 1

    return true_actual_count


def false_prediction(dataframe):
    true_prediction_count = 0

    for index, row in dataframe.iterrows():
        if row['pred_label'] == 0:
            true_prediction_count += 1

    return true_prediction_count


def false_actual(dataframe):
    true_actual_count = 0

    for index, row in dataframe.iterrows():
        if row['Label'] == 0:
            true_actual_count += 1

    return true_actual_count


def actual_ratio(dataframe):
    if false_actual(dataframe) > 0:
        return round(true_actual(dataframe) / false_actual(dataframe), 4)
    else:
        return "Infinity"


def predicted_ratio(dataframe):
    if false_prediction(dataframe) > 0:
        return round(true_prediction(dataframe) / false_prediction(dataframe), 4)
    else:
        return "Infinity"  # This can happen if the model never predicts false


def true_positive(dataframe):
    true_positive_count = 0

    for index, row in dataframe.iterrows():
        if row['Label'] == row['pred_label'] and row['pred_label'] == 1:
            true_positive_count += 1

    return round(true_positive_count / count(dataframe), 4)


def true_negative(dataframe):
    true_negative_count = 0

    for index, row in dataframe.iterrows():
        if row['Label'] == row['pred_label'] and row['pred_label'] == 0:
            true_negative_count += 1

    return round(true_negative_count / count(dataframe), 4)


def false_positive(dataframe):
    false_positive_count = 0

    for index, row in dataframe.iterrows():
        if row['Label'] != row['pred_label'] and row['pred_label'] == 1:
            false_positive_count += 1

    return round(false_positive_count / count(dataframe), 4)


def false_negative(dataframe):
    false_negative_count = 0

    for index, row in dataframe.iterrows():
        if row['Label'] != row['pred_label'] and row['pred_label'] == 0:
            false_negative_count += 1

    return round(false_negative_count / count(dataframe), 4)


def add_results(dataframe, dataset_creator_filter, category_filter):

    try:
        out_put_df = pd.read_csv(OUTPUT_FILE, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"{OUTPUT_FILE} Not Found")


    new_row = {'dataset_creator': dataset_creator_filter,
               'filtered_out': category_filter,
               'accuracy': accuracy(dataframe),
               'time': 0,  # Unused
               'count': count(dataframe),
               'actual_true': true_actual(dataframe),
               'actual_false': false_actual(dataframe),
               'actual_ratio': actual_ratio(dataframe),
               'predicted_true': true_prediction(dataframe),
               'predicted_false': false_prediction(dataframe),
               'predicted_ratio': predicted_ratio(dataframe),
               'true_positive': true_positive(dataframe),
               'true_negative': true_negative(dataframe),
               'false_positive': false_positive(dataframe),
               'false_negative': false_negative(dataframe),
               }

    out_put_df = out_put_df.append(new_row, ignore_index=False)

    out_put_df.to_csv(OUTPUT_FILE, encoding='utf-8-sig')
