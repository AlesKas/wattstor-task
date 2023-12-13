def fill_missing_data(data):
    missing = {}
    columns = [i for i in data.columns if i not in ['Time']]
    for column in columns:
        missing[column] = data[data[column].isna()].index.to_list()

    for key, value in missing.items():
        if len(value) == 0:
            continue
        start_index = value[0] - 1
        end_index = value[-1] + 1

        start_value = data[key][start_index]
        end_value = data[key][end_index]

        fill_data = float((end_value - start_value) / (len(value) + 1))
        new_value = start_value + fill_data
        for item in value:
            data.at[item, key] = new_value
            new_value += fill_data
    return data
