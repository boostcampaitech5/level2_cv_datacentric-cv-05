import json


def load_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def save_data(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file)


def check_duplicate_keys(dict1, dict2):
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    common_keys = keys1.intersection(keys2)

    if len(common_keys) == 0:
        print("두 딕셔너리 간에 겹치는 키가 없습니다.")
    else:
        print("두 딕셔너리 간에 겹치는 키가 있습니다:", common_keys)


def merge_json(json_1, json_2):
    base = json_1
    keys = list(json_2["images"].keys())
    for key in keys:
        base["images"][key] = json_2["images"][key]
    return base


def main():
    origin_file_path = "/opt/ml/input/data/medical/ufo/train.json"
    new_file_path = "/opt/ml/input/data/medical/ufo/annotation.json"

    origin_json = load_data(origin_file_path)
    new_json = load_data(new_file_path)

    check_duplicate_keys(origin_json["images"], new_json["images"])

    new_train_json = merge_json(origin_json, new_json)
    file_name = "new_train.json"

    save_data(new_train_json, "/opt/ml/input/data/medical/ufo/" + file_name)


if __name__ == "__main__":
    main()
    print("done")
