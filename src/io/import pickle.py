import pickle
import json

def load_pickle(pickle_file_path):
    with open(pickle_file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def dump_json(data, json_file_path):
    with open(json_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def convert_pickle_to_json(pickle_file_path, json_file_path):
    data = load_pickle(pickle_file_path)
    if isinstance(data, dict):
        dump_json(data, json_file_path)
    else:
        raise ValueError("Loaded data is not a dictionary or cannot be serialized directly to JSON")

# 使用示例
convert_pickle_to_json('data.pkl', 'data.json')