import json
from sklearn.model_selection import KFold

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_data(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file)
        
def split_data(data, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=5)
    images = list(data['images'].keys())
    fold_data=[]
    count=0
    for train_index, val_index in kf.split(images):
        train_sets = {'images':{}}
        val_sets = {'images':{}}
        train_keys = [images[i] for i in train_index]
        val_keys = [images[i] for i in val_index]
        for train_key in train_keys:
            train_sets['images'][train_key] = data['images'][train_key]
        for val_key in val_keys:
            val_sets['images'][val_key] = data['images'][val_key]
        fold_data.append((train_sets,val_sets))
    return fold_data     

def main():
    json_file_path = '/opt/ml/input/data/medical/ufo/train.json'
    data = load_data(json_file_path)
    fold_data = split_data(data, k=5)
    for i, (train_data, val_data) in enumerate(fold_data):
        train_file_path = f'/opt/ml/input/data/medical/ufo/train_fold{i+1}.json'
        val_file_path = f'/opt/ml/input/data/medical/ufo/val_fold{i+1}.json'
        save_data(train_data, train_file_path)
        save_data(val_data, val_file_path)

if __name__ == '__main__':
    
    main()
    print('done')
