import pickle

from dataset import SceneTextDataset
from east_dataset import EASTDataset
from tqdm import tqdm

train_dataset = SceneTextDataset(
    "/opt/ml/input/data/merged",
    split="merged_train",
    image_size=2048,
    crop_size=1024,
    ignore_tags=["masked", "excluded-region", "maintable", "stamp"],
)

train_data = EASTDataset(train_dataset)
for i in tqdm(range(len(train_data))):
    g = train_data.__getitem__(i)
    with open(file=f"/opt/ml/input/data/merged_preprocessed/{i}.pkl", mode="wb") as f:
        pickle.dump(g, f)
