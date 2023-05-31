import json
import os.path as osp
import pickle

from dataset import SceneTextDataset
from east_dataset import EASTDataset
from tqdm import tqdm

root_dir = "/opt/ml/input/data/medical"
split = "new_train"

train_dataset = SceneTextDataset(
    root_dir,  # 데이터 저장소
    split=split,  # ufo json 파일 이름
    image_size=2048,
    crop_size=1024,
    ignore_tags=["masked", "excluded-region", "maintable", "stamp"],
)

with open(osp.join(root_dir, "ufo/{}.json".format(split)), "r") as f:
    anno = json.load(f)
image_fnames = sorted(anno["images"].keys())


train_data = EASTDataset(train_dataset)
for i in tqdm(range(len(train_data))):
    g = train_data.__getitem__(i)
    with open(
        file=f"/opt/ml/input/data/aa/{image_fnames[i]}.pkl", mode="wb"
    ) as f:  # 저장경로
        pickle.dump(g, f)
