import math
import os
import os.path as osp
import random
import time
from argparse import ArgumentParser
from datetime import timedelta

import numpy as np
import torch
from detect import get_bboxes
from deteval import calc_deteval_metrics
from model import EAST
# from dataset import SceneTextDataset
# from east_dataset import EASTDataset
from precessed_dataset import PreDataset
from torch import cuda
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/medical"),
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "trained_models"),
    )

    parser.add_argument("--device", default="cuda" if cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--image_size", type=int, default=2048)
    parser.add_argument("--input_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_epoch", type=int, default=150)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument(
        "--ignore_tags",
        type=list,
        default=["masked", "excluded-region", "maintable", "stamp"],
    )

    parser.add_argument("--wandb_project", type=str, default="project")
    parser.add_argument("--wandb_name", type=str, default="name")

    parser.add_argument("--seed", type=int, default=5)

    parser.add_argument("--valid", type=bool, default=False)

    # early stopping 기준값 30
    parser.add_argument("--early_stopping", type=int, default=30)
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError("`input_size` must be a multiple of 32")

    return args


def cal_fpr(p_geo_map, p_score_map, gt_geo, gt_score):
    pred = {}
    gt = {}
    for i in tqdm(range(len(p_geo_map))):
        pred_geo_single = p_geo_map[i]
        pred_score_single = p_score_map[i]
        gt_geo_single = gt_geo[i]
        gt_score_single = gt_score[i]

        pred_poly = get_bboxes(
            pred_score_single.detach().cpu().numpy(),
            pred_geo_single.detach().cpu().numpy(),
        )
        gt_poly = get_bboxes(gt_score_single.numpy(), gt_geo_single.numpy())
        if pred_poly is None:
            print("None")
            continue
        if gt_poly is None:
            print("None")
            continue
        pred_rect = pred_poly[:, [0, 1, 4, 5]]
        gt_rect = gt_poly[:, [0, 1, 4, 5]]
        pred[str(i)] = pred_rect
        gt[str(i)] = gt_rect
    result = calc_deteval_metrics(pred, gt)
    return result["total"]


def do_training(
    data_dir,
    model_dir,
    device,
    image_size,
    input_size,
    num_workers,
    batch_size,
    learning_rate,
    max_epoch,
    save_interval,
    ignore_tags,
    wandb_project,
    wandb_name,
    seed,
    valid,
    early_stopping,
):
    seed_everything(seed)
    wandb.init(
        project=args.wandb_project, entity="boostcamp-cv5-dc", name=args.wandb_name
    )
    wandb.config.update(args)

    if valid:
        train_dataset = PreDataset("/opt/ml/input/data/pkl_split/train")
    else:
        train_dataset = PreDataset("/opt/ml/input/data/merged_preprocessed")

    train_num_batches = math.ceil(len(train_dataset) / batch_size)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    if valid:
        valid_dataset = PreDataset("/opt/ml/input/data/pkl_split/val")
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        valid_num_batches = math.ceil(len(valid_dataset) / batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[max_epoch // 2], gamma=0.1
    )

    best_val_f1 = 0
    model.train()

    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=train_num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description("[Epoch {}]".format(epoch + 1))

                loss, extra_info = model.train_step(
                    img, gt_score_map, gt_geo_map, roi_mask
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val
                pbar.update(1)
                val_dict = {
                    "Cls loss": extra_info["cls_loss"],
                    "Angle loss": extra_info["angle_loss"],
                    "IoU loss": extra_info["iou_loss"],
                }

                wandb.log(
                    {
                        "Train_cls_loss": extra_info["cls_loss"],
                        "Train_angle_loss": extra_info["angle_loss"],
                        "Train_iou_loss": extra_info["iou_loss"],
                        "lr": scheduler.optimizer.param_groups[0]["lr"],
                    }
                )

                pbar.set_postfix(val_dict)
        # 전체 train loss
        print(
            "Mean loss: {:.4f} | Elapsed time: {}".format(
                epoch_loss / train_num_batches,
                timedelta(seconds=time.time() - epoch_start),
            )
        )

        if valid and epoch > 150:
            # if valid:
            with torch.no_grad():
                model.eval()
                val_epoch_loss, epoch_start = 0, time.time()
                val_epoch_cls_loss, val_epoch_angle_loss, val_epoch_iou_loss = (
                    0.0,
                    0.0,
                    0.0,
                )
                predict_geo, predict_score, gt_geo, gt_score = "", "", "", ""
                with tqdm(total=valid_num_batches) as pbar:
                    for img, gt_score_map, gt_geo_map, roi_mask in valid_loader:
                        pbar.set_description("[Epoch {}]".format(epoch + 1))
                        loss, extra_info = model.train_step(
                            img, gt_score_map, gt_geo_map, roi_mask
                        )

                        if predict_geo == "":
                            predict_geo = extra_info["geo_map"]
                            predict_score = extra_info["score_map"]
                            gt_geo = gt_geo_map
                            gt_score = gt_score_map

                        else:
                            predict_geo = torch.cat(
                                [predict_geo, extra_info["geo_map"]], dim=0
                            )
                            predict_score = torch.cat(
                                [predict_score, extra_info["score_map"]], dim=0
                            )
                            gt_geo = torch.cat([gt_geo, gt_geo_map], dim=0)
                            gt_score = torch.cat([gt_score, gt_score_map], dim=0)

                        #

                        loss_val = loss.item()
                        val_epoch_loss += loss_val

                        val_epoch_cls_loss += extra_info["cls_loss"]
                        val_epoch_angle_loss += extra_info["angle_loss"]
                        val_epoch_iou_loss += extra_info["iou_loss"]

                        pbar.update(1)
                        val_dict = {
                            "Cls loss": extra_info["cls_loss"],
                            "Angle loss": extra_info["angle_loss"],
                            "IoU loss": extra_info["iou_loss"],
                        }
                        pbar.set_postfix(val_dict)

                print("start validating fpr")
                s_t = time.time()
                tot = cal_fpr(predict_geo, predict_score, gt_geo, gt_score)
                print("val time cost:", time.time() - s_t)
                val_epoch_loss /= valid_num_batches
                val_epoch_cls_loss /= valid_num_batches
                val_epoch_angle_loss /= valid_num_batches
                val_epoch_iou_loss /= valid_num_batches

                wandb.log(
                    {
                        "val_f1": tot["hmean"],
                        "val_precision": tot["precision"],
                        "val_recall": tot["recall"],
                        "val_epoch_loss": val_epoch_loss,
                        "val_cls_loss": val_epoch_cls_loss,
                        "val_angle_loss": val_epoch_angle_loss,
                        "val_iou_loss": val_epoch_iou_loss,
                    }
                )
                # 전체 val loss
                print(
                    "Mean loss: {:.4f} | val f1:{:.4f} | val pre:{:.4f} | val recall:{:.4f} | Elapsed time: {}".format(
                        val_epoch_loss,
                        tot["hmean"],
                        tot["precision"],
                        tot["recall"],
                        timedelta(seconds=time.time() - epoch_start),
                    )
                )

                if tot["hmean"] > best_val_f1:
                    best_val_f1 = tot["hmean"]
                    if not osp.exists(model_dir + "/" + wandb_name):
                        os.makedirs(model_dir + "/" + wandb_name)
                    ckpt_fpath = osp.join(
                        model_dir + "/" + wandb_name, "best_val_f1.pth"
                    )
                    torch.save(model.state_dict(), ckpt_fpath)
                    print(
                        f"Best val loss at epoch {epoch+1}! Saving the model to {ckpt_fpath}..."
                    )
                    count = 0
                else:  # early stopping 구현 count가 지정한 early stopping 이상이면 학습 조료
                    count += 1
                    if count >= early_stopping:
                        print(f"{early_stopping} 동안 학습 진전이 없어 종료")
                        break
        scheduler.step()

        # if (epoch + 1) % save_interval == 0:
        #     if not osp.exists(model_dir):
        #         os.makedirs(model_dir)

        #     ckpt_fpath = osp.join(model_dir, "latest.pth")
        #     torch.save(model.state_dict(), ckpt_fpath)


def main(args):
    do_training(**args.__dict__)


if __name__ == "__main__":
    args = parse_args()
    main(args)