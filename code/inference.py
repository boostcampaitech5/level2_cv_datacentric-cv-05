import os
import os.path as osp
import json
from argparse import ArgumentParser
from glob import glob

import torch
import cv2
from torch import cuda
from model import EAST
from tqdm import tqdm
from deteval import calc_deteval_metrics

from detect import detect


CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', default=os.environ.get('SM_CHANNEL_EVAL', '../data/medical'))
    parser.add_argument('--model_dir', default=os.environ.get('SM_CHANNEL_MODEL', 'trained_models'))
    parser.add_argument('--output_dir', default=os.environ.get('SM_OUTPUT_DATA_DIR', 'predictions'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--input_size', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=5)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_inference(model, ckpt_fpath, data_dir, input_size, batch_size, split='test'):
    model.load_state_dict(torch.load(ckpt_fpath, map_location='cpu'))
    model.eval()

    image_fnames, by_sample_bboxes = [], []

    images = []
    for image_fpath in tqdm(glob(osp.join(data_dir, 'img/{}/*'.format(split)))):
        image_fnames.append(osp.basename(image_fpath))

        images.append(cv2.imread(image_fpath)[:, :, ::-1])
        if len(images) == batch_size:
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []

    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))

    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname] = dict(words=words_info)

    return ufo_result

def do_inference_for_val(model, input_size, split='test', data_dir = '/opt/ml/input/data/medical', val_json_fname = 'val_fold2.json'):

    model.eval()

    with open(osp.join(data_dir, "ufo", val_json_fname)) as f:
        anno = json.load(f)

    image_fnames = sorted(anno["images"].keys())
    image_dir = osp.join(data_dir, "img", "train")

    calc_f1 = []
    calc_precision = []
    calc_recall = []
    for image_fname in image_fnames:
        image = []
        image_fpath = osp.join(image_dir, image_fname)
        image.append(cv2.imread(image_fpath)[:, :, ::-1])
        pred_bboxes = detect(model, image, input_size)
        pred_bboxes_dict = {idx: bbox.tolist() for idx, bbox in enumerate(pred_bboxes)}
        gt_bboxes_bbox_keys = anno['images'][image_fname]['words'].keys()
        gt_bboxes_dict = {}
        for i in gt_bboxes_bbox_keys :
            gt_bboxes_dict[i] = anno['images'][image_fname]['words'][i]['points']
        cal = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict, transcriptions_dict=None,
                         eval_hparams=None, bbox_format='rect', verbose=False)
        calc_f1.append(cal['total']['hmean'])
        calc_precision.append(cal['total']['precision'])
        calc_recall.append(cal['total']['recall'])
    mean_f1 = sum(calc_f1) / len(calc_f1)
    mean_precision = sum(calc_precision) / len(calc_precision)
    mean_recall = sum(calc_recall) / len(calc_recall)


    return (mean_recall, mean_precision, mean_f1)


def main(args):
    # Initialize model
    model = EAST(pretrained=False).to(args.device)

    # Get paths to checkpoint files
    ckpt_fpath = osp.join(args.model_dir, 'latest.pth')

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Inference in progress')

    ufo_result = dict(images=dict())
    split_result = do_inference(model, ckpt_fpath, args.data_dir, args.input_size,
                                args.batch_size, split='test')
    ufo_result['images'].update(split_result['images'])

    output_fname = 'output.csv'
    with open(osp.join(args.output_dir, output_fname), 'w') as f:
        json.dump(ufo_result, f, indent=4)


if __name__ == '__main__':
    args = parse_args()
    main(args)
