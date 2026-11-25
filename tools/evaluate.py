"""Evaluate saved foreground prediction maps against binary masks."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from py_sod_metrics import Emeasure, MAE, Smeasure, WeightedFmeasure


def masks_by_stem(folder):
    masks = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() == ".png":
            if path.stem in masks:
                raise ValueError(f"Duplicate mask name: {path.stem}")
            masks[path.stem] = path
    if not masks:
        raise ValueError(f"No PNG masks found in {folder}")
    return masks


def evaluate(prediction_dir, ground_truth_dir):
    predictions = masks_by_stem(prediction_dir)
    ground_truth = masks_by_stem(ground_truth_dir)
    if predictions.keys() != ground_truth.keys():
        missing = sorted(ground_truth.keys() - predictions.keys())
        extra = sorted(predictions.keys() - ground_truth.keys())
        raise ValueError(f"Mask names differ. Missing: {missing[:5]}. Extra: {extra[:5]}.")

    structure, alignment = Smeasure(), Emeasure()
    weighted_f, mae = WeightedFmeasure(), MAE()
    for stem in sorted(ground_truth):
        with Image.open(predictions[stem]) as image:
            pred = np.asarray(image.convert("L"))
        with Image.open(ground_truth[stem]) as image:
            gt = np.asarray(image.convert("L"))
        if pred.shape != gt.shape:
            raise ValueError(f"{stem}: prediction {pred.shape} and mask {gt.shape} differ")
        cutoff = 0 if gt.max() <= 1 else 127
        gt = (gt > cutoff).astype(np.uint8) * 255
        for metric in (structure, alignment, weighted_f, mae):
            metric.step(pred=pred, gt=gt)

    return {
        "n": len(predictions),
        "Sm": float(structure.get_results()["sm"]),
        "adaptiveEm": float(alignment.get_results()["em"]["adp"]),
        "wF": float(weighted_f.get_results()["wfm"]),
        "MAE": float(mae.get_results()["mae"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.predictions, args.ground_truth), indent=2))
