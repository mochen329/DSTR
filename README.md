# DSTR

This repository provides the prediction maps used in the main benchmark comparison of DSTR, with evaluation and training utilities.
The full DSTR implementation will be released upon acceptance of the manuscript.

## Prediction maps

DSTR was trained and evaluated at 384 × 384.
The archives contain 8-bit grayscale PNG prediction maps saved at the original annotation size.
Each map uses the corresponding test image stem as its filename.

| Test set | Maps | Download |
| --- | ---: | --- |
| CAMO | 250 | [DSTR-384-CAMO.zip](https://github.com/mochen329/DSTR/releases/download/predictions-v1/DSTR-384-CAMO.zip) |
| CHAMELEON | 76 | [DSTR-384-CHAMELEON.zip](https://github.com/mochen329/DSTR/releases/download/predictions-v1/DSTR-384-CHAMELEON.zip) |
| COD10K | 2,026 | [DSTR-384-COD10K.zip](https://github.com/mochen329/DSTR/releases/download/predictions-v1/DSTR-384-COD10K.zip) |
| NC4K | 4,121 | [DSTR-384-NC4K.zip](https://github.com/mochen329/DSTR/releases/download/predictions-v1/DSTR-384-NC4K.zip) |

## Utilities

- [`tools/evaluate.py`](tools/evaluate.py) computes structure measure, adaptive E-measure, weighted F-measure, and mean absolute error from the released prediction maps and the official ground-truth masks.
- [`tools/train_utils.py`](tools/train_utils.py) provides a PyTorch training loop with gradient accumulation and optional gradient clipping. The caller supplies the model, data loader, loss function, optimizer, and AMP scaler.

To evaluate an extracted archive, install the evaluation dependencies and point the script to the two directories that contain the PNG files.

```sh
python -m pip install numpy pillow pysodmetrics
python tools/evaluate.py --predictions path/to/prediction_pngs --ground-truth path/to/gt_pngs
```
