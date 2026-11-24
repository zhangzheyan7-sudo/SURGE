# SURGE

Source code for optical–SAR registration. Install the Python dependencies with `pip install -r requirements.txt`.

## DINOv3

Place the [official DINOv3 repository](https://github.com/facebookresearch/dinov3) at `backbone/dinov3/`. Place the DINOv3 ViT-L/16 SAT-493M pretrained checkpoint at `backbone/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`. To use a different location, edit `model.dino.repo_dir` and `model.dino.weights` in the YAML configuration. These paths are relative to the repository root. The DINOv3 checkpoint initializes the feature backbone. Evaluation uses the SURGE checkpoint produced by training.

## Data

Use `data/whu_opt_sar512/` for WHU and `data/os_dataset/` for OS. Each dataset root provides `training/` and `testing/`. Each contains `image/` with `pair<ID>_1.png` (optical) and `pair<ID>_2.png` (SAR), `gt/gt_<ID>.txt`, and `flow/flow_<ID>.flo`. A `manifest.csv` in either partition can specify `sample_id`, `source_id`, `optical`, `sar`, `gt`, and `flow`, with file paths relative to the partition. Use a common `source_id` for pairs from the same source scene to keep them in the same training or validation split.

The first training run randomly creates a train/validation split and saves it locally under `splits/` for later runs. Evaluation reads `testing/`. It creates a new `test_<random>` directory under the configured evaluation directory on each invocation. Per-pair reports retain the dataset sample IDs.

## Train and evaluate

Run these commands from the repository root. Train each model before running its evaluation command.

```sh
python train.py --config configs/whu_paper_aligned.yaml --device cuda
python test_metrics.py --config configs/whu_paper_aligned.yaml --device cuda

python train.py --config configs/os_paper_aligned.yaml --device cuda
python test_metrics.py --config configs/os_paper_aligned.yaml --device cuda
```

Use `test_metrics.py --checkpoint PATH` to evaluate a particular SURGE checkpoint. `--output-dir PATH` sets the parent directory for the randomly named evaluation output folder.

For matching lines, warped SAR and checkerboard images, use the visualization entry point after training.

```sh
python visualize_registration.py --config configs/whu_paper_aligned.yaml --device cuda --count 5
python visualize_registration.py --config configs/os_paper_aligned.yaml --device cuda --count 5
```
