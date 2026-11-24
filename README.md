# SURGE

Source code for optical–SAR registration. Install the Python dependencies with `pip install -r requirements.txt`.

## DINOv3

Place the [official DINOv3 repository](https://github.com/facebookresearch/dinov3) at `backbone/dinov3/`. Place the DINOv3 ViT-L/16 SAT-493M pretrained checkpoint at `backbone/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`. To use a different location, edit `model.dino.repo_dir` and `model.dino.weights` in the YAML configuration. These paths are relative to the repository root. The DINOv3 checkpoint initializes the feature backbone. Evaluation uses the SURGE checkpoint produced by training.
