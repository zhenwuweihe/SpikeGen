# SpikeGen: Generative Framework for Visual Spike Stream Processing

## Preparation

### Environment & Checkpoint

Please Refer to [Masked Autoregressive Model](https://github.com/LTH14/mage) for setting up the environment and download the checkpoint.

### Datasets

* **Task 0** - Pretraining：[ImageNet-1k dataset](https://image-net.org/download.php)
* **Task 1**- Conditional Video Deblur Task：[S- SDM converted GOPRO dataset](https://pan.baidu.com/s/1ZvRNF4kqVB8qe1K78hmnzg?pwd=1623#list/path=%2F)
* **Task 2** - Dense Frame Reconstruction Task：Please follw [STIR](https://pan.baidu.com/s/1ZvRNF4kqVB8qe1K78hmnzg?pwd=1623#list/path=%2F) to convert [SREDS](https://pan.baidu.com/share/init?surl=clA43FcxjOibL1zGTaU82g) dataset (password 2728)
* **Task 3** - High-Speed Scene Novel View Synthesis: [SpikeGS converted Blender dataset](https://pan.baidu.com/s/14gmAUQ78rfGs2hMzKuxEZQ?pwd=2xjx#list/path=%2F) (originated from [NeRF-Blender](https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4) and [DeblurNeRF-Blender](https://drive.google.com/drive/folders/1_TkpcJnw504ZOWmgVTD7vWqPdzbk9Wx_))

### Spike Converter

You can either use our online converter (Task 0/1) in **dataset folder** or offline converter provided by **[SpikeCV](https://github.com/spikecv/spikecv)** (Task 2/3). The reason is that the offline converter could provide more functionalities such as TFP/TFI conversion and can be used for two-stage high-speed scene novel view synthesis.

## Evaluation

For calculation of metrics (PSNR, SSIM and LPIPS), we adopt the code from [S-SDM](https://github.com/chenkang455/S-SDM/blob/master/codes/metrics.py) (Task 0/1/2) and [3DGS](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/metrics.py) (Task 3)

## Training

Please refer to the subfolder contents.

For Task 0, we provided the code of basic latent loss.

For Task 1 we provided the code of finetune pixel loss

For Task 2 we provided the code of finetune pixel loss in grayscale

For Task 3 we provided the code of finetune pixel loss, please refer to [3DGS](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/metrics.py) for 3D reconstructing the deblured views


# Demo

```
conda activate spikegen
python demo.py 
```
Or run the demo.ipynb file
