2022-AdaIN-pytorch
============================
This is an unofficial Pytorch implementation of the paper, `Style Transfer with Adaptive Instance Normalization` [arxiv](https://arxiv.org/abs/1703.06868). I referred to the [official implementation](https://github.com/xunhuang1995/AdaIN-style) in Torch. I used pretrained weights of vgg19 and decoder from [naoto0804](https://github.com/naoto0804/pytorch-AdaIN).

Requirements
----------------------------
Install requirements by `$ pip install -r requirement.txt`.

* Python 3.7+
* PyTorch 1.10
* Pillow
* TorchVision
* Numpy
* imageio
* tqdm

Usage
----------------------------

### Training

The encoder uses pretrained vgg19 network. Download the [vgg19 weight](https://drive.google.com/file/d/1UcSl-Zn3byEmn15NIPXMf9zaGCKc2gfx/view?usp=sharing). The decoder is trained on MSCOCO and wikiart dataset. 
Run the script train.py
```
$ python train.py --content_dir $CONTENT_DIR --style_dir STYLE_DIR --cuda

usage: train.py [-h] [--content_dir CONTENT_DIR] [--style_dir STYLE_DIR]
                [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--resume RESUME] [--cuda]

optional arguments:
  -h, --help            show this help message and exit
  --content_dir CONTENT_DIR
                        content images folder path
  --style_dir STYLE_DIR
                        style images folder path
  --epochs EPOCHS       Number of epoch
  --batch_size BATCH_SIZE
                        Batch size
  --resume RESUME       Continue training from epoch
  --cuda                Use CUDA
```

### Test Image Style Transfer

Download [vgg19 weight](https://drive.google.com/file/d/1UcSl-Zn3byEmn15NIPXMf9zaGCKc2gfx/view?usp=sharing), [decoder weight](https://drive.google.com/file/d/18JpLtMOapA-vwBz-LRomyTl24A9GwhTF/view?usp=sharing) under the main directory.

To test basic style transfer, run the script test_image.py. Specify `--content_image`, `--style_img` to the image path, or specify `--content_dir`, `--style_dir` to iterate all images under this directory. Specify `--grid_pth` to collect all outputs in a grid image.

```
$ python test.py --content_image $IMG --style_image $STYLE --cuda

optional arguments:
  -h, --help            show this help message and exit
  --content_image CONTENT_IMAGE
                        single content image file
  --content_dir CONTENT_DIR
                        content image directory, iterate all images under this directory
  --style_image STYLE_IMAGE
                        single style image
  --style_dir STYLE_DIR
                        style image directory, iterate all images under this directory
  --decoder_weight DECODER_WEIGHT       decoder weight file (default='decoder.pth')
  --alpha {Alpha Range}
                        Alpha [0.0, 1.0] controls style transfer level
  --cuda                Use CUDA
  --grid_pth GRID_PTH
                        Specify a grid image path (default=None) if generate a grid image
                        that contains all style transferred images
```

### Test Image Interpolation Style Transfer

To test style transfer interpolation, run the script test_interpolate.py. Specify `--style_image` with multiple paths separated by comma. Specify `--interpolation_weights` to interpolate once. Specify `--grid_pth` to interpolate with different built-in weights and provide 4 style images.

```
$ python test_interpolation.py --content_image $IMG --style_image $STYLE $WEIGHT --cuda

optional arguments:
  -h, --help            show this help message and exit
  --content_image CONTENT_IMAGE
                        single content image file
  --style_image STYLE_IMAGE
                        multiple style images file separated by comma
  --decoder_weight DECODER_WEIGHT       decoder weight file (default='decoder.pth')
  --alpha {Alpha Range}
                        Alpha [0.0, 1.0] (default=1.0) controls style transfer level
  --interpolation_weights INTERPOLATION_WEIGHTS
                        Interpolation weight of each style image, separated by comma.
                        Do not specify if input grid_pth.
  --cuda                Use CUDA
  --grid_pth GRID_PTH
                        Specify a grid image path (default=None) to perform interpolation style
                        transfer multiple times with different built-in weights and generate a
                        grid image that contains all style transferred images. Provide 4 style
                        images. Do not specify if input interpolation_weights.
```

### Test Video Style Transfer

To test video style transfer, run the script test_video.py. 

```
$ python test_video.py --content_video $VID --style_image $STYLE --cuda

optional arguments:
  -h, --help            show this help message and exit
  --content_image CONTENT_IMAGE
                        single content video file
  --style_image STYLE_IMAGE
                        single style image
  --decoder_weight DECODER_WEIGHT       decoder weight file (default='decoder.pth')
  --alpha {Alpha Range}
                        Alpha [0.0, 1.0] controls style transfer level
  --cuda                Use CUDA
```

Examples
----------------------------
### Basic Style Transfer
![grid](https://github.com/media-comp/2022-AdaIN-pytorch/blob/main/examples/grid.jpg)

### Different levels of style transfer
![grid_alpha](https://github.com/media-comp/2022-AdaIN-pytorch/blob/main/examples/grid_alpha.png)

### Interpolation Style Transfer
![grid_inter](https://github.com/media-comp/2022-AdaIN-pytorch/blob/main/examples/grid_interpolation.png)

### Video Style Transfer
Original Video

https://user-images.githubusercontent.com/42717345/163805137-d7ba350b-a42e-4b91-ac2b-4916b1715153.mp4


Style Image

<img src="https://github.com/media-comp/2022-AdaIN-pytorch/blob/main/images/art/picasso_self_portrait.jpg" alt="drawing" width="200"/>

Style Transfer Video

https://user-images.githubusercontent.com/42717345/163805886-a1199a40-6032-4baf-b2d4-30e6e05b3385.mp4


References
----------------------------
* X. Huang and S. Belongie. "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.", in ICCV, 2017. [arxiv](https://arxiv.org/abs/1703.06868)
* [Original implementation in Torch](https://github.com/xunhuang1995/AdaIN-style)
* [Pretrained weights](https://github.com/naoto0804/pytorch-AdaIN)
