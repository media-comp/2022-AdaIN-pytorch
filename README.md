2022-AdaIN-pytorch
============================
This is an unofficial Pytorch implementation of the paper, Style Transfer with Adaptive Instance Normalization [arxiv](https://arxiv.org/abs/1703.06868). I referred to the [official implementation](https://github.com/xunhuang1995/AdaIN-style) in Torch. I used pretrained weights for vgg19 and decoder from [naoto0804](https://github.com/naoto0804/pytorch-AdaIN).

Requirement
----------------------------
* Python 3.7
* PyTorch 1.10
* Pillow
* TorchVision
* Numpy
* tqdm


Usage
----------------------------

### Training

The encoder uses pretrained vgg19 network. Download the [weight of vgg19](https://drive.google.com/file/d/1UcSl-Zn3byEmn15NIPXMf9zaGCKc2gfx/view?usp=sharing). The decoder is trained on MSCOCO and wikiart dataset. 
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

Download the [decoder weight](https://drive.google.com/file/d/18JpLtMOapA-vwBz-LRomyTl24A9GwhTF/view?usp=sharing).

To test basic style transfer, run the script test_image.py.

```
$ python test.py --content_image $IMG --style_image $STYLE --decoder_weight $WEIGHT --cuda

usage: test_style_transfer.py [-h] [--content_image CONTENT_IMAGE] [--content_dir CONTENT_DIR]
                              [--style_image STYLE_IMAGE] [--style_dir STYLE_DIR]
                              [--decoder_weight DECODER_WEIGHT] [--alpha {Alpha Range}]
                              [--cuda] [--grid_pth GRID_PTH]

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
  --decoder_weight DECODER_WEIGHT       decoder weight file
  --alpha {Alpha Range}
                        Alpha [0.0, 1.0] controls style transfer level
  --cuda                Use CUDA
  --grid_pth GRID_PTH
                        Specify a grid image path (default=None) if generate a grid image that contains all style transferred images
```

### Test Image Interpolation Style Transfer

To test style transfer interpolation, run the script test_interpolate.py.

```
$ python test_interpolation.py --content_image $IMG --style_image $STYLE --decoder_weight $WEIGHT --cuda

usage: test_style_transfer.py [-h] [--content_image CONTENT_IMAGE] [--style_image STYLE_IMAGE] 
                              [--decoder_weight DECODER_WEIGHT] [--alpha {Alpha Range}]
                              [--interpolation_weights INTERPOLATION_WEIGHTS]
                              [--cuda] [--grid_pth GRID_PTH]

optional arguments:
  -h, --help            show this help message and exit
  --content_image CONTENT_IMAGE
                        single content image file
  --style_image STYLE_IMAGE
                        multiple style images file separated by comma
  --decoder_weight DECODER_WEIGHT       decoder weight file
  --alpha {Alpha Range}
                        Alpha [0.0, 1.0] controls style transfer level
  --interpolation_weights INTERPOLATION_WEIGHTS
                        Interpolation weight of each style image, separated by comma. Perform interpolation style transfer once. Do not specify if input grid_pth.
  --cuda                Use CUDA
  --grid_pth GRID_PTH
                        Specify a grid image path (default=None) to perform interpolation style transfer multiple times with different built-in weights and generate a grid image that contains all style transferred images. Provide 4 style images. Do not specify if input interpolation_weights.
```

### Test Video Style Transfer
----------------------------

To test video style transfer, run the script test_video.py. 


```
$ python test_video.py --content_video $VID --style_image $STYLE --decoder_weight $WEIGHT --cuda

usage: test_style_transfer.py [-h] [--content_video CONTENT_VID] [--style_image STYLE_IMAGE]
                              [--decoder_weight DECODER_WEIGHT] [--alpha {Alpha Range}]
                              [--cuda]

optional arguments:
  -h, --help            show this help message and exit
  --content_image CONTENT_IMAGE
                        single content video file
  --style_image STYLE_IMAGE
                        single style image
  --decoder_weight DECODER_WEIGHT       decoder weight file
  --alpha {Alpha Range}
                        Alpha [0.0, 1.0] controls style transfer level
  --cuda                Use CUDA
```


### References
----------------------------

-[1]: X. Huang and S. Belongie. "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.", in ICCV, 2017. [arxiv](https://arxiv.org/abs/1703.06868)
-[2]: [Original implementation in Torch](https://github.com/xunhuang1995/AdaIN-style)
-[3]: [Pretrained weights](https://github.com/naoto0804/pytorch-AdaIN)