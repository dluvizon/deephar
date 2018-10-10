# Deep Human Action Recognition

This software is provided as a supplementary material for our CVPR'18 paper:
> 2D/3D Pose Estimation and Action Recognition using Multitask Deep Learning

![Predictions](images/preds.png)


## Warning! This is a WIP!

During the next few days I will be uploading code and weights corresponding
to the CVPR paper, which will be tagged appropriately. Meanwhile, I will still
make changes in that code.


## How to install

Please refer to the [installation guide](INSTALL.md).


## Evaluation

<!--In order to reproduce the results reported in the paper, please make sure-->
<!--that you are using the correct version by doing `git checkout v1.0-cvpr18`.-->

### 2D pose estimation on MPII

This model reached 91.2% on the test set using multi-crop and horizontal
flipping, and 89.1% on the validation set, single-crop. To reproduce results on
validation, run:
```
  python3 exp/mpii/eval_mpii_singleperson.py output/eval-mpii
```
The output will be stored in `output/eval-mpii/log.txt`.


## Citing

Please cite our paper if this software (or any part of it) or weights are
useful for you.
```
@InProceedings{Luvizon_2018_CVPR,
  author = {Luvizon, Diogo C. and Picard, David and Tabia, Hedi},
  title = {2D/3D Pose Estimation and Action Recognition Using Multitask Deep Learning},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2018}
}
```

## License

MIT License

