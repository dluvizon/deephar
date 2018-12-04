# Deep Human Action Recognition

This software is provided as a supplementary material for our CVPR'18 paper:
> 2D/3D Pose Estimation and Action Recognition using Multitask Deep Learning

![Predictions](images/preds.png)

A demonstration video can be seen [here](https://www.youtube.com/watch?v=MNEZACbFA4Y&t=6s).


## Warning! This is a WIP!

During the next few days I will be uploading code and weights corresponding
to our CVPR paper, which will be tagged appropriately. Meanwhile, I will still
make changes in that code.


## How to install

Please refer to the [installation guide](INSTALL.md).


## Evaluation

<!--In order to reproduce the results reported in the paper, please make sure-->
<!--that you are using the correct version by doing `git checkout v1.0-cvpr18`.-->

### 2D pose estimation on MPII

The model trained on MPII data reached 91.2% on the test set using multi-crop
and horizontal flipping data augmentation, and 89.1% on the validation set,
single-crop.
To reproduce results on validation, do:
```
  python3 exp/mpii/eval_mpii_singleperson.py output/eval-mpii
```
The output will be stored in `output/eval-mpii/log.txt`.

### 3D pose estimation on Human3.6M

This model was trained using MPII and Human3.6M data.
Evaluation on Human3.6M is performed on the validation set.
To reproduce our results, do:
```
  python3 exp/h36m/eval_h36m.py output/eval-h36m
```
The mean per joint position error is 55.1 mm on single crop.
Note that some scores on individual activities differ from reported results
on the paper. That is because for the paper we computed scores using one frame
every 60, instead of using one frame every 64. The average score is the same.

### 2D action recognition on PennAction

For 2D action recognition, the pose estimation model was trained on mixed
data from MPII and PennAction, and the full model for action recognition was
trained and fine-tuned on PennAction only.
To reproduce our scores, do:
```
  python3 exp/pennaction/eval_penn_ar_pe_merge.py output/eval-penn
```

### 3D action recognition on NTU

For 3D action recognition, the pose estimation model was trained on mixed
data from MPII, Human3.6 and NTU, and the full model for action recognition was
trained and fine-tuned on NTU only.
To reproduce our scores, do:
```
  python3 exp/ntu/eval_ntu_ar_pe_merge.py
```


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

