# Image_Classification_with_2D_Convolution
Image Classification with 2D Convolutions, Deeplearning

(60000, 28, 28, 1)
(60000, 10)
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 26, 26, 132)       1320
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 13, 13, 132)       0
_________________________________________________________________
flatten_1 (Flatten)          (None, 22308)             0
_________________________________________________________________
dense_1 (Dense)              (None, 64)                1427776
_________________________________________________________________
dense_2 (Dense)              (None, 10)                650
=================================================================
Total params: 1,429,746
Trainable params: 1,429,746
Non-trainable params: 0
_________________________________________________________________
None
Train on 60000 samples, validate on 10000 samples