from typing import Optional, List, Union

import xaby as xb
from ..layers import linear, conv2d
from ..batchnorm import batchnorm2d
from ..functions import relu6, dropout


def residual(func: xb.Fn) -> xb.Fn:
    return xb.parallel(func, xb.skip) >> xb.add


def conv_norm(
    in_f: int,
    out_f: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
    norm: Optional[xb.Fn] = None,
    activation: Optional[xb.Fn] = None,
):

    if norm is None:
        norm = batchnorm2d
    if activation is None:
        activation = relu6

    conv = conv2d(in_f, out_f, kernel_size, stride, padding, groups, bias=False)
    func = conv >> norm(out_f) >> activation
    func.name = "conv_norm"
    return func


def bottleneck(
    in_features: int,
    out_features: int,
    stride: int = 1,
    expand_ratio: Union[int, float] = 1.0,
) -> xb.Fn:
    """ Returns an Inverted Residual Bottleneck function """

    expand_f = int(round(expand_ratio * in_features))

    if expand_ratio != 1:
        expand = conv_norm(in_features, expand_f, kernel_size=1)
    else:
        expand = None

    depthwise = conv_norm(expand_f, expand_f, stride=stride, padding=1, groups=expand_f)

    transform = conv2d(
        expand_f, out_features, kernel_size=1, stride=1, padding=0, bias=False
    )

    if expand is not None:
        func = expand >> depthwise >> transform >> batchnorm2d(out_features)
    else:
        func = depthwise >> transform >> batchnorm2d(out_features)

    if in_features == out_features and stride == 1:
        func = residual(func)

    func.name = "bottleneck"

    return func


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def mobilenetv2(
    num_classes: int = 1000,
    width_mult: float = 1.0,
    round_nearest: int = 1,
    architecture: Optional[List[List[int]]] = None,
) -> xb.Fn:

    in_features = 32
    last_features = 1280

    # Architecture from the MobileNetV2 paper
    if architecture is None:
        architecture = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

    in_features = _make_divisible(in_features * width_mult, round_nearest)
    last_features = _make_divisible(last_features * max(1.0, width_mult), round_nearest)

    features = conv2d(3, in_features, stride=2, padding=1) >> batchnorm2d(in_features)

    # Add bottleneck layers
    for t, c, n, s in architecture:
        for i in range(n):
            out_features = c
            # We only change the dimensions at the first bottleneck layer of any series
            stride = s if i == 0 else 1

            features >> bottleneck(
                in_features, out_features, stride=stride, expand_ratio=t
            )

            in_features = out_features

    features = features >> conv_norm(in_features, last_features, kernel_size=1)
    features.name = "features"

    classifier = dropout(0.2) >> linear(last_features, num_classes)
    classifier.name = "classifier"

    # The mean here computes the spatial average of each feature. This is sometimes known as
    # global average pooling. It's becoming more popular in new models. I like it because it
    # avoids imposing a fixed input size for the model. You can train the model on 224x224 images
    # like normal, but then use the model for images with other sizes and it still works.
    model = features >> xb.mean(axis=(2, 3)) >> classifier
    model.name = "mobilenetv2"

    return model
