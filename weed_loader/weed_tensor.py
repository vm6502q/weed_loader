# (C) Daniel Strano and the Qrack contributors 2017-2025. All rights reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

import ctypes

from .dtype import DType


class WeedTensor:
    """Container to hold tensor inputs and outputs

    This is a "dumb" container that performs no math.
    All it does is validate that (column-major) data
    dimension matches "shape" and "stride."

    Attributes:
        shape(List[int]): Shape of tensor indices
        stride(List[int]): Stride of tensor storage
        data(List): Real or complex data values
        dtype(DType): Real or complex type of data
    """

    def __init__(self, data, shape, stride, dtype=DType.REAL):
        if len(shape) != len(stride):
            raise ValueError("WeedTensor shape length must match stride length!")

        st = 1
        for i in range(len(stride)):
          if stride[i] == 0:
              continue

          if stride[i] != st:
              raise ValueError("WeedTensor shape and stride must be contiguous!")

          st *= shape[i]

        sz = 0
        for i in range(len(stride)):
            sz += (shape[i] - 1) * stride[i]

        if len(shape):
            sz = sz + 1

        if sz > len(data):
            raise ValueError("WeedTensor shape and stride do not match data length!")

        self.data = data
        self.shape = shape
        self.stride = stride
        self.dtype = dtype
