# (C) Daniel Strano and the Qrack contributors 2017-2025. All rights reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

import ctypes

from .weed_system import Weed
from .dtype import DType
from .weed_tensor import WeedTensor


class WeedModule:
    """Holds a Weed module loaded from disk

    Attributes:
        mid(int): Corresponding module id.
    """

    def _get_error(self):
        return Weed.weed_lib.get_error(self.mid)

    def _throw_if_error(self):
        if self._get_error() != 0:
            raise RuntimeError("Weed C++ library raised exception.")

    def __init__(self, file_path):
        byte_string = file_path.encode('utf-8')
        c_string_ptr = ctypes.c_char_p(byte_string)
        self.mid = Weed.weed_lib.load_module(c_string_ptr)
        self._throw_if_error()

    def __del__(self):
        if self.mid is not None:
            Weed.weed_lib.free_module(self.mid)
            self.mid = None

    @staticmethod
    def _int_byref(a):
        return (ctypes.c_int * len(a))(*a)

    @staticmethod
    def _ulonglong_byref(a):
        return (ctypes.c_ulonglong * len(a))(*a)

    @staticmethod
    def _longlong_byref(a):
        return (ctypes.c_longlong * len(a))(*a)

    @staticmethod
    def _double_byref(a):
        return (ctypes.c_double * len(a))(*a)

    @staticmethod
    def _complex_byref(a):
        t = [(c.real, c.imag) for c in a]
        return WeedModule._double_byref([float(item) for sublist in t for item in sublist])

    @staticmethod
    def _bool_byref(a):
        return (ctypes.c_bool * len(a))(*a)

    def forward(self, t):
        """Applies forward inference on tensor.

        Applies the loaded model to the tensor "t" as input.

        Args:
            t (WeedTensor): The Tensor on which to apply the Module

        Returns:
            The WeedTensor output from the Module

        Raises:
            RuntimeError: Weed C++ library raised an exception.
        """
        Weed.weed_lib.forward(
            self.mid,
            t.dtype,
            len(t.shape),
            WeedModule._ulonglong_byref(t.shape),
            WeedModule._ulonglong_byref(t.stride),
            WeedModule._double_byref(t.data) if t.dtype == DType.REAL else WeedModule._complex_byref(t.data)
        )
        self._throw_if_error()

        n = Weed.weed_lib.get_result_index_count(self.mid)

        shape_out = (ctypes.c_ulonglong * n)()
        stride_out = (ctypes.c_ulonglong * n)()
        Weed.weed_lib.get_result_dims(self.mid, shape_out, stride_out)

        dtype_out = DType(Weed.weed_lib.get_result_type(self.mid))

        d_size_out = Weed.weed_lib.get_result_size(self.mid)
        d_offset = Weed.weed_lib.get_result_offset(self.mid)

        data_out = (ctypes.c_double * d_size_out)() if dtype_out == DType.REAL else (ctypes.c_double * (d_size_out << 1))()
        Weed.weed_lib.get_result(self.mid, data_out)

        self._throw_if_error()

        shape_ptr = ctypes.cast(shape_out, ctypes.POINTER(ctypes.c_longlong))
        stride_ptr = ctypes.cast(stride_out, ctypes.POINTER(ctypes.c_longlong))
        double_ptr = ctypes.cast(data_out, ctypes.POINTER(ctypes.c_double))

        if dtype_out == DType.REAL:
            data = double_ptr[:d_size_out]
        else:
            data = []
            for i in range(d_size_out):
                j = i << 1
                complex_num = complex(double_ptr[j], double_ptr[j + 1])
                data.append(complex_num)

        del double_ptr
        del data_out

        return WeedTensor(data, shape_ptr[:n], stride_ptr[:n], dtype_out, d_offset)
