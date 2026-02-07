# (C) Daniel Strano and the Qrack contributors 2017-2026. All rights reserved.
#
# Weed is for minimalist AI/ML inference and backprogation in the style of
# Qrack.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

import os
from ctypes import *
from sys import platform as _platform


class WeedSystem:
    def __init__(self):
        shared_lib_path = ""
        if os.environ.get("WEED_SHARED_LIB_PATH") != None:
            shared_lib_path = os.environ.get("WEED_SHARED_LIB_PATH")
        elif _platform == "win32":
            shared_lib_path = os.path.dirname(__file__) + "/weed_lib/weed_shared.dll"
        elif _platform == "darwin":
            shared_lib_path = os.path.dirname(__file__) + "/weed_lib/libweed_shared.dylib"
        else:
            shared_lib_path = os.path.dirname(__file__) + "/weed_lib/libweed_shared.so"

        try:
            self.weed_lib = CDLL(shared_lib_path)
        except Exception as e:
            if _platform == "win32":
                shared_lib_path = "C:/Program Files (x86)/Weed/bin/weed_shared.dll"
            elif _platform == "darwin":
                shared_lib_path = "/usr/local/lib/weed/libweed_shared.dylib"
            else:
                shared_lib_path = "/usr/local/lib/weed/libweed_shared.so"

            try:
                self.weed_lib = CDLL(shared_lib_path)
            except Exception as e:
                if _platform == "win32":
                    shared_lib_path = "C:/Program Files (x86)/Weed/bin/weed_shared.dll"
                elif _platform == "darwin":
                    shared_lib_path = "/usr/lib/weed/libweed_shared.dylib"
                else:
                    shared_lib_path = "/usr/lib/weed/libweed_shared.so"

                try:
                    self.weed_lib = CDLL(shared_lib_path)
                except Exception as e:
                    print(
                        "IMPORTANT: Did you remember to install OpenCL, if your Weed version was built with OpenCL?"
                    )
                    raise e

        self.weed_lib.get_error.restype = c_int
        self.weed_lib.get_error.argtypes = [c_ulonglong]

        self.weed_lib.load_module.restype = c_ulonglong
        self.weed_lib.load_module.argtypes = [c_char_p]

        self.weed_lib.free_module.restype = None
        self.weed_lib.free_module.argtypes = [c_ulonglong]

        self.weed_lib.forward.restype = None
        self.weed_lib.forward.argtypes = [
            c_ulonglong,
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong),
            POINTER(c_double)
        ]

        self.weed_lib.get_result_index_count.restype = c_ulonglong
        self.weed_lib.get_result_index_count.argtypes = [c_ulonglong]

        self.weed_lib.get_result_dims.restype = None
        self.weed_lib.get_result_dims.argtypes = [
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong)
        ]

        self.weed_lib.get_result_size.restype = c_ulonglong
        self.weed_lib.get_result_size.argtypes = [c_ulonglong]

        self.weed_lib.get_result_offset.restype = c_ulonglong
        self.weed_lib.get_result_offset.argtypes = [c_ulonglong]

        self.weed_lib.get_result_type.restype = c_ulonglong
        self.weed_lib.get_result_type.argtypes = [c_ulonglong]

        self.weed_lib.get_result.restype = None
        self.weed_lib.get_result.argtypes = [
            c_ulonglong,
            POINTER(c_double)
        ]
