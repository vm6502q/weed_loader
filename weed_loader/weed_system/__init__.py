# (C) Daniel Strano and the Qrack contributors 2017-2026. All rights reserved.
#
# Weed is for minimalist AI/ML inference and backprogation in the style of
# Qrack.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

from .weed_system import WeedSystem

# Global entry-point for Weed shared library
Weed = WeedSystem()
