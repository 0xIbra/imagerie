# from . import _ni_support
from imagerie_lite.ndimage import _ni_label
# from . import _nd_image
# from . import morphology
from imagerie_lite.operations import morphology
import numpy as np
import numpy


def label(input, structure=None, output=None):
    """
    Label features in an array.
    """
    input = numpy.asarray(input)
    if numpy.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    if structure is None:
        structure = morphology.generate_binary_structure(input.ndim, 1)
    structure = numpy.asarray(structure, dtype=bool)
    if structure.ndim != input.ndim:
        raise RuntimeError('structure and input must have equal rank')
    for ii in structure.shape:
        if ii != 3:
            raise ValueError('structure dimensions must be equal to 3')

    # Use 32 bits if it's large enough for this image.
    # _ni_label.label()  needs two entries for background and
    # foreground tracking
    need_64bits = input.size >= (2**31 - 2)

    if isinstance(output, numpy.ndarray):
        if output.shape != input.shape:
            raise ValueError("output shape not correct")
        caller_provided_output = True
    else:
        caller_provided_output = False
        if output is None:
            output = numpy.empty(input.shape, numpy.intp if need_64bits else numpy.int32)
        else:
            output = numpy.empty(input.shape, output)

    # handle scalars, 0-dim arrays
    if input.ndim == 0 or input.size == 0:
        if input.ndim == 0:
            # scalar
            maxlabel = 1 if (input != 0) else 0
            output[...] = maxlabel
        else:
            # 0-dim
            maxlabel = 0
        if caller_provided_output:
            return maxlabel
        else:
            return output, maxlabel

    try:
        max_label = _ni_label._label(input, structure, output)
    except _ni_label.NeedMoreBits:
        # Make another attempt with enough bits, then try to cast to the
        # new type.
        tmp_output = np.empty(input.shape, numpy.intp if need_64bits else numpy.int32)
        max_label = _ni_label._label(input, structure, tmp_output)
        output[...] = tmp_output[...]
        if not numpy.all(output == tmp_output):
            # refuse to return bad results
            raise RuntimeError("insufficient bit-depth in requested output type")

    if caller_provided_output:
        # result was written in-place
        return max_label
    else:
        return output, max_label
