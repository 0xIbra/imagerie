from __future__ import division, print_function, absolute_import
from imagerie_lite._lib.six import string_types
import numpy


def _extend_mode_to_code(mode):
    """Convert an extension mode to the corresponding integer code.
    """
    if mode == 'nearest':
        return 0
    elif mode == 'wrap':
        return 1
    elif mode == 'reflect':
        return 2
    elif mode == 'mirror':
        return 3
    elif mode == 'constant':
        return 4
    else:
        raise RuntimeError('boundary mode not supported')


def _normalize_sequence(input, rank):
    """If input is a scalar, create a sequence of length equal to the
    rank by duplicating the input. If input is a sequence,
    check if its length is equal to the length of array.
    """
    is_str = isinstance(input, string_types)
    if hasattr(input, '__iter__') and not is_str:
        normalized = list(input)
        if len(normalized) != rank:
            err = "sequence argument must have length equal to input rank"
            raise RuntimeError(err)
    else:
        normalized = [input] * rank
    return normalized


def _get_output(output, input, shape=None):
    if shape is None:
        shape = input.shape
    if output is None:
        output = numpy.zeros(shape, dtype=input.dtype.name)
    elif type(output) in [type(type), type(numpy.zeros((4,)).dtype)]:
        output = numpy.zeros(shape, dtype=output)
    elif type(output) in string_types:
        output = numpy.typeDict[output]
        output = numpy.zeros(shape, dtype=output)
    elif output.shape != shape:
        raise RuntimeError("output shape not correct")
    return output


def _check_axis(axis, rank):
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError('invalid axis')
    return axis
