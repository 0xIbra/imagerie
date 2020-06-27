from __future__ import division, print_function, absolute_import
from imagerie_lite.ndimage import _ni_support
from imagerie_lite.ndimage import _nd_image
from imagerie_lite import ndimage as ndi
from warnings import warn
import numpy as np
import numpy


def generate_binary_structure(rank, connectivity):
    """
    Generate a binary structure for binary morphological operations.
    """
    if connectivity < 1:
        connectivity = 1
    if rank < 1:
        return numpy.array(True, dtype=bool)
    output = numpy.fabs(numpy.indices([3] * rank) - 1)
    output = numpy.add.reduce(output, 0)
    return output <= connectivity


def _center_is_true(structure, origin):
    structure = numpy.array(structure)
    coor = tuple([oo + ss // 2 for ss, oo in zip(structure.shape,
                                                 origin)])
    return bool(structure[coor])


def _binary_erosion(input, structure, iterations, mask, output,
                    border_value, origin, invert, brute_force):
    input = numpy.asarray(input)
    if numpy.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    if structure is None:
        structure = generate_binary_structure(input.ndim, 1)
    else:
        structure = numpy.asarray(structure, dtype=bool)
    if structure.ndim != input.ndim:
        raise RuntimeError('structure and input must have same dimensionality')
    if not structure.flags.contiguous:
        structure = structure.copy()
    if numpy.product(structure.shape, axis=0) < 1:
        raise RuntimeError('structure must not be empty')
    if mask is not None:
        mask = numpy.asarray(mask)
        if mask.shape != input.shape:
            raise RuntimeError('mask and input must have equal sizes')
    origin = _ni_support._normalize_sequence(origin, input.ndim)
    cit = _center_is_true(structure, origin)
    if isinstance(output, numpy.ndarray):
        if numpy.iscomplexobj(output):
            raise TypeError('Complex output type not supported')
    else:
        output = bool
    output = _ni_support._get_output(output, input)

    if iterations == 1:
        _nd_image.binary_erosion(input, structure, mask, output,
                                 border_value, origin, invert, cit, 0)
        return output
    elif cit and not brute_force:
        changed, coordinate_list = _nd_image.binary_erosion(
            input, structure, mask, output,
            border_value, origin, invert, cit, 1)
        structure = structure[tuple([slice(None, None, -1)] *
                                    structure.ndim)]
        for ii in range(len(origin)):
            origin[ii] = -origin[ii]
            if not structure.shape[ii] & 1:
                origin[ii] -= 1
        if mask is not None:
            mask = numpy.asarray(mask, dtype=numpy.int8)
        if not structure.flags.contiguous:
            structure = structure.copy()
        _nd_image.binary_erosion2(output, structure, mask, iterations - 1,
                                  origin, invert, coordinate_list)
        return output
    else:
        tmp_in = numpy.empty_like(input, dtype=bool)
        tmp_out = output
        if iterations >= 1 and not iterations & 1:
            tmp_in, tmp_out = tmp_out, tmp_in
        changed = _nd_image.binary_erosion(
            input, structure, mask, tmp_out,
            border_value, origin, invert, cit, 0)
        ii = 1
        while ii < iterations or (iterations < 1 and changed):
            tmp_in, tmp_out = tmp_out, tmp_in
            changed = _nd_image.binary_erosion(
                tmp_in, structure, mask, tmp_out,
                border_value, origin, invert, cit, 0)
            ii += 1
        return output


def binary_dilation(input, structure=None, iterations=1, mask=None,
                    output=None, border_value=0, origin=0,
                    brute_force=False):
    """
    Multi-dimensional binary dilation with the given structuring element.
    """
    input = numpy.asarray(input)
    if structure is None:
        structure = generate_binary_structure(input.ndim, 1)
    origin = _ni_support._normalize_sequence(origin, input.ndim)
    structure = numpy.asarray(structure)
    structure = structure[tuple([slice(None, None, -1)] *
                                structure.ndim)]
    for ii in range(len(origin)):
        origin[ii] = -origin[ii]
        if not structure.shape[ii] & 1:
            origin[ii] -= 1

    return _binary_erosion(input, structure, iterations, mask,
                           output, border_value, origin, 1, brute_force)


def binary_fill_holes(input, structure=None, output=None, origin=0):
    """
    Fill the holes in binary objects.
    """
    mask = numpy.logical_not(input)
    tmp = numpy.zeros(mask.shape, bool)
    inplace = isinstance(output, numpy.ndarray)
    if inplace:
        binary_dilation(tmp, structure, -1, mask, output, 1, origin)
        numpy.logical_not(output, output)
    else:
        output = binary_dilation(tmp, structure, -1, mask, None, 1,
                                 origin)
        numpy.logical_not(output, output)
        return output


def _check_dtype_supported(ar):
    # Should use `issubdtype` for bool below, but there's a bug in numpy 1.7
    if not (ar.dtype == bool or numpy.issubdtype(ar.dtype, numpy.integer)):
        raise TypeError("Only bool or integer image types are supported. "
                        "Got %s." % ar.dtype)


def remove_small_objects(ar, min_size=64, connectivity=1, in_place=False):
    """Remove objects smaller than the specified size.
    """
    # Raising type error if not int or bool
    _check_dtype_supported(ar)

    if in_place:
        out = ar
    else:
        out = ar.copy()

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = generate_binary_structure(ar.ndim, connectivity)
        ccs = numpy.zeros_like(ar, dtype=numpy.int32)
        ndi.label(ar, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")

    if len(component_sizes) == 2 and out.dtype != bool:
        warn("Only one label was provided to `remove_small_objects`. "
             "Did you mean to use a boolean array?")

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out
