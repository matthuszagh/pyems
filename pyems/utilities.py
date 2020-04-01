import sys
from bisect import bisect_left
from typing import List
import numpy as np
from CSXCAD.CSXCAD import ContinuousStructure
from CSXCAD.CSPrimitives import CSPrimitives
from CSXCAD.CSTransform import CSTransform


# TODO should set max precision instead of precision list. Precision
# should be computed automatically based on the last digit that
# differs between each value.
def print_table(
    data: np.array, col_names: List[str], prec: List[int], out_file=sys.stdout,
) -> None:
    """
    Data is multidimensional list, where each inner list corresponds
    to a column.
    """
    extra_space = 3
    data = np.array(data)
    col_widths = [
        int(
            _val_digits(np.amax(np.absolute(data[col])))
            + prec[col]
            + 2
            + extra_space
        )
        for col in range(len(col_names))
    ]
    for i, col in enumerate(col_names):
        out_file.write("{:{width}}".format(col, width=col_widths[i]))
    out_file.write("\n")

    data = data.T
    for row in data:
        for i, val in enumerate(row):
            out_file.write(
                "{:<{width}.{prec}f}".format(
                    val, width=col_widths[i], prec=prec[i]
                )
            )
        out_file.write("\n")


def _val_digits(val: float) -> int:
    """
    Compute the number of decimal digits needed to display the
    integral portion of a value.
    """
    # assume negative for simplicity
    extra_digits = 2

    if val < 10:
        return extra_digits + 1

    return int(np.log10(val)) + extra_digits


def array_index(val, arr) -> int:
    """
    Return the index of the closest array value to a given value.

    :param val: The value for which the closest index is desired.
    :param arr: The array from which the index is computed.

    :returns: The array index whose corresponding value is nearest the
              given value.
    """
    lbound_idx = bisect_left(arr, val)
    if lbound_idx == len(arr):
        return len(arr) - 1
    lbound = arr[lbound_idx]
    if lbound_idx == len(arr):
        return lbound_idx

    ubound_idx = lbound_idx + 1
    ubound = arr[ubound_idx]

    if val - lbound < ubound - val:
        return lbound_idx
    else:
        return ubound_idx


def sort_table_by_col(arr: np.array, col: int = 0):
    """
    Sort a 2D numpy array in ascending order by column index.
    """
    return arr[np.argsort(arr[:, col])]


def table_insertion_idx(val, arr: np.array, col: int = 0):
    """
    Find the insertion index of a value for a sorted 2D numpy array.
    """
    return np.searchsorted(arr[:, col], val)


def interp_lin(xval, xlow, xhigh, ylow, yhigh):
    """
    Get the linear-interpolated y-value for a given x-value between x
    bounds.

    :param xval: The x-value for which you want the y-value.
    :param xlow: The lower-bound x-value.
    :param xhigh: The upper-bound x-value.
    :param ylow: The lower-bound y-value.
    :param yhigh: The upper-bound y-value.
    """
    if xval < xlow or xval > xhigh:
        raise ValueError("xval must be between xlow and xhigh")

    dy = (yhigh - ylow) / (xhigh - xlow)
    dx = xval - xlow
    return ylow + (dy * dx)


def table_interp_val(
    arr: np.array, target_col, sel_val, sel_col: int = 0, permit_outside=False
):
    """
    Get the interpolated column value in a table.

    :param arr: The sorted 2D numpy array.
    :param target_col: Column corresponding to the desired return
        value.
    :param sel_val: Value of the selection column for the desired
        target column.
    :param sel_col: Column index of the selection column.
    :param permit_outside: If True, return lower or upper bound value
        if sel_val is outside table bounds.
    """
    arr = np.array(arr)
    if permit_outside:
        if sel_val < arr[0][sel_col]:
            return arr[0][target_col]
        if sel_val > arr[-1][sel_col]:
            return arr[-1][target_col]

    if sel_val == arr[0][sel_col]:
        return arr[0][target_col]
    if sel_val == arr[-1][sel_col]:
        return arr[-1][target_col]

    ins_idx = table_insertion_idx(sel_val, arr, sel_col)
    xlow = arr[ins_idx - 1][sel_col]
    xhigh = arr[ins_idx][sel_col]
    ylow = arr[ins_idx - 1][target_col]
    yhigh = arr[ins_idx][target_col]

    return interp_lin(sel_val, xlow, xhigh, ylow, yhigh)


def max_priority() -> int:
    """
    Priority that won't be overriden.

    :returns: highest priority.
    """
    return 999


def get_unit(csx: ContinuousStructure) -> float:
    """
    """
    return csx.GetGrid().GetDeltaUnit()


def apply_transform(
    prim: CSPrimitives, transform: CSTransform = None, replace: bool = False
) -> None:
    """
    Apply a transformation to a primitive.  This allows you to pass
    around and apply CSTransform objects directly, rather than having
    to set all transforms via prim.AddTransform.

    :param prim: The primitive to which the transform should be
        applied.
    :param transform: The transformation to apply.
    :param replace: Replace any transforms previously applied to the
        primitive.  Defaults to False, meaning any transforms applied
        here will be applied on top of any existing transforms already
        applied.
    """
    if transform is not None:
        tr = prim.GetTransform()
        concatenate = not replace
        tr.SetMatrix(transform.GetMatrix(), concatenate)


def append_transform(tr1: CSTransform, tr2: CSTransform) -> CSTransform:
    """
    Append two transforms.

    :param tr1: First transform.
    :param tr2: Transform to append to first transform.  The first
        transform will be applied to the primitive first, followed by
        this transform.

    :returns: Combined transform.
    """
    if tr1 is None and tr2 is None:
        return None

    tr_ret = CSTransform()
    if tr1 is not None:
        tr_ret.SetMatrix(tr1.GetMatrix(), True)
    if tr2 is not None:
        tr_ret.SetMatrix(tr2.GetMatrix(), True)
    return tr_ret


def mil_to_mm(mil_val: float) -> float:
    """
    """
    return mil_val * 0.0254


def mm_to_mil(mm_val: float) -> float:
    """
    """
    return mm_val / 0.0254
