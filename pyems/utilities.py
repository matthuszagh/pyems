import sys
from typing import List
import numpy as np


def pretty_print(
    data: List[List[float]], col_names=List[str], out_file=sys.stdout
) -> None:
    """
    """
    col_width = (
        int(np.log10(np.amax([np.amax(sublist) for sublist in data]))) + 5
    )
    for col in col_names:
        out_file.write("{:{width}}".format(col, width=col_width))
    out_file.write("\n")

    for row in range(len(data[0])):
        for col in range(len(data)):
            out_file.write(
                "{:<{width}.2f}".format(data[col][row], width=col_width)
            )
        out_file.write("\n")


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
    if permit_outside:
        if sel_val < arr[0][sel_col]:
            return arr[0][target_col]
        if sel_val > arr[-1][sel_col]:
            return arr[-1][target_col]

    if sel_val == arr[0][sel_col]:
        return arr[0][target_col]
    if sel_val == arr[-1][sel_col]:
        return arr[-1][sel_val]

    ins_idx = table_insertion_idx(sel_val, arr, sel_col)
    xlow = arr[ins_idx - 1][sel_col]
    xhigh = arr[ins_idx][sel_col]
    ylow = arr[ins_idx - 1][target_col]
    yhigh = arr[ins_idx][target_col]

    return interp_lin(sel_val, xlow, xhigh, ylow, yhigh)
