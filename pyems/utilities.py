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
