from functools import reduce
from typing import List

import polars as pl


def listify(df_in: pl.DataFrame, cols: List[str]) -> list:
    """
    zip cols into a list of tuples
    super mega useful because it stops my fingers bleeding
    from all the typing
    """
    zippedy_lists = []
    for col in cols:
        zippedy_lists.append(df_in[col].to_list())
    return list(zip(*zippedy_lists))


def compose_n(*functions):
    """Compose multiple functions"""

    def compose(f, g):
        return lambda x: f(g(x))

    return reduce(compose, functions, lambda x: x)
