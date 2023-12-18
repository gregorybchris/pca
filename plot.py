from typing import List

import altair as alt
import pandas as pd


def plot(data: pd.DataFrame, x: str, y: str, colors: List[str], tooltip: List[str]):
    base = (
        alt.Chart(data)
        .mark_circle(size=50)
        .encode(x=x, y=y, tooltip=tooltip)
        .properties(width=200, height=200)
        .interactive()
    )

    chart = alt.hconcat()
    for color in colors:
        chart |= base.encode(color=color)

    alt.renderers.enable("altair_viewer")
    chart.show()
