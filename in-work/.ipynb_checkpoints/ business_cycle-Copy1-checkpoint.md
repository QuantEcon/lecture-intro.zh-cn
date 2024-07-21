---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell}
!pip install wbgapi
```

# Business Cycle

## Overview

This lecture is about illustrateing business cycles in different countries and period.

Business cycle is one of the widely studied field since the birth of economics as a subject from .

In this lecture, we will see expensions and contractions of economies throughout the history with an emphasise on contemprary business cycles.

We use the following imports.

```{code-cell}
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as st
import wbgapi as wb
```

```{code-cell}
wb.series.info(q='GDP growth')
```

## GDP Growth Rate and Unemployment

First we look at the GDP growth rate and unemployment rate.

Let's source our data from the World Bank and clean the data

```{code-cell}
gdp_growth = wb.data.DataFrame('NY.GDP.MKTP.KD.ZG',['CHN', 'USA', 'BRA', 'GBR'], labels=True)
gdp_growth = gdp_growth.set_index('Country')
gdp_growth.columns = gdp_growth.columns.str.replace("YR", "").astype(int)
```

```{code-cell}
gdp_growth
```

```{code-cell}
fig, ax = plt.subplots()
plt.locator_params(axis='x', nbins=10)
ax.set_xticks([i for i in range(1960, 2021, 10)], minor=False)

def plot_gdp_growth(countries, title, ax, timeline, blocks, g_params, b_params):
    for country in countries:
        ax.plot(gdp_growth.loc[country], label=country)
        for b in blocks:
            ax.axvspan(*b, **b_params)
        for t in timeline:
            ax.text(*t, **t_params) 
    ax.set_title(title, pad=40)
    ax.set_ylim(ymin=-10, ymax=10)
    return ax

g_params = {"alpha": 0.6, "linestyle": ["-", "--"]}
b_params = {"color":'grey', "alpha": 0.2}
t_params = {"color":'grey', "fontsize": 9, "va":"center", "ha":"center"}
countries = ["United Kingdom", "United States"]
title = "United States and United Kingdom"
timeline = [[1991, 11, "1990s recession\n(1991)"],
             [1974, 11, "Oil Crisis\n(1974)"],
             [2008, 11, "GFC\n(2008)"],
             [2020, 11, "Covid-19\n(2020)"]]
blocks = [[1990, 1992],
             [1973, 1975],
             [2007, 2009],
             [2019, 2021]]

plot_gdp_growth(countries, title, ax, timeline, blocks, g_params, b_params)

```

```{code-cell}

```
