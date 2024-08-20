---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# 长期增长

## 概述

在本讲座中，我们将使用Python、{doc}`pandas<pyprog:pandas>` 和 {doc}`Matplotlib<pyprog:matplotlib>` 来下载、组织和可视化经济增长的历史数据。

除了学习如何更广泛地部署这些工具之外，我们还将使用它们来描述跨越多个世纪的多个国家的经济增长经历。

这样的“增长事实”出于多种原因是有趣的。

解释增长事实是“发展经济学”和“经济史学”的主要目的。

增长事实是历史学家研究地缘政治力量和动态的重要输入。

因此，亚当·图兹（Adam Tooze）对第一次世界大战的地缘政治前因和后果的描述始于描述欧洲大国在1914年前70年间的国内生产总值（GDP）如何演变（见{cite}`Tooze_2014`的第1章）。

使用图兹构建他的图表所使用的相同数据（时间线稍长一点），以下是我们版本的第1章图表。

```{figure} _static/lecture_specific/long_run_growth/tooze_ch1_graph.png
:width: 100%
```

（这只是我们的图{numref}`gdp1`的副本。我们稍后在本讲座中描述了它是如何构建的。）

{cite}`Tooze_2014`的第1章使用他的图表展示了美国GDP如何在19世纪初大大落后于大英帝国的GDP。

到19世纪末，美国GDP已经赶上了大英帝国的GDP，而在20世纪上半叶，美国GDP超过了大英帝国的GDP。

对于亚当·图兹来说，这一事实是“美国世纪”的关键地缘政治基础。

查看此图及其如何为“美国（20世纪）世纪”奠定地缘政治基础自然会使人想要了解2014年或之后的相应图表。

（焦急的读者现在可能想要跳到前面并查看图{numref}`gdp2`以获得答案提示。）

正如我们将看到的，通过类比推理，这张图表或许奠定了“XXX（21世纪）世纪”的基础，你可以自由猜测国家XXX。

在我们收集数据以构建这两个图表的过程中，我们还将研究多国在尽可能长的时间范围内的增长经历。

这些图表将描绘“工业革命”如何在18世纪末始于英国，然后迁移到一个国家又一个国家。

简而言之，本讲座记录了若干国家在长时间里的增长轨迹。

虽然有些国家经历了持续一百年的长期快速增长，其他国家却没有。

由于国家间的人口有所不同，并且在一个国家内随时间变化，

描述总GDP和人均GDP如何随时间变化也是很有趣的。

首先，让我们导入探索长期增长数据所需的包

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from collections import namedtuple
```

## 设置

由[Angus Maddison](https://en.wikipedia.org/wiki/Angus_Maddison)发起的一个项目收集了许多与经济增长相关的历史时间序列，有些数据可以追溯到第一世纪。

可以从[Maddison Historical Statistics](https://www.rug.nl/ggdc/historicaldevelopment/maddison/)下载数据，点击“Latest Maddison Project Release”即可获取。

我们将从一个QuantEcon GitHub仓库读取数据。

我们在这一部分的目标是生成一个包含不同国家人均GDP的便捷`DataFrame`实例。

这里我们将Maddison的数据读入pandas的`DataFrame`中：

```{code-cell} ipython3
data_url = "https://github.com/QuantEcon/lecture-python-intro/raw/main/lectures/datasets/mpd2020.xlsx"
data = pd.read_excel(data_url, 
                     sheet_name='Full data')
data.head()
```


我们可以看到这个数据集包含多个国家和年份的人均GDP (`gdppc`) 和人口 (pop)。

让我们看看这个数据集中有多少个国家以及是哪些国家。

```{code-cell} ipython3
countries = data.country.unique()
len(countries)
```

通过运行上面的代码，我们可以找到有多少个国家的数据。

下面，我们将列出这些国家。

```{code-cell} ipython3
countries
```

我们现在可以探索一些可用的169个国家。

让我们遍历每个国家，以了解每个国家可用的年份。

```{code-cell} ipython3
country_years = []
for country in countries:
    cy_data = data[data.country == country]['year']
    ymin, ymax = cy_data.min(), cy_data.max()
    country_years.append((country, ymin, ymax))
country_years = pd.DataFrame(country_years,
                    columns=['country', 'min_year', 'max_year']).set_index('country')
country_years.head()
```

现在我们已经设置了基本的数据框架，接下来让我们提取我们感兴趣的国家。

在此过程中，我们将构建一个新的数据框架，只包含各国人均收入的时间序列。

首先，让我们指定我们感兴趣的顶级国家。

```{code-cell} ipython3
top_countries = [
    'United States',
    'United Kingdom',
    'China',
    'Japan',
    'Germany',
    'India',
    'Australia',
    'Brazil',
    'Canada',
    'France',
    'Italy',
    'Russian Federation',
    'Indonesia'
]
```

让我们现在重塑原始数据为一些便于快速访问各国时间序列数据的变量。

我们可以在该数据集中构建国家代码（`countrycode`）与国家名称（`country`）之间的有用映射。

```{code-cell} ipython3
code_to_name = data[
    ['countrycode', 'country']].drop_duplicates().reset_index(drop=True).set_index(['countrycode'])
```

Now we are ready to extract and plot GDP data for each of these countries.

```{code-cell} ipython3
gdp_pc = data.pivot(index='year', 
                   columns='country',
                   values='gdppc')
pop = data.pivot(index='year', 
                columns='country',
                values='pop')
gdp_pc = gdp_pc[top_countries]
gdp_pc.head()
```

```{code-cell} ipython3
gdp_pc.plot(figsize=(10, 6), fontsize=12)
plt.title('Top Countries GDP per capita')
plt.legend(loc='upper left', fontsize=10, ncol=2)
plt.xlabel('Year', fontsize=12)
plt.ylabel('GDP per capita', fontsize=12)
plt.show()
```

让我们专注于人均GDP（`gdppc`）并生成宽格式的数据

```{code-cell} ipython3
gdp_pc = data.set_index(['countrycode', 'year'])['gdppc']
gdp_pc = gdp_pc.unstack('countrycode')
```

让我们检查这张表的尾部，以了解最新的数据。

```{code-cell} ipython3
gdp_pc.tail()
```

可以看到，并不是所有国家的年份都是相同的。

接下来，我们通过绘制一些国家对应的数据，开始进行一些初步的研究。

为了生成这些图表，我们将制作一张国家代码与国家名称的映射表，并使用它来标记图表。

```{code-cell} ipython3
# 我们将绘制四个国家的GDP时间序列
countries = ['AUS', 'BRA', 'CHN', 'USA']

plt.figure(figsize=(10, 6))

for code in countries:
    plt.plot(gdp_pc.index, gdp_pc[code], label=code_to_name.loc[code].country)

plt.title('Selected Countries GDP per capita')
plt.legend(loc='upper left')
plt.xlabel('Year')
plt.ylabel('GDP per capita')
plt.show()
```

上述绘图显示了几个国家的GDP，每个国家在每一年的变化情况。

接下来，我们将研究个别国家的图表，并与其他国家进行相对比较。

## 人均GDP

在这一部分，我们将研究几个不同国家的人均GDP的长期变化。

### 英国

首先，我们研究英国的GDP增长。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 人均GDP（GBR）
    name: gdppc_gbr1
    width: 500px
---
fig, ax = plt.subplots(dpi=300)
country = 'GBR'
gdp_pc[country].plot(
        ax=ax,
        ylabel='国际美元',
        xlabel='年份',
        color=color_mapping[country]
    );
```


:::{note}
[国际美元](https://en.wikipedia.org/wiki/international_dollar) 是一种假想的货币单位，其购买力平价与特定时期内的美元在美国的购买力平价相同。它们也被称为 Geary–Khamis 美元（GK 美元）。
:::

我们可以看到，在这一千年的前250年里，数据在较长时间内是不连续的，因此我们可以选择插值以获得连续的线图。

在这里我们使用虚线来表示插值趋势

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: GDP per Capita (GBR)
    name: gdppc_gbr2
---
fig, ax = plt.subplots(dpi=300)
country = 'GBR'
ax.plot(gdp_pc[country].interpolate(),
        linestyle='--',
        lw=2,
        color=color_mapping[country])

ax.plot(gdp_pc[country],
        lw=2,
        color=color_mapping[country])
ax.set_ylabel('international dollars')
ax.set_xlabel('year')
plt.show()
```



# 比较美国、英国和中国

在本节中，我们将比较美国、英国和中国的GDP增长。

第一步，我们创建一个函数，为一组国家生成绘图

```{code-cell} ipython3
def draw_interp_plots(series,        # pandas series
                      country,       # list of country codes
                      ylabel,        # label for y-axis
                      xlabel,        # label for x-axis
                      color_mapping, # code-color mapping
                      code_to_name,  # code-name mapping
                      lw,            # line width
                      logscale,      # log scale for y-axis
                      ax             # matplolib axis
                     ):

    for c in country:
        # 获取插值数据
        df_interpolated = series[c].interpolate(limit_area='inside')
        interpolated_data = df_interpolated[series[c].isnull()]

        # 用虚线绘制插值数据
        ax.plot(interpolated_data,
                linestyle='--',
                lw=lw,
                alpha=0.7,
                color=color_mapping[c])

        # 用实线绘制非插值数据
        ax.plot(series[c],
                lw=lw,
                color=color_mapping[c],
                alpha=0.8,
                label=code_to_name.loc[c]['country'])

        if logscale:
            ax.set_yscale('log')

    # 将图例绘制在图外
    ax.legend(loc='upper left', frameon=False)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
```

首先，让我们为美国和英国绘制GDP增长图

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 人均GDP（美国/英国）
    name: gdppc_gb_usa
tags: [hide-input]
---
fig, ax = plt.subplots(dpi=300)

country = ['USA', 'GBR']
draw_interp_plots(gdp_pc[country].loc[1500:], 
                  country,
                  '国际美元','年份',
                  color_mapping, code_to_name, 2, True, ax)
plt.show()
```

从图中可以看到英国和美国从1750年开始的增长变化情况，在大萧条（1929-1939）期间出现显著下降。

### 美国，中国和英国

最后，让我们加入中国并观察这三个国家的比较

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 人均GDP，中国、英国、美国（1500年以来）
    name: gdppc_comparison
tags: [hide-input]
---
fig, ax = plt.subplots(dpi=300, figsize=(10, 6))

country = ['CHN', 'GBR', 'USA']
draw_interp_plots(gdp_pc[country].loc[1500:], 
                  country,
                  '国际美元','年份',
                  color_mapping, code_to_name, 2, False, ax)
plt.show()
```

上面这张人均GDP的前面图表展示了工业革命的传播如何随着时间的推移逐渐提高了大量人群的生活水平

- 大部分增长发生在工业革命之后的过去150年间。
- 从1820年到1940年，美国和英国的人均GDP上升并与中国的有所差距。
- 1950年后，尤其是1970年代末后，差距迅速缩小。
- 这些结果反映了技术和经济政策因素的复杂组合，经济增长的研究者们试图理解并量化这些因素。

### 聚焦于中国

看到中国从1500年到1970年的人均GDP水平非常有趣。

请注意，从1700年到20世纪初的长期下降的GDP水平。

因此，图表显示了

- 清政府闭关锁国政策后的长期经济下行和停滞。
- 中国与英国工业革命开始后的不同经历。
- 自强运动似乎主要帮助了中国的增长。
- 现代中国经济政策的惊人增长成果，最终在1970年代末进行的改革和开放政策中达到高潮。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 人均GDP，1500-2000（中国）
    name: gdppc_china
tags: [hide-input]
---
fig, ax = plt.subplots(dpi=300, figsize=(10, 6))

country = ['CHN']
draw_interp_plots(gdp_pc[country].loc[1600:2000], 
                  country,
                  '国际美元','年份',
                  color_mapping, code_to_name, 2, True, ax)

ylim = ax.get_ylim()[1]

events = [
Event((1655, 1684), ylim + ylim*0.06, 
      '闭关锁国政策\n(1655-1684)', 
      'tab:orange', 1),
Event((1760, 1840), ylim + ylim*0.06, 
      '工业革命\n(1760-1840)', 
      'grey', 1),
Event((1839, 1842), ylim + ylim*0.2, 
      '第一次鸦片战争\n(1839–1842)', 
      'tab:red', 1.07),
Event((1861, 1895), ylim + ylim*0.4, 
      '自强运动\n(1861–1895)', 
      'tab:blue', 1.14),
Event((1939, 1945), ylim + ylim*0.06, 
      '第二次世界大战\n(1939-1945)', 
      'tab:red', 1),
Event((1948, 1950), ylim + ylim*0.23, 
      '中华人民共和国成立\n(1949)', 
      color_mapping['CHN'], 1.08),
Event((1958, 1962), ylim + ylim*0.5, 
      '大跃进\n(1958-1962)', 
      'tab:orange', 1.18),
Event((1978, 1979), ylim + ylim*0.7, 
      '改革开放\n(1978-1979)', 
      'tab:blue', 1.24)
]

# 绘制事件
draw_events(events, ax)
plt.show()
```

### 聚焦于美国和英国

现在我们更详细地研究美国（USA）和英国（GBR）。

在以下图表中，请注意
- 贸易政策（航海法）的影响。
- 工业革命带来的生产力变化。
- 美国逐渐接近并超过英国，为“美国世纪”奠定基础。
- 战争的意外后果。
- [经济周期](business_cycle)衰退和萧条带来的中断和伤痕。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 人均GDP，1500-2000（英国和美国）
    name: gdppc_ukus
tags: [hide-input]
---
fig, ax = plt.subplots(dpi=300, figsize=(10, 6))

country = ['GBR', 'USA']
draw_interp_plots(gdp_pc[country].loc[1500:2000],
                  country,
                  '国际美元','年份',
                  color_mapping, code_to_name, 2, True, ax)

ylim = ax.get_ylim()[1]

# 创建数据点列表
events = [
    Event((1651, 1651), ylim + ylim*0.15, 
          '航海法（英国）\n(1651)', 
          'tab:orange', 1),
    Event((1765, 1791), ylim + ylim*0.15, 
          '美国独立战争\n(1765-1791)',
          color_mapping['USA'], 1),
    Event((1760, 1840), ylim + ylim*0.6, 
          '工业革命\n(1760-1840)', 
          'grey', 1.08),
    Event((1848, 1850), ylim + ylim*1.1, 
          '废除航海法（英国）\n(1849)', 
          'tab:blue', 1.14),
    Event((1861, 1865), ylim + ylim*1.8, 
          '美国内战\n(1861-1865)', 
          color_mapping['USA'], 1.21),
    Event((1914, 1918), ylim + ylim*0.15, 
          '第一次世界大战\n(1914-1918)', 
          'tab:red', 1),
    Event((1929, 1939), ylim + ylim*0.6, 
          '大萧条\n(1929–1939)', 
          'grey', 1.08),
    Event((1939, 1945), ylim + ylim*1.1, 
          '第二次世界大战\n(1939-1945)', 
          'tab:red', 1.14)
]

# 绘制事件
draw_events(events, ax)
plt.show()
```

# GDP增长

现在我们将构建一些对地缘政治历史学家如亚当·图兹感兴趣的图表。

我们将关注总国内生产总值（GDP）（作为“国家地缘政治-军事力量”的代理变量），而不是人均GDP（作为生活水平的代理变量）。

```{code-cell} ipython3
data = pd.read_excel(data_url, sheet_name='Full data')
data.set_index(['countrycode', 'year'], inplace=True)
data['gdp'] = data['gdppc'] * data['pop']
gdp = data['gdp'].unstack('countrycode')
```

现在我们有了每年每个国家的GDP，我们可以绘制一些对地缘政治学家和历史学家有趣的比较图表。

## 早期工业化（1820至1940年）

我们首先可视化中国、前苏联、日本、英国和美国的趋势。

最显著的趋势是美国的崛起，在1860年代超过英国，并在1880年代超过中国。

这种增长持续到1930年代的大萧条到来时出现大幅下降。

与此同时，俄罗斯在第一次世界大战期间经历了重大挫折，并在二月革命后显著恢复。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 早期工业化时代的GDP
    name: gdp1
---
fig, ax = plt.subplots(dpi=300)
country = ['CHN', 'SUN', 'JPN', 'GBR', 'USA']
start_year, end_year = (1820, 1945)
draw_interp_plots(gdp[country].loc[start_year:end_year], 
                  country,
                  '国际美元', '年份',
                  color_mapping, code_to_name, 2, False, ax)
```

#### 构建类似于图兹的图

在本节中，我们描述如何构建从{cite}`Tooze_2014`的第1章中讨论的引人注目的图表的版本。

首先定义一个国家集合，这些国家包括大英帝国（BEM），这样我们就可以在图兹的图表中复制该系列。

```{code-cell} ipython3
BEM = ['GBR', 'IND', 'AUS', 'NZL', 'CAN', 'ZAF']
# 插值不完全时间序列
gdp['BEM'] = gdp[BEM].loc[start_year-1:end_year].interpolate(method='index').sum(axis=1)
```

### 扩展后的结果

现在让我们绘制所有这些国家在更长时间范围内的总GDP。

下图（{numref}`gdp2`) 展示了在1500-2020年期间的这些系列。

指示美国和英国崛起的相同模式也适用于长时间范围。

尽管如此，自1970年代末以来的中国增长更显著。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 各国总GDP（1500-2020）
    name: gdp2
---
fig, ax = plt.subplots(dpi=300, figsize=(10, 6))
country = ['USA', 'GBR', 'CHN', 'FRA', 'DEU', 'JPN', 'RUS', 'BEM']
draw_interp_plots(gdp[country].iloc[87:], 
                  country,
                  '国际美元', '年份',
                  color_mapping, code_to_name, 2, True, ax)
```

## 参考文献

```{bibliography} ../bibliographies/main.bib
:filter: docname in docnames
```

在这节课开始时，我们指出了美国GDP如何在19世纪初从“无”到与大英帝国的GDP相抗衡，然后在19世纪末超过大英帝国的GDP，为“美国（20世纪）世纪”奠定了地缘政治基础。

让我们向前推进时间，开始将图兹的图表与二战后的情况作比较。

按照图兹第1章的分析精神，这将提供一些关于当前地缘政治现状的信息。

### 现代时代（1950至2020年）

下图显示了中国增长的迅速程度，特别是自1970年代后期以来。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 现代时代的GDP
    name: gdp2
---
fig, ax = plt.subplots(dpi=300)
country = ['CHN', 'SUN', 'JPN', 'GBR', 'USA']
start_year, end_year = (1950, 2020)
draw_interp_plots(gdp[country].loc[start_year:end_year], 
                  country,
                  '国际美元', '年份',
                  color_mapping, code_to_name, 2, False, ax)
```

很自然的我们会将这个图表与图{numref}`gdp1`相比较，后者显示了美国在“美国世纪”开始时超过英国的情况，该图表是{cite}`Tooze_2014`第1章的一个版本。

## 区域分析

我们经常要研究“世界大国”俱乐部之外的国家的历史经历。

[Maddison Historical Statistics](https://www.rug.nl/ggdc/historicaldevelopment/maddison/) 数据集还包括区域聚合

```{code-cell} ipython3
data = pd.read_excel(data_url, 
                     sheet_name='Regional data', 
                     header=(0,1,2),
                     index_col=0)
data.columns = data.columns.droplevel(level=2)
```

我们可以将原始数据存储在更方便的格式中，以构建区域人均GDP的单一表格

```{code-cell} ipython3
regionalgdp_pc = data['gdppc_2011'].copy()
regionalgdp_pc.index = pd.to_datetime(regionalgdp_pc.index, format='%Y')
```

让我们基于时间进行插值，以填补数据集中的任何缺口，以便于绘图

```{code-cell} ipython3
regionalgdp_pc.interpolate(method='time', inplace=True)
```

进行更深入的研究，让我们将`Western Offshoots`和`Sub-Saharan Africa`的时间序列与世界各地多个不同区域进行比较。

再次展示了工业革命之后西方与世界其他地区的差距，以及1950年之后的世界趋同

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 区域人均GDP
    name: region_gdppc
---
fig, ax = plt.subplots(dpi=300)
regionalgdp_pc.plot(ax=ax, xlabel='年份',
                    lw=2,
                    ylabel='国际美元')
ax.set_yscale('log')
plt.legend(loc='lower center',
           ncol=3, bbox_to_anchor=[0.5, -0.5])
plt.show()
```