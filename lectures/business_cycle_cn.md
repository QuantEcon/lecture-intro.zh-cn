---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# 商业周期

## 概述

在本次讲座中，我们回顾了一些商业周期的经验方面。

商业周期是经济活动随时间波动的现象。

这些包括扩张（也称为繁荣）和收缩（也称为衰退）。

为了我们的研究，我们将使用来自[世界银行](https://documents.worldbank.org/en/publication/documents-reports/api) 和 [FRED](https://fred.stlouisfed.org/) 的经济指标。

除 Anaconda 已经安装的软件包外，本讲座还需要

```{code-cell} ipython3
:tags: [hide-output]

!pip install wbgapi
!pip install pandas-datareader
```

我们使用以下导入

```{code-cell} ipython3
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import wbgapi as wb
import pandas_datareader.data as web
```

这是一些帮助我们在绘图中使用颜色的小代码。

```{code-cell} ipython3
:tags: [hide-input]

# 设置图形参数
cycler = plt.cycler(linestyle=['-', '-.', '--', ':'], 
        color=['#377eb8', '#ff7f00', '#4daf4a', '#ff334f'])
plt.rc('axes', prop_cycle=cycler)
```


## 数据获取

我们将使用世界银行的数据 API `wbgapi` 和 `pandas_datareader` 来检索数据。

我们可以使用 `wb.series.info` 并带有参数 `q` 来查询来自 [世界银行](https://www.worldbank.org/en/home) 的可用数据。

例如，让我们检索 GDP 增长数据 ID 以查询 GDP 增长数据。

```{code-cell} ipython3
wb.series.info(q='GDP growth')
```


现在我们使用这个系列 ID 来获取数据。

```{code-cell} ipython3
gdp_growth = wb.data.DataFrame('NY.GDP.MKTP.KD.ZG',
            ['USA', 'ARG', 'GBR', 'GRC', 'JPN'], 
            labels=True)
gdp_growth
```


我们可以查看该系列的元数据以了解更多关于该系列的信息（点击展开）。

```{code-cell} ipython3
:tags: [hide-output]

wb.series.metadata.get('NY.GDP.MKTP.KD.ZG')
```



(gdp_growth)=
## GDP 增长率

首先我们来看一下 GDP 增长。 

让我们从世界银行获取数据并对其进行清理。

```{code-cell} ipython3
# 使用之前检索到的系列 ID
gdp_growth = wb.data.DataFrame('NY.GDP.MKTP.KD.ZG',
            ['USA', 'ARG', 'GBR', 'GRC', 'JPN'], 
            labels=True)
gdp_growth = gdp_growth.set_index('Country')
gdp_growth.columns = gdp_growth.columns.str.replace('YR', '').astype(int)
```

以下是数据的初步观察

```{code-cell} ipython3
gdp_growth
```

我们编写一个函数来生成考虑衰退的单个国家的图表。

```{code-cell} ipython3
:tags: [hide-input]

def plot_series(data, country, ylabel, 
                txt_pos, ax, g_params,
                b_params, t_params, ylim=15, baseline=0):
    """
    绘制带有衰退高亮的时间序列。

    参数
    ----------
    data : pd.DataFrame    要绘制的数据
    country : str
        要绘制的国家名称
    ylabel : str
        y轴标签
    txt_pos : float
        衰退标签的位置
    y_lim : float
        y轴的限制
    ax : matplotlib.axes._subplots.AxesSubplot
        要绘制的轴
    g_params : dict
        线条参数
    b_params : dict
        衰退高亮的参数
    t_params : dict
        衰退标签的参数
    baseline : float, optional
        图中的虚线基线，默认为 0
    
    返回
    -------
    ax : matplotlib.axes.Axes
        带有绘图的轴
    """

    ax.plot(data.loc[country], label=country, **g_params)
    
    # 高亮显示衰退
    ax.axvspan(1973, 1975, **b_params)
    ax.axvspan(1990, 1992, **b_params)
    ax.axvspan(2007, 2009, **b_params)
    ax.axvspan(2019, 2021, **b_params)
    if ylim != None:
        ax.set_ylim([-ylim, ylim])
    else:
        ylim = ax.get_ylim()[1]
    ax.text(1974, ylim + ylim*txt_pos,
            '石油危机\n(1974)', **t_params) 
    ax.text(1991, ylim + ylim*txt_pos,
            '1990年代衰退\n(1991)', **t_params) 
    ax.text(2008, ylim + ylim*txt_pos,
            '全球金融危机\n(2008)', **t_params) 
    ax.text(2020, ylim + ylim*txt_pos,
            '新冠肺炎\n(2020)', **t_params)

    # 添加参考的基线
    if baseline != None:
        ax.axhline(y=baseline, 
                   color='black', 
                   linestyle='--')
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax

# 定义图形参数 
g_params = {'alpha': 0.7}
b_params = {'color':'grey', 'alpha': 0.2}
t_params = {'color':'grey', 'fontsize': 9, 
            'va':'center', 'ha':'center'}
```


让我们从美国开始。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "美国（GDP增长率%）"
    name: us_gdp
---

fig, ax = plt.subplots()

country = 'United States'
ylabel = 'GDP增长率 (%)'
plot_series(gdp_growth, country, 
            ylabel, 0.1, ax, 
            g_params, b_params, t_params)
plt.show()
```

+++ {"user_expressions": []}

GDP 增长率平均为正，并且随着时间略有下降。

我们还可以看到 GDP 增长随着时间的波动，其中一些波动相当大。

让我们再看看几个国家以进行比较。

+++

英国（UK）的模式与美国相似，增长率缓慢下降，并且存在显著波动。

请注意，在新冠肺炎大流行期间发生的非常大的下降。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "英国（GDP增长率%）"
    name: uk_gdp
---

fig, ax = plt.subplots()

country = 'United Kingdom'
plot_series(gdp_growth, country, 
            ylabel, 0.1, ax, 
            g_params, b_params, t_params)
plt.show()
```

+++ {"user_expressions": []}

现在让我们考虑一下日本，日本在 1960 年代和 1970 年代经历了快速增长，随后在过去的二十年中扩张速度放缓。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "日本（GDP增长率%）"
    name: jp_gdp
---

fig, ax = plt.subplots()

country = 'Japan'
plot_series(gdp_growth, country, 
            ylabel, 0.1, ax, 
            g_params, b_params, t_params)
plt.show()
```

日本的增长率大幅下降与1970年代的石油危机、全球金融危机（GFC）和新冠疫情相吻合。

现在让我们研究希腊。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "希腊（GDP增长率%）"
    name: gc_gdp
---

fig, ax = plt.subplots()

country = 'Greece'
plot_series(gdp_growth, country, 
            ylabel, 0.1, ax, 
            g_params, b_params, t_params)
plt.show()
```

希腊在2010-2011年期间遭遇了大幅的GDP增长下滑，当时正是希腊债务危机的高峰期。

接下来我们来考虑一下阿根廷。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "阿根廷（GDP增长率%）"
    name: arg_gdp
---

fig, ax = plt.subplots()

country = 'Argentina'
plot_series(gdp_growth, country, 
            ylabel, 0.1, ax, 
            g_params, b_params, t_params)
plt.show()
```

注意，阿根廷经历了比上述经济体更为剧烈的周期波动。

同时，与发达经济体在1970年代和1990年代的衰退不同，阿根廷的增长率在这些时期没有下降。

## 失业率

另一个衡量商业周期的重要指标是失业率。

我们使用FRED的数据研究从1929年到1942年的失业率数据，从1948年到2022年的失业率数据，结合美国人口普查局[1942-1948年失业率估算数据](https://www.census.gov/library/publications/1975/compendia/hist_stats_colonial-1970.html)。

```{code-cell} ipython3
:tags: [hide-input]

start_date = datetime.datetime(1929, 1, 1)
end_date = datetime.datetime(1942, 6, 1)

unrate_history = web.DataReader('M0892AUSM156SNBR', 
                    'fred', start_date,end_date)
unrate_history.rename(columns={'M0892AUSM156SNBR': 'UNRATE'}, 
                inplace=True)

start_date = datetime.datetime(1948, 1, 1)
end_date = datetime.datetime(2022, 12, 31)

unrate = web.DataReader('UNRATE', 'fred', 
                    start_date, end_date)
```

让我们绘制从1929年到2022年的美国失业率，并定义NBER定义的衰退期。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "长期失业率，美国（%）"
    name: lrunrate
tags: [hide-input]
---

# 我们使用美国人口普查局对1942年至1948年期间的失业率估算
years = [datetime.datetime(year, 6, 1) for year in range(1942, 1948)]
unrate_census = [4.7, 1.9, 1.2, 1.9, 3.9, 3.9]

unrate_census = {'DATE': years, 'UNRATE': unrate_census}
unrate_census = pd.DataFrame(unrate_census)
unrate_census.set_index('DATE', inplace=True)

# 获取NBER定义的衰退期
start_date = datetime.datetime(1929, 1, 1)
end_date = datetime.datetime(2022, 12, 31)

nber = web.DataReader('USREC', 'fred', start_date, end_date)

fig, ax = plt.subplots()

ax.plot(unrate_history, **g_params, 
        color='#377eb8', 
        linestyle='-', linewidth=2)
ax.plot(unrate_census, **g_params, 
        color='black', linestyle='--', 
        label='Census estimates', linewidth=2)
ax.plot(unrate, **g_params, color='#377eb8', 
        linestyle='-', linewidth=2)

# 根据NBER衰退指标绘制灰色框
ax.fill_between(nber.index, 0, 1,
                where=nber['USREC']==1, 
                color='grey', edgecolor='none',
                alpha=0.3, 
                transform=ax.get_xaxis_transform(), 
                label='NBER衰退指标')
ax.set_ylim([0, ax.get_ylim()[1]])
ax.legend(loc='upper center', 
          bbox_to_anchor=(0.5, 1.1),
          ncol=3, fancybox=True, shadow=True)
ax.set_ylabel('失业率 (%)')

plt.show()
```

该图显示：

- 劳动力市场的扩张和收缩与衰退高度相关。
- 周期通常是不对称的：失业率急剧上升后是缓慢的恢复。

它还向我们展示了后疫情恢复期间美国劳动力市场条件的独特性。

劳动力市场在2020-2021年冲击后的恢复速度前所未有。

## 同步性

在我们之前的讨论中，我们发现发达经济体经历了相对同步的衰退时期。

同时，这种同步性在阿根廷直到2000年才出现。

让我们进一步研究这个趋势。

通过稍加修改，我们可以使用之前的函数绘制包含多个国家的图表。

```{code-cell} ipython3
---
tags: [hide-input]
---

def plot_comparison(data, countries, 
                    ylabel, txt_pos, y_lim, ax, 
                    g_params, b_params, t_params, 
                    baseline=0):
    """
    在相同图表上绘制多个系列。

    参数
    ----------
    data : pd.DataFrame
        要绘制的数据
    countries : list
        要绘制的国家列表
    ylabel : str
        y轴标签
    txt_pos : float
        衰退标签的位置
    y_lim : float
        y轴的限制
    ax: matplotlib.axes._subplots.AxesSubplot
        要绘制的轴
    g_params : dict
        线条参数
    b_params : dict
        衰退高亮的参数
    t_params : dict
        衰退标签的参数
    baseline : float, optional
        图中的虚线基线，默认为 0
    
    返回
    -------
    ax : matplotlib.axes.Axes
        带有绘图的轴
    """
    
    # 允许函数处理多个系列
    for country in countries:
        ax.plot(data.loc[country], label=country, **g_params)
    
    # 高亮显示衰退
    ax.axvspan(1973, 1975, **b_params)
    ax.axvspan(1990, 1992, **b_params)
    ax.axvspan(2007, 2009, **b_params)
    ax.axvspan(2019, 2021, **b_params)
    if y_lim != None:
        ax.set_ylim([-y_lim, y_lim])
    ylim = ax.get_ylim()[1]
    ax.text(1974, ylim + ylim*txt_pos, 
            '石油危机\n(1974)', **t_params) 
    ax.text(1991, ylim + ylim*txt_pos, 
            '1990年代衰退\n(1991)', **t_params) 
    ax.text(2008, ylim + ylim*txt_pos, 
            '全球金融危机\n(2008)', **t_params) 
    ax.text(2020, ylim + ylim*txt_pos, 
            '新冠肺炎\n(2020)', **t_params) 
    if baseline != None:
        ax.hlines(y=baseline, xmin=ax.get_xlim()[0], 
                xmax=ax.get_xlim()[1], color='black', 
                linestyle='--')
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax

# 定义图形参数 
g_params = {'alpha': 0.7}
b_params = {'color':'grey', 'alpha': 0.2}
t_params = {'color':'grey', 'fontsize': 9, 
            'va':'center', 'ha':'center'}
```

在这里我们比较发达经济体和发展中经济体的GDP增长率。

```{code-cell} ipython3
---
tags: [hide-input]
---

# 获取一个国家列表的GDP增长率
gdp_growth = wb.data.DataFrame('NY.GDP.MKTP.KD.ZG',
            ['CHN', 'USA', 'DEU', 'BRA', 'ARG', 'GBR', 'JPN', 'MEX'], 
            labels=True)
gdp_growth = gdp_growth.set_index('Country')
gdp_growth.columns = gdp_growth.columns.str.replace('YR', '').astype(int)

```

我们使用英国、美国、德国和日本作为发达经济体的例子。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "发达经济体（GDP增长率%）"
    name: adv_gdp
tags: [hide-input]
---

fig, ax = plt.subplots()
countries = ['United Kingdom', 'United States', 'Germany', 'Japan']
ylabel = 'GDP增长率 (%)'
plot_comparison(gdp_growth.loc[countries, 1962:], 
                countries, ylabel,
                0.1, 20, ax, 
                g_params, b_params, t_params)
plt.show()
```

我们选择巴西、中国、阿根廷和墨西哥作为代表性发展中经济体。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "发展中经济体（GDP增长率%）"
    name: deve_gdp
tags: [hide-input]
---

fig, ax = plt.subplots()
countries = ['Brazil', 'China', 'Argentina', 'Mexico']
plot_comparison(gdp_growth.loc[countries, 1962:], 
                countries, ylabel, 
                0.1, 20, ax, 
                g_params, b_params, t_params)
plt.show()
```


上述GDP增长率的比较表明
在21世纪的衰退期，商业周期变得更加同步。

然而，发展中和较不发达的经济体通常在整个经济周期中经历更剧烈的变化。

尽管GDP增长同步，但各国在衰退期间的经历往往不同。

我们使用失业率和劳动力市场条件的恢复作为另一个例子。

在这里我们比较美国、英国、日本和法国的失业率。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "发达经济体（失业率%）"
tags: [hide-input]
---

unempl_rate = wb.data.DataFrame('SL.UEM.TOTL.NE.ZS',
    ['USA', 'FRA', 'GBR', 'JPN'], labels=True)
unempl_rate = unempl_rate.set_index('Country')
unempl_rate.columns = unempl_rate.columns.str.replace('YR', '').astype(int)

fig, ax = plt.subplots()

countries = ['United Kingdom', 'United States', 'Japan', 'France']
ylabel = '失业率 (全国估计) (%)'
plot_comparison(unempl_rate, countries, 
                ylabel, 0.05, None, ax, g_params, 
                b_params, t_params, baseline=None)
plt.show()
```

我们看到，法国由于其强大的劳工工会，通常在负面冲击后经历相对较慢的劳动力市场恢复。

我们还注意到，日本有着非常低且稳定的失业率的历史。

## 先行指标和相关因素

研究先行指标和相关因素有助于政策制定者理解商业周期的成因和结果。

我们将从三个角度讨论潜在的先行指标和相关因素：消费、生产和信用水平。

### 消费

消费取决于消费者对其未来收入和整体经济表现的信心。

一个广泛引用的消费者信心指标是由密歇根大学发布的[消费者信心指数](https://fred.stlouisfed.org/series/UMCSENT)。

这里我们绘制了密歇根大学消费者信心指数和1978-2022年美国的年度[核心消费价格指数](https://fred.stlouisfed.org/series/CPILFESL)（CPI）变化。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "消费者信心指数和CPI同比变化，美国"
    name: csicpi
tags: [hide-input]
---

start_date = datetime.datetime(1978, 1, 1)
end_date = datetime.datetime(2022, 12, 31)

# 将绘图限制在特定范围
start_date_graph = datetime.datetime(1977, 1, 1)
end_date_graph = datetime.datetime(2023, 12, 31)

nber = web.DataReader('USREC', 'fred', start_date, end_date)
consumer_confidence = web.DataReader('UMCSENT', 'fred', 
                                start_date, end_date)

fig, ax = plt.subplots()
ax.plot(consumer_confidence, **g_params, 
        color='#377eb8', linestyle='-', 
        linewidth=2)
ax.fill_between(nber.index, 0, 1, 
            where=nber['USREC']==1, 
            color='grey', edgecolor='none',
            alpha=0.3, 
            transform=ax.get_xaxis_transform(), 
            label='NBER衰退指标')
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_ylabel('消费者信心指数')

# 在另一个y轴上绘制CPI
ax_t = ax.twinx()
inflation = web.DataReader('CPILFESL', 'fred', 
                start_date, end_date).pct_change(12)*100

# 在图例中添加CPI而不再绘制线条
ax_t.plot(2020, 0, **g_params, linestyle='-', 
          linewidth=2, label='消费者信心指数')
ax_t.plot(inflation, **g_params, 
          color='#ff7f00', linestyle='--', 
          linewidth=2, label='CPI同比变化 (%)')

ax_t.fill_between(nber.index, 0, 1,
                  where=nber['USREC']==1, 
                  color='grey', edgecolor='none',
                  alpha=0.3, 
                  transform=ax.get_xaxis_transform(), 
                  label='NBER衰退指标')
ax_t.set_ylim([0, ax_t.get_ylim()[1]])
ax_t.set_xlim([start_date_graph, end_date_graph])
ax_t.legend(loc='upper center',
            bbox_to_anchor=(0.5, 1.1),
            ncol=3, fontsize=9)
ax_t.set_ylabel('CPI同比变化 (%)')
plt.show()
```

我们发现

* 消费者信心通常在扩展期间保持高位，并在衰退前下降。
* 消费者信心与CPI之间存在明显的负相关关系。

当消费者商品价格上涨时，消费者信心减弱。

在[滞胀](https://zh.wikipedia.org/wiki/%E6%BB%9E%E8%83%80)期间，这一趋势尤为显著。

### 生产

实际工业产出与经济中的衰退高度相关。

然而，它不是领先指标，因为生产萎缩的高峰相对于消费者信心和通货膨胀是滞后的。

我们绘制了1919年至2022年间美国实际工业产出变化的图表，以显示这一趋势。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "同比实际产出变化，美国（%）"
    name: roc
tags: [hide-input]
---

start_date = datetime.datetime(1919, 1, 1)
end_date = datetime.datetime(2022, 12, 31)

nber = web.DataReader('USREC', 'fred', 
                    start_date, end_date)
industrial_output = web.DataReader('INDPRO', 'fred', 
                    start_date, end_date).pct_change(12)*100

fig, ax = plt.subplots()
ax.plot(industrial_output, **g_params, 
        color='#377eb8', linestyle='-', 
        linewidth=2, label='工业生产指数')
ax.fill_between(nber.index, 0, 1,
                where=nber['USREC']==1, 
                color='grey', edgecolor='none',
                alpha=0.3, 
                transform=ax.get_xaxis_transform(), 
                label='NBER衰退指标')
ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1]])
ax.set_ylabel('同比实际产出变化 (%)')
plt.show()
```

我们在图中观察到跨越衰退的滞后收缩。

### 信贷水平

信贷在衰退期间通常会收缩，因为贷款人变得更加谨慎，借款人对承担更多债务持犹豫态度。

这是由于整体经济活动减少和对未来预期黯淡等因素所致。

一个例子是英国银行对私人部门的国内信贷。

下图显示了1970年至2022年间英国银行向私人部门提供的国内信贷占GDP的百分比。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "银行向私人部门提供的国内信贷（占GDP%）"
    name: dcpc
tags: [hide-input]
---

private_credit = wb.data.DataFrame('FS.AST.PRVT.GD.ZS', 
                ['GBR'], labels=True)
private_credit = private_credit.set_index('Country')
private_credit.columns = private_credit.columns.str.replace('YR', '').astype(int)

fig, ax = plt.subplots()

countries = 'United Kingdom'
ylabel = '信贷水平 (占GDP%)'
ax = plot_series(private_credit, countries, 
                 ylabel, 0.05, ax, g_params, b_params, 
                 t_params, ylim=None, baseline=None)
plt.show()
```

注意，信贷水平在经济扩张期间上升，并在衰退后停滞甚至收缩。