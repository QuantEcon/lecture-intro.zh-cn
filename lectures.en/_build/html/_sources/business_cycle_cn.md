

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

在本讲座中，我们回顾了一些商业周期的实证方面。

商业周期是经济活动随时间波动的一种现象。

这些波动包括扩张（也称为繁荣）和收缩（也称为衰退）。

为了我们的研究，我们将使用来自[世界银行](https://documents.worldbank.org/en/publication/documents-reports/api)和[FRED](https://fred.stlouisfed.org/)的经济指标。

除了已经由Anaconda安装的软件包外，本讲座还需要

```{code-cell} ipython3
:tags: [hide-output]

!pip install wbgapi
!pip install pandas-datareader
```
图像输入功能：已启用

我们使用以下导入

```{code-cell} ipython3
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import wbgapi as wb
import pandas_datareader.data as web
```

以下代码可帮助我们在作图时设置颜色。

```{code-cell} ipython3
:tags: [hide-input]

# 设置图形参数
cycler = plt.cycler(linestyle=['-', '-.', '--', ':'], 
        color=['#377eb8', '#ff7f00', '#4daf4a', '#ff334f'])
plt.rc('axes', prop_cycle=cycler)
```

## 数据获取

我们将使用世界银行的数据API `wbgapi` 和 `pandas_datareader` 来获取数据。

我们可以使用 `wb.series.info` 和参数 `q` 来查询可用数据
来自[世界银行](https://www.worldbank.org/en/home)。

例如，让我们检索GDP增长数据ID来查询GDP增长数据。

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

我们可以查看系列的元数据以了解更多关于该系列的信息（点击展开）。

```{code-cell} ipython3
:tags: [hide-output]

wb.series.metadata.get('NY.GDP.MKTP.KD.ZG')
```

## GDP增长率

首先我们来看GDP增长率。

我们从世界银行获取数据并进行清理。

```{code-cell} ipython3
# 使用之前获取的系列 ID
gdp_growth = wb.data.DataFrame('NY.GDP.MKTP.KD.ZG',
            ['USA', 'ARG', 'GBR', 'GRC', 'JPN'], 
            labels=True)
gdp_growth = gdp_growth.set_index('Country')
gdp_growth.columns = gdp_growth.columns.str.replace('YR', '').astype(int)
```

让我们绘制年度GDP增长率。

```{code-cell} ipython3
ax = gdp_growth.T.plot(title='年度GDP增长率')
ax.set_xlabel('年度')
ax.set_ylabel('%')
plt.show()
```

## 美国失业率

现在我们获取美国的失业率数据。

我从FRED数据库中进行检索。

```{code-cell} ipython3
key = 'UNRATE'
start=datetime.datetime(1980, 1, 1)
end=datetime.datetime(2022, 1, 1)
unemployment = web.DataReader(key, 'fred', start, end)
unemployment.plot(title='美国失业率')
```

## 日本

我们也可以获取和处理日本的数据。让我们来看一下日本的失业率。

```{code-cell} ipython3
key = 'LRUNTTTTJPM156S'
japan_unemployment = web.DataReader(key, 'fred', start, end)
japan_unemployment.plot(title='日本失业率')
```

## 汇总 

现在我们有GDP增长率数据和失业率数据，可以进行一些总结。

```{code-cell} ipython3
# 绘制日本和美国的失业率进行比较
fig, ax = plt.subplots()
japan_unemployment.plot(ax=ax, label='日本失业率')
unemployment.plot(ax=ax, label='美国失业率')

ax.set_title('日本 vs 美国失业率')
ax.legend()
plt.show()
```

现在让我们考虑经历了1960年代和1970年代的快速增长，随后在过去二十年中扩张速度放缓的日本。

主要的增长率下滑与1970年代的石油危机、全球金融危机（GFC）和Covid-19大流行同时发生。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "日本（GDP增长率百分比）"
    name: jp_gdp
---

fig, ax = plt.subplots()

country = 'Japan'
plot_series(gdp_growth, country, 
            ylabel, 0.1, ax, 
            g_params, b_params, t_params)
plt.show()
```

现在让我们学习一下希腊。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "希腊（GDP增长率百分比）"
    name: gc_gdp
---

fig, ax = plt.subplots()

country = 'Greece'
plot_series(gdp_growth, country, 
            ylabel, 0.1, ax, 
            g_params, b_params, t_params)
plt.show()
```

希腊在2010-2011年，希腊债务危机的高峰期，其GDP增长率经历了非常大的下降。

接下来让我们考虑阿根廷。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "阿根廷（GDP增长率百分比）"
    name: arg_gdp
---

fig, ax = plt.subplots()

country = 'Argentina'
plot_series(gdp_growth, country, 
            ylabel, 0.1, ax, 
            g_params, b_params, t_params)
plt.show()
```

注意到阿根廷经历的经济周期比上面讨论的经济更为波动。

与此同时，阿根廷的增长率在1970年代和1990年代的两次发达经济体衰退期间并未下降。


## 失业率

另一个衡量经济周期的重要指标是失业率。

我们使用FRED的失业率数据来研究从[1929-1942](https://fred.stlouisfed.org/series/M0892AUSM156SNBR)到[1948-2022](https://fred.stlouisfed.org/series/UNRATE)的失业率数据，结合由[人口普查局](https://www.census.gov/library/publications/1975/compendia/hist_stats_colonial-1970.html)估计的1942-1948年的失业率数据。

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

unrate = web.DataReader('UNRATE', 
                        'fred', start_date, end_date)
```

让我们绘制1929年至2022年美国失业率的图表，并定义由NBER定义的衰退期。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "长期失业率，美国 (%)"
    name: lrunrate
tags: [hide-input]
---

# 我们使用人口普查局对1942年至1948年失业率的估计
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
        label='人口普查估计', linewidth=2)
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

图表显示了以下内容

* 劳动力市场的扩张和收缩与经济衰退高度相关。
* 周期总体上是不对称的：失业率急剧上升后，复苏较慢。

它还显示了大流行后复苏期间美国劳动力市场条件的独特性。

2020-2021年冲击后，劳动力市场以前所未有的速度复苏。

(synchronization)=
## 同步

在我们{ref}`之前的讨论<gdp_growth>`中，我们发现发达经济体在衰退期内的同步程度相对较高。

与此同时，这种同步性直到2000年代才出现在阿根廷。

让我们进一步研究这一趋势。

通过稍作修改，我们可以使用之前的函数来绘制包含多个国家的图表。

```{code-cell} ipython3
---
tags: [hide-input]
---


def plot_comparison(data, countries, 
                        ylabel, txt_pos, y_lim, ax, 
                        g_params, b_params, t_params, 
                        baseline=0):
    """
    在同一图表上绘制多个系列

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
    ax : matplotlib.axes._subplots.AxesSubplot
        要绘制的轴
    g_params : dict
        线条的参数
    b_params : dict
        衰退高亮的参数
    t_params : dict
        衰退标签的参数
    baseline : float, optional
        图表上的虚线基线，默认值为 0
    
    返回结果
    -------
    ax : matplotlib.axes.Axes
        带有图表的轴。
    """
    
    # 允许函数遍历多个系列
    for country in countries:
        ax.plot(data.loc[country], label=country, **g_params)
    
    # 高亮显示衰退期
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
            '新冠疫情\n(2020)', **t_params) 
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

### 发达经济体 vs 发展中经济体

我们来比较发达经济体和发展中经济体的GDP增长率。

```{code-cell} ipython3
---
tags: [hide-input]
---

fig, ax = plt.subplots()

gdp_growth = wb.data.DataFrame('NY.GDP.MKTP.KD.ZG',
            ['CHN', 'USA', 'DEU', 'BRA', 'ARG', 'GBR', 'JPN', 'MEX'], 
            labels=True)
gdp_growth = gdp_growth.set_index('Country')
gdp_growth.columns = gdp_growth.columns.str.replace('YR', '').astype(int)

countries = ['CHN', 'BRA',  'ARG']
ylabel = 'GDP增长率 (%)'
txt_pos = 0.02
y_lim = 15
ax = plot_comparison(gdp_growth, countries, ylabel, 
                     txt_pos, y_lim, ax, 
                     g_params, b_params, t_params, 
                     baseline=0)

plt.show()
```

我们使用英国、美国、德国和日本作为发达经济体的例子。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "发达经济体（GDP增长率 %）"
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

我们选择巴西、中国、阿根廷和墨西哥作为代表性的发展中经济体。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "发展中经济体（GDP增长率 %）"
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

上述GDP增长率的比较表明，在21世纪的衰退期内，商业周期变得更加同步。

然而，新兴和欠发达经济体往往在整个经济周期中经历更为剧烈的波动。

尽管GDP增长率同步，但各国在衰退期间的经历往往不同。

我们使用失业率和劳动力市场状况的恢复作为另一个例子。

在这里，我们比较美国、英国、日本和法国的失业率。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "发达经济体（失业率 %）"
    name: adv_unemp
tags: [hide-input]
---

unempl_rate = wb.data.DataFrame('SL.UEM.TOTL.NE.ZS',
    ['USA', 'FRA', 'GBR', 'JPN'], labels=True)
unempl_rate = unempl_rate.set_index('Country')
unempl_rate.columns = unempl_rate.columns.str.replace('YR', '').astype(int)

fig, ax = plt.subplots()

countries = ['United Kingdom', 'United States', 'Japan', 'France']
ylabel = '失业率（国家估计） (%)'
plot_comparison(unempl_rate, countries, 
                ylabel, 0.05, None, ax, g_params, 
                b_params, t_params, baseline=None)
plt.show()
```

我们看到，法国由于其强大的工会，在负面冲击之后通常经历相对较慢的劳动力市场恢复。

我们也注意到，日本的失业率一直非常低且稳定。

## 领先指标和相关因素

研究领先指标和相关因素可以帮助政策制定者理解商业周期的起因和结果。

我们将从消费、生产和信贷水平三个角度讨论潜在的领先指标和相关因素。

### 消费

消费取决于消费者对其收入和未来总体经济表现的信心。

一个广泛引用的消费者信心指标是密歇根大学发布的[消费者信心指数](https://fred.stlouisfed.org/series/UMCSENT)。

这里我们绘制了密歇根大学消费者信心指数和1978-2022年美国核心消费者价格指数（CPI）同比变化。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "消费者信心指数和同比CPI变化，美国"
    name: csicpi
tags: [hide-input]
---

start_date = datetime.datetime(1978, 1, 1)
end_date = datetime.datetime(2022, 12, 31)

# 限制绘图范围
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

# 将CPI添加到图例中而不重新绘制线
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


我们看到 

* 消费者信心通常在扩张期间保持高涨，并在衰退前下降。
* 消费者信心与CPI呈明显的负相关关系。

当消费者商品价格上涨时，消费者信心减弱。

这种趋势在[滞涨](https://en.wikipedia.org/wiki/Stagflation)期间尤为显著。

### 生产

实际工业产出与经济中的衰退高度相关。

然而，它并不是一个领先指标，因为生产收缩的峰值相对于消费者信心和通胀而言是延迟的。

我们绘制了1919年至2022年美国的实际工业产出同比变化，以展示这种趋势。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "同比实际产出变化，美国 (%)"
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

我们观察到在衰退期内，图表中存在延迟的收缩现象。

### 信贷水平

信贷收缩通常发生在衰退期间，因为放贷者变得更加谨慎，借款人也更加犹豫是否承担额外债务。

这是因为整体经济活动减少以及对未来预期悲观等因素的影响。

一个例子是英国银行对私营部门的国内信贷。

下图显示了1970年至2022年间，英国银行对私营部门的国内信贷占GDP的百分比。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "银行对私营部门的国内信贷（占GDP百分比）"
    name: dcpc
tags: [hide-input]
---

private_credit = wb.data.DataFrame('FS.AST.PRVT.GD.ZS', 
                ['GBR'], labels=True)
private_credit = private_credit.set_index('Country')
private_credit.columns = private_credit.columns.str.replace('YR', '').astype(int)

fig, ax = plt.subplots()

countries = 'United Kingdom'
ylabel = '信贷水平（占GDP百分比）'
ax = plot_series(private_credit, countries, 
                 ylabel, 0.05, ax, g_params, b_params, 
                 t_params, ylim=None, baseline=None)
plt.show()
```

请注意，信贷在经济扩张期间上升，并在衰退后停滞甚至收缩。