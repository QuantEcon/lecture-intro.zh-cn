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

# 收入和财富不平等


## 概览

在本节中我们将

* 为讲座中采用的技术提供动机
* 导入我们工作中需要的代码库。

### 一些历史

许多历史学家认为，不平等在罗马共和国的衰落中起到了关键作用。

在击败迦太基和入侵西班牙后，钱财流入罗马，大大丰富了当权者。

同时，普通公民被征召去打仗很长时间，财富减少。

由此而来的不平等增长引起了动荡，动摇了共和国的基础。

最终，罗马共和国让位于一系列独裁者，从公元前27年的屋大维（奥古斯都）开始。

这一历史本身就很吸引人，而且我们可以看到与现代世界某些国家的相似之处。

许多近期的政治辩论都围绕不平等展开。

许多经济政策，从税收到福利国家，
都旨在解决不平等。


### 测量

这些辩论的一个问题是，不平等常常定义不明确。

此外，关于不平等的辩论往往与政治信仰联系在一起。

这对经济学家来说是危险的，因为让政治信仰塑造我们的发现会降低客观性。

要以真正科学的视角看待不平等这一话题，我们必须从谨慎的定义开始。

在这节课中我们讨论经济研究中使用的不平等的标准衡量方法。

对于这些衡量方法中的每一个，我们将查看模拟数据和真实数据。

我们将安装以下库。

```{code-cell} ipython3
:tags: [hide-output]

!pip install --upgrade quantecon interpolation
```

图像输入功能：启用

```{code-cell} ipython3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
import random as rd
from interpolation import interp
```

## 洛伦兹曲线

衡量不平等的一个流行方法是洛伦兹曲线。

在本节中我们定义洛伦兹曲线并检查其属性。


### 定义

洛伦兹曲线取一个样本$w_1, \ldots, w_n$并生成一条曲线$L$。

我们假设样本$w_1, \ldots, w_n$已经从最小到最大排序。

为了帮助理解，我们假设我们在测量财富

*  $w_1$是最贫穷人口的财富
*  $w_n$是最富有人口的财富

曲线$L$只是一个函数$y = L(x)$，我们可以绘制和解释。

为了创建它，我们首先根据以下公式生成数据点$(x_i, y_i)$：

\begin{equation*}
    x_i = \frac{i}{n},
    \qquad
    y_i = \frac{\sum_{j \leq i} w_j}{\sum_{j \leq n} w_j},
    \qquad i = 1, \ldots, n
\end{equation*}

现在洛伦兹曲线$L$是用这些数据点插值形成的。

（如果我们在Matplotlib中使用线状图，插值将自动完成。）

声明“$y = L(x)$”的意思是最底层的$(100 \times x)$\%的人拥有全部财富的$(100 \times y)$\%。

* 如果$x=0.5$且$y=0.1$，那么底层50%的人口拥有10%的财富。

在上面的讨论中我们专注于财富，但同样的想法也适用于收入、消费等。

+++

### 模拟数据的洛伦兹曲线

让我们看看一些例子并尝试建立理解。

在下一张图中，我们生成$n=2000$个来自对数正态分布的抽样，并将这些抽样作为我们的样本人口。

直线（所有$x$上的$x=L(x)$）对应于完全平等。

对数正态抽样生成了一个不那么平等的分布。

例如，如果我们将这些抽样看作是家庭财富的观测值，那么虚线显示底层80%的家庭拥有总财富的超过40%。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "模拟数据的洛伦兹曲线"
    name: lorenz_simulated
---
n = 2000
sample = np.exp(np.random.randn(n))

fig, ax = plt.subplots()

f_vals, l_vals = qe.lorenz_curve(sample)
ax.plot(f_vals, l_vals, label=f'对数正态样本', lw=2)
ax.plot(f_vals, f_vals, label='平等', lw=2)

ax.legend(fontsize=12)

ax.vlines([0.8], [0.0], [0.43], alpha=0.5, colors='k', ls='--')
ax.hlines([0.43], [0], [0.8], alpha=0.5, colors='k', ls='--')

ax.set_ylim((0, 1))
ax.set_xlim((0, 1))

plt.show()
```


### 美国数据的洛伦兹曲线

接下来我们来看一下真实数据，重点关注2016年美国的收入和财富。

以下代码块导入数据集 `SCF_plus` 的一个子集，
该数据集来自[消费者财务调查](https://en.wikipedia.org/wiki/Survey_of_Consumer_Finances) (SCF)。

```{code-cell} ipython3
url = 'https://media.githubusercontent.com/media/QuantEcon/high_dim_data/main/SCF_plus/SCF_plus_mini.csv'
df = pd.read_csv(url)
df = df.dropna()
df_income_wealth = df
```

```{code-cell} ipython3
df_income_wealth.head()
```


以下代码块使用存储在数据框 ``df_income_wealth`` 中的数据生成洛伦兹曲线。

（由于我们需要根据 SCF 提供的人口权重调整数据，代码有点复杂。）

```{code-cell} ipython3
:tags: [hide-input]

df = df_income_wealth 

varlist = ['n_wealth',    # 净财富 
           't_income',    # 总收入
           'l_income']    # 劳动收入

years = df.year.unique()

# 创建列表来存储洛伦兹曲线数据

F_vals, L_vals = [], []

for var in varlist:
    # 创建列表来存储洛伦兹曲线数据
    f_vals = []
    l_vals = []
    for year in years:

        # 根据权重重复观测值
        counts = list(round(df[df['year'] == year]['weights'] )) 
        y = df[df['year'] == year][var].repeat(counts)
        y = np.asarray(y)
        
        # 打乱顺序以改进图像
        rd.shuffle(y)    
               
        # 计算并存储洛伦兹曲线数据
        f_val, l_val = qe.lorenz_curve(y)
        f_vals.append(f_val)
        l_vals.append(l_val)
        
    F_vals.append(f_vals)
    L_vals.append(l_vals)

f_vals_nw, f_vals_ti, f_vals_li = F_vals
l_vals_nw, l_vals_ti, l_vals_li = L_vals
```

现在我们为2016年美国的净财富、总收入和劳动收入绘制洛伦兹曲线。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "2016年美国洛伦兹曲线"
    name: lorenz_us
  image:
    alt: lorenz_us
---
fig, ax = plt.subplots()

ax.plot(f_vals_nw[-1], l_vals_nw[-1], label=f'净财富')
ax.plot(f_vals_ti[-1], l_vals_ti[-1], label=f'总收入')
ax.plot(f_vals_li[-1], l_vals_li[-1], label=f'劳动收入')
ax.plot(f_vals_nw[-1], f_vals_nw[-1], label=f'平等')

ax.legend(fontsize=12)   
plt.show()
```

这里所有的收入和财富衡量标准都是税前的。

总收入是家庭所有收入来源的总和，包括劳动收入但不包括资本收益。

这个图的一个关键发现是，财富不平等比收入不平等更为极端。

+++

## 基尼系数

洛伦兹曲线是表示分布不平等的一个有用的视觉工具。

另一个衡量收入和财富不平等的流行方法是基尼系数。

基尼系数只是一个数值，而不是一条曲线。

在本节中我们讨论基尼系数及其与洛伦兹曲线的关系。

### 定义

如前所述，假设样本$w_1, \ldots, w_n$已经从最小到最大排序。

基尼系数定义为

\begin{equation}
    \label{eq:gini}
    G :=
    \frac
        {\sum_{i=1}^n \sum_{j = 1}^n |w_j - w_i|}
        {2n\sum_{i=1}^n w_i}.
\end{equation}

基尼系数与洛伦兹曲线密切相关。

实际上，可以证明其值是等于平等线和洛伦兹曲线之间面积的两倍（例如下面图中的阴影区域）。

其思想是$G=0$表示完全平等，而$G=1$表示完全不平等。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "模拟数据的阴影洛伦兹曲线"
    name: lorenz_gini
  image:
    alt: lorenz_gini
---
fig, ax = plt.subplots()

f_vals, l_vals = qe.lorenz_curve(sample)
ax.plot(f_vals, l_vals, label=f'对数正态样本', lw=2)
ax.plot(f_vals, f_vals, label='平等', lw=2)

ax.legend(fontsize=12)

ax.vlines([0.8], [0.0], [0.43], alpha=0.5, colors='k', ls='--')
ax.hlines([0.43], [0], [0.8], alpha=0.5, colors='k', ls='--')

ax.fill_between(f_vals, l_vals, f_vals, alpha=0.06)

ax.set_ylim((0, 1))
ax.set_xlim((0, 1))

ax.text(0.04, 0.5, r'$G = 2 \times$ 阴影面积', fontsize=12)
  
plt.show()
```


### 示例

让我们使用上面的例子从洛伦兹曲线的阴影区域计算基尼系数。

```{code-cell} ipython3
qe.gini_coefficient(sample)
```

### 模拟数据的基尼系数动态变化

让我们在一些模拟中检查基尼系数。

下面的代码计算了五个不同人口的基尼系数。

每一个人口是通过从具有参数 $\mu$（均值）和 $\sigma$（标准差）的对数正态分布中抽样生成的。

为了创建这五个人口，我们将 $\sigma$ 在 0.2 和 4 之间的长度为 5 的网格上变化。

在每种情况下，我们设定 $\mu = - \sigma^2 / 2$。

这意味着分布的均值不会随 $\sigma$ 改变。

（你可以通过查看对数正态分布的均值表达式来验证这一点。）

```{code-cell} ipython3
k = 5
σ_vals = np.linspace(0.2, 4, k)
n = 2_000

ginis = []

for σ in σ_vals:
    μ = -σ**2 / 2
    y = np.exp(μ + σ * np.random.randn(n))
    ginis.append(qe.gini_coefficient(y))
```


接下来我们绘制$\sigma$和相应基尼系数$G$的值。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "标准差与基尼系数"
    name: gini_std
  image:
    alt: gini_std
---
fig, ax = plt.subplots()

ax.plot(σ_vals, ginis, label='基尼系数')

ax.legend(fontsize=12)

ax.set_xlabel('$sigma$', fontsize=12)
ax.set_ylabel('$G$', fontsize=12)

plt.show()
```

图表显示了$\sigma$对基尼系数有显著影响（尽管$\sigma$影响分布的方差，而不是均值）。

现在我将基尼系数的上述代码封装到一个函数中。

```{code-cell} ipython3
def gini_coeff_vs_sigma(σ_vals, n=2_000):
    """
    给定不同行参数值后计算基尼系数
    """

    ginis = []

    for σ in σ_vals:
        μ = -σ**2 / 2  # 设定均值恒定
        y = np.exp(μ + σ * np.random.randn(n))
        ginis.append(qe.gini_coefficient(y))
    
    return ginis
```

现在我们运行该函数并将基尼系数对$\sigma$绘制出来。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "标准差与基尼系数"
    name: gini_std_function
  image:
    alt: gini_std_function
---
σ_vals = np.linspace(0.2, 4, 100)
ginis = gini_coeff_vs_sigma(σ_vals)

fig, ax = plt.subplots()

ax.plot(σ_vals, ginis, label='基尼系数')

ax.legend(fontsize=12)

ax.set_xlabel('$sigma$', fontsize=12)
ax.set_ylabel('$G$', fontsize=12)

plt.show()
```


### 模拟的不平等度量

为了理解不平等度量在模拟中的表现，让我们看看下面的例子。

这段代码首先生成一组具有高度不平等的随机数据集，然后重新计算洛伦兹曲线和基尼系数。

```{code-cell} ipython3
def plot_inequality_measures(x_vals, y_vals, label, x_label, y_label):
    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label=f'{label}数据', lw=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(fontsize=12)
    plt.show()

σ_vals = np.linspace(0.2, 4, 100)
ginis = gini_coeff_vs_sigma(σ_vals)

plot_inequality_measures(σ_vals, 
                         ginis, 
                         '模拟', 
                         '$sigma$', 
                         '基尼系数')
```

图表显示了随着$\sigma$增加，不平等度（基尼系数）也在增加。

+++

### 美国数据的基尼系数动态变化

现在让我们看看从SCF数据框架 ``df_income_wealth`` 中得出的美国数据的基尼系数。

以下代码创建一个名为 ``Ginis`` 的列表。

 它从数据框 ``df_income_wealth`` 和 [QuantEcon](https://quantecon.org/quantecon-py/) 库中的方法 [gini_coefficient](https://quanteconpy.readthedocs.io/en/latest/tools/inequality.html#quantecon.inequality.gini_coefficient) 中生成基尼系数数据。

```{code-cell} ipython3
:tags: [hide-input]

varlist = ['n_wealth',   # 净财富 
           't_income',   # 总收入
           'l_income']   # 劳动收入

df = df_income_wealth

# 创建列表来存储每个不平等度量的基尼系数

Ginis = []

for var in varlist:
    # 创建列表来存储基尼系数
    ginis = []
    
    for year in years:
        # 根据权重重复观测值
        counts = list(round(df[df['year'] == year]['weights'] ))
        y = df[df['year'] == year][var].repeat(counts)
        y = np.asarray(y)
        
        rd.shuffle(y)    # 打乱顺序
        
        # 计算并存储基尼系数
        gini = qe.gini_coefficient(y)
        ginis.append(gini)
        
    Ginis.append(ginis)
```

现在我们从``Ginis``列表中生成分别代表净财富、总收入和劳动收入的三组基尼系数数据。

为了更清楚地看到每个变量的趋势，我们将每年的数据绘制在一张图表中并添加图例。

```{code-cell} ipython3
ginis_nw, ginis_ti, ginis_li = Ginis

fig, ax = plt.subplots()

ax.plot(years, ginis_nw, label=f'净财富')
ax.plot(years, ginis_ti, label=f'总收入')
ax.plot(years, ginis_li, label=f'劳动收入')

ax.legend(fontsize=12)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('基尼系数', fontsize=12)

plt.show()
```

**注意**： 这里有一个基尼系数的异常值。
为了分析更具代表性的趋势，让我们将该异常值替换为相邻年份的平均值。

```{code-cell} ipython3
# 用平均值替换劳动收入基尼的一个异常值
ginis_li_new = ginis_li
ginis_li_new[5] = (ginis_li[4] + ginis_li[6]) / 2
```


```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "美国收入和财富的基尼系数（无异常值）"
    name: gini_no_outliers
  image:
    alt: gini_no_outliers
---
fig, ax = plt.subplots()

ax.plot(years, ginis_nw, label=f'净财富')
ax.plot(years, ginis_ti, label=f'总收入')
ax.plot(years, ginis_li_new, label=f'劳动收入')

ax.legend(fontsize=12)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('基尼系数', fontsize=12)

plt.show()
```
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Gini coefficients of US income"
    name: gini_income_us
  image:
    alt: gini_income_us
---
xlabel = "年份"
ylabel = "基尼系数"

fig, ax = plt.subplots()

ax.plot(years, ginis_li_new, marker='o', label="劳动收入")
ax.plot(years, ginis_ti, marker='o', label="总收入")

ax.set_xlabel(xlabel, fontsize=12)
ax.set_ylabel(ylabel, fontsize=12)

ax.legend(fontsize=12)
plt.show()
```

图表显示，收入和财富不平等都在增加。

此现象特别自1980年以来尤为明显。

+++

## 顶层份额

另一个受欢迎的不平等衡量方法是顶层份额。

计算特定份额较洛伦兹曲线或基尼系数更简单。

在本节中我们展示如何计算顶层份额。

### 定义

如前所述，假设样本$w_1, \ldots, w_n$已经从最小到最大排序。

给定如上定义的洛伦兹曲线 $y = L(x)$，顶层$100 \times p \%$
份额定义为

$$
T(p) = 1 - L (1-p) 
    \approx \frac{\sum_{j\geq i} w_j}{ \sum_{j \leq n} w_j}, \quad i = \lfloor n (1-p)\rfloor
$$(topshares)

这里的 $\lfloor \cdot \rfloor$ 是地板函数，它将任何
数字向下舍入到小于或等于该数字的整数。

+++

下面的代码使用数据框 ``df_income_wealth`` 中的数据生成另一个数据框 ``df_topshares``。

``df_topshares`` 存储了1950年至2016年间美国总收入、劳动收入和净财富的前10%份额。

```{code-cell} ipython3
:tags: [hide-input]

# 将调查权重从绝对值转换为相对值
df1 = df_income_wealth
df2 = df1.groupby('year').sum(numeric_only=True).reset_index()
df3 = df2[['year', 'weights']]
df3.columns = 'year', 'r_weights'
df4 = pd.merge(df3, df1, how="left", on=["year"])
df4['r_weights'] = df4['weights'] / df4['r_weights']

# 创建加权净财富、总收入、劳动收入

df4['weighted_n_wealth'] = df4['n_wealth'] * df4['r_weights']
df4['weighted_t_income'] = df4['t_income'] * df4['r_weights']
df4['weighted_l_income'] = df4['l_income'] * df4['r_weights']

# 提取按净财富和总收入前10%分组的两个群体

df6 = df4[df4['nw_groups'] == 'Top 10%']
df7 = df4[df4['ti_groups'] == 'Top 10%']

# 计算按净财富、总收入和劳动收入加权的前10%之和

df5 = df4.groupby('year').sum(numeric_only=True).reset_index()
df8 = df6.groupby('year').sum(numeric_only=True).reset_index()
df9 = df7.groupby('year').sum(numeric_only=True).reset_index()

df5['weighted_n_wealth_top10'] = df8['weighted_n_wealth']
df5['weighted_t_income_top10'] = df9['weighted_t_income']
df5['weighted_l_income_top10'] = df9['weighted_l_income']

# 计算这三个变量的前10%份额

df5['topshare_n_wealth'] = df5['weighted_n_wealth_top10'] / \
    df5['weighted_n_wealth']
df5['topshare_t_income'] = df5['weighted_t_income_top10'] / \
    df5['weighted_t_income']
df5['topshare_l_income'] = df5['weighted_l_income_top10'] / \
    df5['weighted_l_income']

# 我们只需要这些变量用于前10%份额
df_topshares = df5[['year', 'topshare_n_wealth',
                    'topshare_t_income', 'topshare_l_income']]
```

### 美国数据的顶层份额

然后我们绘制前10%份额。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "美国顶层份额"
    name: top_shares_us
  image:
    alt: top_shares_us
---
xlabel = "年份"
ylabel = "顶层 $10\%$ 份额"

fig, ax = plt.subplots()

ax.plot(years, df_topshares["topshare_l_income"],
        marker='o', label="劳动收入")
ax.plot(years, df_topshares["topshare_n_wealth"],
        marker='o', label="净财富")
ax.plot(years, df_topshares["topshare_t_income"],
        marker='o', label="总收入")

ax.set_xlabel(xlabel, fontsize=12)
ax.set_ylabel(ylabel, fontsize=12)

ax.legend(fontsize=12)
plt.show()
```


## 练习

+++

```{exercise}
:label: inequality_ex1

使用仿真计算与随机变量$w_\sigma = \exp(\mu + \sigma Z)$相关的对数正态分布集合的
前10%份额，其中$Z \sim N(0, 1)$且$\sigma$在$0.2$到$4$的有限网格上变化。

随着$\sigma$的增加，$w_\sigma$的方差也增加。

为了关注波动性，在每一步调整$\mu$以保持等式
$\mu=-\sigma^2/2$。

对于每个$\sigma$，生成$w_\sigma$的2000个独立抽样并
计算洛伦兹曲线和基尼系数。

确认较高的方差
在样本中产生更多的离散性，从而导致更大的不平等。
```

+++

```{solution-start} inequality_ex1
:class: dropdown
```

这是一个解决方案：

```{code-cell} ipython3
def calculate_top_share(s, p=0.1):
    
    s = np.sort(s)
    n = len(s)
    index = int(n * (1 - p))
    return s[index:].sum() / s.sum()
```

为了计算和生成图表，我们使用以下代码：

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import quantecon as qe

# 定义参数
sigma_vals = np.linspace(0.2, 4, 20)
p = 0.1
n = 2000

top_shares = []
gini_coeffs = []

# 进行仿真
for sigma in sigma_vals:
    mu = -sigma**2 / 2
    w_sigma = np.exp(mu + sigma * np.random.randn(n))
    top_shares.append(calculate_top_share(w_sigma, p=p))
    gini_coeffs.append(qe.gini_coefficient(w_sigma))

# 生成图表
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

ax[0].plot(sigma_vals, top_shares, 'b-', label=f"前 {int(p*100)}% 份额")
ax[0].set_xlabel("标准差 $\sigma$")
ax[0].set_ylabel("前10%份额")
ax[0].legend()

ax[1].plot(sigma_vals, gini_coeffs, 'r-', label="基尼系数")
ax[1].set_xlabel("标准差 $\sigma$")
ax[1].set_ylabel("基尼系数")
ax[1].legend()

plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
k = 5
σ_vals = np.linspace(0.2, 4, k)
n = 2_000

topshares = []
ginis = []
f_vals = []
l_vals = []

for σ in σ_vals:
    μ = -σ ** 2 / 2
    y = np.exp(μ + σ * np.random.randn(n))
    f_val, l_val = qe._inequality.lorenz_curve(y)
    f_vals.append(f_val)
    l_vals.append(l_val)
    ginis.append(qe._inequality.gini_coefficient(y))
    topshares.append(calculate_top_share(y))
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "仿真数据的顶层份额"
    name: top_shares_simulated
  image:
    alt: top_shares_simulated
---
plot_inequality_measures(σ_vals, 
                         topshares, 
                         "仿真数据", 
                         "$\sigma$", 
                         "顶层 $10\%$ 份额") 
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "仿真数据的基尼系数"
    name: gini_coef_simulated
  image:
    alt: gini_coef_simulated
---
plot_inequality_measures(σ_vals, 
                         ginis, 
                         "仿真数据", 
                         "$\sigma$", 
                         "基尼系数") 
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "仿真数据的洛伦兹曲线"
    name: lorenz_curve_simulated
  image:
    alt: lorenz_curve_simulated
---
fig, ax = plt.subplots()
ax.plot([0,1],[0,1], label=f"平等")
for i in range(len(f_vals)):
    ax.plot(f_vals[i], l_vals[i], label=f"$\sigma$ = {σ_vals[i]}")
plt.legend()
plt.show()
```
```{solution-end}
```

```{exercise}
:label: inequality_ex2

根据顶层份额的定义{eq}`topshares`，我们还可以使用洛伦兹曲线计算顶层百分比份额。

使用相应的洛伦兹曲线数据 ``f_vals_nw, l_vals_nw`` 和线性插值计算美国净财富的顶层份额。

将洛伦兹曲线生成的顶层份额与从数据近似得到的顶层份额一起绘制出来。

```

+++

```{solution-start} inequality_ex2
:class: dropdown
```

这是一个解决方案：

```{code-cell} ipython3
def lorenz2top(f_val, l_val, p=0.1):
    t = lambda x: interp(f_val, l_val, x)
    return 1- t(1 - p)
```

```{code-cell} ipython3
top_shares_nw = []
for f_val, l_val in zip(f_vals_nw, l_vals_nw):
    top_shares_nw.append(lorenz2top(f_val, l_val))
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "美国顶层份额：近似值与洛伦兹曲线"
    name: top_shares_us_al
  image:
    alt: top_shares_us_al
---
xlabel = "年份"
ylabel = "顶层 $10\%$ 份额"

fig, ax = plt.subplots()

ax.plot(years, df_topshares["topshare_n_wealth"], marker='o',\
   label="净财富-近似值")
ax.plot(years, top_shares_nw, marker='o', label="净财富-洛伦兹曲线")

ax.set_xlabel(xlabel, fontsize=12)
ax.set_ylabel(ylabel, fontsize=12)

ax.legend(fontsize=12)
plt.show()
```

```{solution-end}
```

Image input capabilities: Enabled