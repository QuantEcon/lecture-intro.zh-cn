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

(geom_series)=
```{raw} jupyter
<div id="qe-notebook-header" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

```{index} single: python
```

# 几何级数在基础经济学中的应用

```{admonition} 已迁移的讲座
:class: warning

本讲座已从我们[中级定量经济学 Python](https://python.quantecon.org/intro.html)讲座系列迁移，现在是[定量经济学入门课程](https://intro.quantecon.org/intro.html)的一部分。
```

## 概述

本讲座描述了在经济学中使用几何级数数学的一些重要思想。

其中包括

- 凯恩斯的**乘数**
- 在部分准备金银行系统中普遍存在的货币**乘数**
- 利率和资产收益流的现值

（正如我们将在下面看到的，术语**乘数**归结为**收敛几何级数的和**）

这些及其他应用证明了以下箴言的正确性

```{epigraph}
"在经济学中，对几何级数的一点点了解会带来很大的帮助 "
```

下面我们将使用以下导包：

```{code-cell} ipython
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  # 设置默认图尺寸
import numpy as np
import sympy as sym
from sympy import init_printing
from matplotlib import cm
```
图像输入功能：已启用

## 关键公式

首先，让 $c$ 是一个严格介于 $-1$ 和 $1$ 之间的实数。

- 我们通常写作 $c \in (-1,1)$。
- 这里 $(-1,1)$ 表示所有严格小于 $1$ 且严格大于 $-1$ 的实数的集合。
- 符号 $\in$ 表示*在*或*属于符号后的集合*。

我们想要评估两类几何级数——无限和有限。

### 无限几何级数

我们感兴趣的第一类几何级数是无限级数

$$
1 + c + c^2 + c^3 + \cdots
$$

其中 $\cdots$ 表示该级数永远继续下去。

关键公式是

```{math}
:label: infinite

1 + c + c^2 + c^3 + \cdots = \frac{1}{1 -c }
```

为了证明关键公式 {eq}`infinite`，将两边乘以 $(1-c)$ 并验证如果 $c \in (-1,1)$，则结果是 $1 = 1$ 的等式。

### 有限几何级数

我们感兴趣的第二类级数是有限几何级数

$$
1 + c + c^2 + c^3 + \cdots + c^T
$$

其中 $T$ 是一个正整数。

这里的关键公式是

$$
1 + c + c^2 + c^3 + \cdots + c^T  = \frac{1 - c^{T+1}}{1-c}
$$

**备注：** 上面的公式适用于任何标量 $c$ 的值。我们不必将 $c$ 限制在集合 $(-1,1)$ 中。

现在我们继续描述几何级数的一些著名的经济应用。

## 示例：分数准备金银行系统中的货币乘数

在分数准备金银行系统中，银行仅持有其发行的**存款单**背后的现金的一部分 $r \in (0,1)$

* 近年来
    - 现金由政府发行的称为美元或英镑或 $\ldots$ 的纸币组成
    - *存款* 是支票帐户或储蓄帐户中的余额，持有者有权要求银行立即支付现金
* 当英国、法国和美国在金本位或银本位制（例如 1914 年之前）时
    - 现金是金币或银币
    - *存款单* 是银行承诺按需转换为黄金或白银的*银行票据*；（有时它也是支票帐户或储蓄帐户的余额）

经济学家和金融家通常将**货币供应量**定义为全经济范围内的**现金**加**存款**的总和。

在**分数准备金银行系统**（保留率 $r$ 符合 $0 < r < 1$ 的系统）中，**银行通过**发行以分数准备金为*后盾*的存款和发放给客户的贷款*创造货币**。

几何级数是理解银行如何在分数准备金系统中创造货币（即存款）的关键工具。

几何级数公式 {eq}`infinite` 是货币创造过程经典模型的核心——该模型引导我们到达著名的**货币乘数**。

### 一个简单的模型

有一组银行，命名为 $i = 0, 1, 2, \ldots$。

银行 $i$ 的贷款 $L_i$，存款 $D_i$ 和储备 $R_i$ 必须满足资产负债表等式（因为**资产负债表需要平衡**）：

```{math}
:label: balance
```

```{math}
L_i + R_i = D_i
```

上述公式的左边是银行的**资产**之和，即未偿还的贷款 $L_i$ 加上其现金准备金 $R_i$。

公式的右边是银行 $i$ 的负债，即其储户持有的存款 $D_i$；这些存款是银行对储户的欠条，形式是支票账户或储蓄账户（或在 1914 年之前，银行发行的承诺按需兑换黄金或白银的银行票据）。

每个银行 $i$ 设定其准备金满足方程

```{math}
:label: reserves
```

```{math}
R_i = r D_i
```

其中 $r \in (0,1)$ 是银行的**准备金-存款比率**或简称为**准备金比率**

- 准备金比率要么由政府设定，要么由银行出于预防目的自定

接下来我们添加一个理论，表明银行 $i+1$ 的存款完全取决于银行 $i$ 发放的贷款，即

```{math}
:label: deposits
```

```{math}
D_{i+1} = L_i
```

因此，我们可以认为这些银行按顺序排列，银行 $i$ 的贷款立即存入银行 $i+1$

- 这样，银行 $i$ 的债务人成为银行 $i+1$ 的债权人

最后，我们添加一个关于银行 $0$ 的存款外生水平的*初始条件*

$$
D_0 \ \text{ 是外生给定的 }
$$

我们可以认为 $D_0$ 是第一个储户存入系统中第一家银行（编号 $i=0$）的现金。

现在我们做一些代数运算。

结合公式 {eq}`balance` 和 {eq}`reserves` 得出

```{math}
:label: fraction
```

```{math}
L_i = (1-r) D_i
```

这表明银行 $i$ 将其存款的 $(1-r)$ 部分贷出，并保留 $r$ 部分作为现金准备金。

将公式 {eq}`fraction` 与公式 {eq}`deposits` 结合起来，可以得出

$$
D_{i+1} = (1-r) D_i  \ \text{ 对于 } i \geq 0
$$

这意味着

```{math}
:label: geomseries
```

```{math}
D_i = (1 - r)^i D_0  \ \text{ 对于 } i \geq 0
```

公式 {eq}`geomseries` 表示 $D_i$ 是 $D_0$ 和几何级数的 $i$ 项的乘积

$$
1, (1-r), (1-r)^2, \cdots
$$

因此，我们银行系统中所有存款的总和（从 $i=0, 1, 2, \ldots$）是

```{math}
:label: sumdeposits
```

```{math}
\sum_{i=0}^\infty (1-r)^i D_0 =  \frac{D_0}{1 - (1-r)} = \frac{D_0}{r}
```

### 货币乘数

**货币乘数**是一个数值，表示对银行 $0$ 的现金外生注入将导致银行系统中总存款增加的倍数。

公式 {eq}`sumdeposits` 声明 **货币乘数** 是 $\frac{1}{r}$

- 银行 $0$ 的初始现金存款 $D_0$ 导致整个银行系统创造总存款 $\frac{D_0}{r}$。
- 初始存款 $D_0$ 持有为准备金，分布在整个银行系统中，满足公式 $D_0 = \sum_{i=0}^\infty R_i$。

## 示例：凯恩斯乘数

著名经济学家约翰·梅纳德·凯恩斯及其追随者创造了一个简单模型，旨在确定国民收入 $y$ 在......的情况下。

- 有大量的失业资源，尤其是劳动和资本的**超额供给**。
- 价格和利率不能调整以使总**供给等于需求**（例如，价格和利率被冻结）。
- 国民收入完全由总需求决定。

### 静态版本

一个简单的凯恩斯国民收入决定模型包括
描述总需求 $y$ 及其组成部分的三个方程。

第一个方程是一个国民收入恒等式，断言消费 $c$ 加上投资 $i$ 等于国民收入
$y$：

$$
c + i = y
$$

第二个方程是一个凯恩斯消费函数，断言
人们消费其收入的一部分 $b \in (0, 1)$：

$$
c = b y
$$

部分 $b \in (0, 1)$ 被称为**边际消费倾向**。

部分 $1-b \in (0, 1)$ 被称为**边际储蓄倾向**。

第三个方程简单地说明投资是外生的，水平为 $i$。

- *外生*意味着*由该模型外部决定*。

将第二个方程代入第一个方程，得到 $(1-b) y = i$。

解出 $y$ 得

$$
y = \frac{1}{1-b} i
$$

量 $\frac{1}{1-b}$ 称为**投资乘数**或简称**乘数**。

应用一个无穷几何级数的和公式，我们可以写出上述方程为

$$
y = i \sum_{t=0}^\infty b^t
$$

其中 $t$ 是非负整数。

所以我们得到乘数的以下等效表达式：

$$
\frac{1}{1-b} = \sum_{t=0}^\infty b^t
$$

表达式 $\sum_{t=0}^\infty b^t$ 激发了我们对乘数的解释，即为我们接下来描述的动态过程的结果。

### 动态版本

通过将非负整数 $t$ 解释为时间索引，并更改我们的消费函数规范以考虑时间，我们得出动态版本

- 我们添加了一个*滞后一期*的收入如何影响消费

我们设 $c_t$ 为时间 $t$ 的消费，$i_t$ 为时间 $t$ 的投资

我们修正我们的消费函数假设公式为

$$
c_t = b y_{t-1}
$$

使得 $b$ 是上一期收入的*边际消费倾向*。

我们从一个初始条件开始，声明

$$
y_{-1} = 0
$$

我们还假设

$$
i_t = i \ \ \textrm {对所有 }  t \geq 0
$$

使得投资随时间保持不变。

因此，

$$
y_0 = i + c_0 = i + b y_{-1} = i
$$

并且

$$
y_1 = c_1 + i = b y_0 + i = (1 + b) i
$$

并且

$$
y_2 = c_2 + i = b y_1 + i = (1 + b + b^2) i
$$

更普遍地

$$
y_t = b y_{t-1} + i = (1+ b + b^2 + \cdots + b^t) i
$$

或

$$
y_t = \frac{1-b^{t+1}}{1 -b } i
$$

显然，随着 $t \rightarrow + \infty$，

$$
y_t \rightarrow \frac{1}{1-b} i
$$

**备注 1:** 上述公式通常用于断言，在时间 $0$ 外生增加投资 $\Delta i$ 点燃了一系列连续的增加国民收入的动态过程

$$

\Delta i, (1 + b )\Delta i, (1+b + b^2) \Delta i , \cdots
$$

在时间 $0, 1, 2, \cdots$。

**备注 2** 令 $g_t$ 为外生的政府支出序列。

如果我们推广模型使得国民收入恒等式变为

$$
c_t + i_t + g_t  = y_t
$$

那么前面论证的一个版本表明，**政府支出乘数**也是 $\frac{1}{1-b}$，因此政府支出的永久增加最终导致国民收入增加，等于乘数乘以政府支出的增加量。

## 示例：利率与现值

我们可以应用几何级数公式来研究利率如何影响延续到未来的美元支付流的价值。

我们在离散时间内工作，并假设 $t = 0, 1, 2, \cdots$ 索引时间。

我们令 $r \in (0,1)$ 为一期**净名义利率**

- 如果名义利率为 5%，那么 $r= .05$

一期**总名义利率** $R$ 定义为

$$
R = 1 + r \in (1, 2)
$$

- 如果 $r=.05$，那么 $R = 1.05$

**备注：** 总名义利率 $R$ 是在时间 $t$ 和 $t+1$ 之间美元的**汇率**或**相对价格**。 $R$ 的单位是 $t$ 时的美元到 $t+1$ 时的美元。

当人们借贷时，他们用现在的美元交换未来的美元或用未来的美元交换现在的美元。

这些兑换的价格就是总名义利率。

- 如果我今天卖给你 $x$ 美元，你明天就支付我 $R x$ 美元。
- 这意味着你以总利率 $R$ 和净利率 $r$ 向我借了 $x$ 美元。

我们假设净名义利率 $r$ 在整个时间内保持不变，因此 $R$ 是在时间 $t=0, 1, 2, \cdots$ 的总名义利率。

两个重要的几何序列是

```{math}
:label: geom1

1, R, R^2, \cdots
```

和

```{math}
:label: geom2

1, R^{-1}, R^{-2}, \cdots
```

序列 {eq}`geom1` 告诉我们投资**累计**的美元价值随时间的变化。

序列 {eq}`geom2` 告诉我们如何**折现**未来的美元，以便在今天的美元中计算其价值。

### 累积

几何序列 {eq}`geom1` 告诉我们一美元投资并再投资在一个总一时期名义回报率的项目中的累积情况。

- 这里我们假设净利息收入再投资于项目中
- 因此，时间 $0$ 投资的 $1$ 美元在一段时间后产生 $r$ 美元的利息，因此我们在时间 $1$ 拥有 $1+r = R$ 美元。
- 在时间 $1$ 我们再投资 $1+r =R$ 美元，并在时间 $2$ 获得 $r R$ 美元的利息加上*本金* $R$ 美元，因此我们在第 $2$ 期末获得 $r R + R = (1+r)R = R^2$ 美元。
- 依此类推

显然，如果我们在时间 $0$ 投资 $x$ 美元并将收益再投资，那么序列

$$
x , xR , x R^2, \cdots
$$

告诉我们账户余额在时间 $t=0, 1, 2, \cdots$ 的累积情况。

### 折现

几何序列 {eq}`geom2` 告诉我们以今天的美元计算未来几期的**美元折现值**。

我们考虑用于剩余转换的公式。

- 单位 $R$ 是 $t+1$ 期美元/ $t$ 期美元
- 这就意味着单位 $R^{-1}$ 是 $t$ 期美元/ $t+1$ 期美元。
- 单位 $R^{-j}$ 是 $t$ 期美元/ $t+j$ 期美元。

几何序列 {eq}`geom2` 告诉我们用今天的美元计算未来几期美元的价值。

假设有人在 $t+j$ 期有 $x$ 美元的索赔权，其在 $t$ 期的价值是 $x R^{-j}$ 美元（例如，今天）。

### 应用于资产定价

一个**租赁**需要在 $t = 0, 1, 2, \ldots$ 时期支付 $x_t$ 美元的支付流，其中

$$
x_t = G^t x_0
$$

其中 $G = (1+g)$ 并且 $g \in (0,1)$。

因此，租赁支付随着每期增幅 $g$ 百分比增加。

由于某些即将揭示的原因，我们假设 $G < R$。

租赁的**现值**为

$$
\begin{aligned} 
p_0  & = x_0 + x_1/R + x_2/(R^2) + \cdots \\
     & = x_0 (1 + G R^{-1} + G^2 R^{-2} + \cdots ) \\
     & = x_0 \frac{1}{1 - G R^{-1}} 
\end{aligned}
$$

最后一行使用了无限几何级数的公式。

回想 $R = 1+r$ 和 $G = 1+g$ 并且 $R > G$
和 $r > g$ 并且 $r$ 和 $g$ 通常是小数，例如, .05 或 .03。

使用泰勒级数 $\frac{1}{1+r}$ 展开关于 $r=0$，

$$
\frac{1}{1+r} = 1 - r + r^2 - r^3 + \cdots
$$

事实上 $r$ 很小，可以近似

$$
\frac{1}{1+r} \approx 1 - r
$$

用这个近似表达 $p_0$ 为

$$
\begin{aligned}
 p_0 &= x_0 \frac{1}{1 - G R^{-1}} \\
     &= x_0 \frac{1}{1 - (1+g) (1-r) } \\
     &= x_0 \frac{1}{1 - (1+g - r - rg)} \\
     & \approx x_0 \frac{1}{r -g }
\end{aligned}
$$

最后一步使用了近似 $r g \approx 0$。

近似公式

$$
p_0 = \frac{x_0 }{r -g }
$$

被称为当名义一期间利率为 $r$ 且 $r > g$ 时，无限支付流 $x_0 G^t$ 的现值或当前价格的**戈登公式**。

我们还可以扩展资产定价公式，使其适用于有限租赁。

假设租赁的支付流为 $x_t$，对于 $t= 1,2, \dots,T$，其中再次

$$
x_t = G^t x_0
$$

该租赁的现值为：

$$
\begin{aligned} 
\begin{split}
p_0&=x_0 + x_1/R  + \dots +x_T/R^T \\
   &= x_0(1+GR^{-1}+\dots +G^{T}R^{-T}) \\
   &= \frac{x_0(1-G^{T+1}R^{-(T+1)})}{1-GR^{-1}}  
\end{split}
\end{aligned}
$$

对于 $R^{-(T+1)}$ 应用泰勒级数展开关于 $r=0$ 我们得到：

$$
\frac{1}{(1+r)^{T+1}}= 1-r(T+1)+\frac{1}{2}r^2(T+1)(T+2)+\dots \approx 1-r(T+1)
$$

同样，对于 $G^{T+1}$ 应用泰勒级数展开关于 $g=0$：

$$
(1+g)^{T+1} = 1+(T+1)g+\frac{T(T+1)}{2!}g^2+\frac{(T-1)T(T+1)}{3!}g^3+\dots \approx 1+ (T+1)g
$$

于是我们得到以下近似：

$$
p_0 =\frac{x_0(1-(1+(T+1)g)(1-r(T+1)))}{1-(1-r)(1+g) }
```
```

```{math}
\begin{aligned} 
p_0 &=\frac{x_0(1-1+(T+1)^2 rg -r(T+1)+g(T+1))}{1-1+r-g+rg}  \\
&=\frac{x_0(T+1)((T+1)rg+r-g)}{r-g+rg} \\
&\approx \frac{x_0(T+1)(r-g)}{r-g}+\frac{x_0rg(T+1)}{r-g}\\ 
&= x_0(T+1) + \frac{x_0rg(T+1)}{r-g}  
\end{aligned}
```
我们其实也可以通过移除第二项 $rgx_0(T+1)$ 来近似，当 $T$ 相对较小且与 $1/(rg)$ 比较时，可以得到有限流的近似 $x_0(T+1)$。

我们将在 Python 中绘制真实的有限流现值和两种近似值，在不同的 $T$、$g$ 和 $r$ 值下。

首先我们计算出真实的有限流现值并绘制

```{code-cell} ipython3
# True present value of a finite lease
def finite_lease_pv_true(T, g, r, x_0):
    G = (1 + g)
    R = (1 + r)
    return (x_0 * (1 - G**(T + 1) * R**(-T - 1))) / (1 - G * R**(-1))
# First approximation for our finite lease

def finite_lease_pv_approx_1(T, g, r, x_0):
    p = x_0 * (T + 1) + x_0 * r * g * (T + 1) / (r - g)
    return p

# Second approximation for our finite lease
def finite_lease_pv_approx_2(T, g, r, x_0):
    return (x_0 * (T + 1))

# Infinite lease
def infinite_lease(g, r, x_0):
    G = (1 + g)
    R = (1 + r)
    return x_0 / (1 - G * R**(-1))
'''
```

现在我们已经定义好了我们的函数，可以绘制一些结果。

首先我们研究我们近似的质量

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "有限租赁在 $T$ 期时的现值"
    name: finite_lease_present_value
---
def plot_function(axes, x_vals, func, args):
    axes.plot(x_vals, func(*args), label=func.__name__)

T_max = 50

T = np.arange(0, T_max+1)
g = 0.02
r = 0.03
x_0 = 1

our_args = (T, g, r, x_0)
funcs = [finite_lease_pv_true,
        finite_lease_pv_approx_1,
        finite_lease_pv_approx_2]
        # 我们想要比较的三个函数

fig, ax = plt.subplots()
for f in funcs:
    plot_function(ax, T, f, our_args)
ax.legend()
ax.set_xlabel('$T$ 期')
ax.set_ylabel('现值, $p_0$')
plt.show()
```

显然我们的近似在 $T$ 小时运行良好。

但是，保持 $g$ 和 $r$ 不变，我们的近似在 $T$ 增加时会恶化。

接下来我们比较 无限与 有限期租赁现值 在不同租赁期 $T$ 下的表现。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "无限和有限租赁现值 在 $T$ 期后的比较"
    name: infinite_and_finite_lease_present_value
---
# 无限与有限的收敛
T_max = 1000
T = np.arange(0, T_max+1)
fig, ax = plt.subplots()
f_1 = finite_lease_pv_true(T, g, r, x_0)
f_2 = np.full(T_max+1, infinite_lease(g, r, x_0))
ax.plot(T, f_1, label='T期租赁现值')
ax.plot(T, f_2, '--', label='无限期租赁现值')
ax.set_xlabel('$T$ 期')
ax.set_ylabel('现值, $p_0$')
ax.legend()
plt.show()
```

显然，我们的有限期租赁随着租赁期的增加，其现值趋近于无限期租赁的现值。

### 不同 $r$ 和 $g$ 的影响

上面的图表展示了当租赁期 $T \rightarrow +\infty$ 时，租赁的现值如何接近无限期租赁的现值。

现在我们考虑 $r$ 和 $g$ 协变时的两种不同观点

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "不同租赁期限 $T$ 的租赁现值"
    name: value_of_lease
---
# 第一种观点
# 改变 r 和 g
fig, ax = plt.subplots()
ax.set_ylabel('现值, $p_0$')
ax.set_xlabel('$T$ 期数')
T_max = 10
T=np.arange(0, T_max+1)

rs, gs = (0.9, 0.5, 0.4001, 0.4), (0.4, 0.4, 0.4, 0.5),
comparisons = ('$\gg$', '$>$', r'$\approx$', '$<$')
for r, g, comp in zip(rs, gs, comparisons):
    ax.plot(finite_lease_pv_true(T, g, r, x_0), label=f'r(={r}) {comp} g(={g})')

ax.legend()
plt.show()
```

显然，当 $r \approx g$ 时，现值不再是固定的常数。

接下来我们将 $r=g$ 的租赁期：

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "租赁期限 $T=50$， $r$ 和 $g$ 的不同组合"
    name: T_50_different_r_g
---
# 第二种观点
# T=50 与r和g变化
T = 50
all_rs = np.arange(0.01, 0.99, 0.01)
all_gs = [0.01, 0.02, 0.03, 0.06]
fig, ax = plt.subplots()
for g in all_gs:
    out = finite_lease_pv_true(T, g, all_rs, x_0)
    ax.plot(finite_lease_pv_true(T, g, all_rs, x_0), label=f'g(={g})')
ax.legend()
plt.yscale('log')
ax.set_ylabel('现值, $p_0$')
ax.set_xlabel('$r$')
plt.show()
```

我们显然看到租赁需要很大时期 $T$ 才能便宜得多，但当$g$趋近于 $r$时租赁的现值趋于无限。再不同$g$情况下对项目变高的租赁现值进行了重新评估并与之前的计算作比较，如通过不同$g$计算时对r的某些存在价值比我们先前测试所报通过$\infty值低。但由于我们很少在计算机上工作绝对无穷。我们来看 3-d图显示。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "50 年期租赁现值的三维图"
    name: value_of_lease_3d
---
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
rs = np.arange(0.001, 0.10, 0.005)
gs = np.arange(0.001, 0.10, 0.005)
X, Y = np.meshgrid(rs, gs)
zs = np.array([finite_lease_pv_true(50, g, r, 1) for r, g in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z, cmap=cm.viridis)
ax.set_xlabel('$r$')
ax.set_ylabel('$g$')
ax.set_zlabel('$p_0$')
plt.show()
```

显然，假设支付现金流保持不变，现值在 $r$ 约为 1 时会急剧增加。

```{code-cell} ipython3
print('dp0 / dg is:')
dp_dg = sym.diff(p0, g)
dp_dg
```

```{code-cell} ipython3
print('dp0 / dr is:')
dp_dr = sym.diff(p0, r)
dp_dr
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "复合绘图显示后期显著偏差"
    name: 复合绘图
---
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
rs = np.arange(0.001, 0.10, .001)
gs = np.arange(0.001, 0.99, .001)
rs, gs = np.meshgrid(rs, gs)
_Z = p_0_prior(rs, gs, $1) 
ax.plot_surface(rs, gs, _Z, cmap=cm.coolwarm)
ax.set_xlabel('$g$')
ax.set_ylabel('$r$')
ax.set_zlabel('$p_0$')
plt.show()
```

上面的复合绘图显示了在不同 $r$ 和 $g$ 值下，50 年期租赁现值的显著变化。

更高的 $r$ 值和更低的 $g$ 值导致租赁的现值较低，而当 $g$ 接近或高于 $r$ 时，租赁的现值显著增加。