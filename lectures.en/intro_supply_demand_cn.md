---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# 供需介绍

## 概述

本讲座讨论了一些均衡价格和数量模型，这是初级微观经济学的核心课题之一。

在整个讲座中，我们专注于单一商品和单一价格的模型。

```{seealso}
在{doc}`后续讲座 <supply_demand_multiple_goods>`中，我们将研究多商品设置。
```

### 为什么这个模型很重要？

在15世纪、16世纪、17世纪和18世纪，重商主义思想在大多数欧洲国家的统治者中占据了主导地位。

出口被认为是好的，因为它们带来了金银（黄金流入国家）。

进口被认为是坏的，因为需要支付金银（黄金流出）。

这种[零和](https://en.wikipedia.org/wiki/Zero-sum_game)的经济观最终被古典经济学家如[亚当·斯密](https://en.wikipedia.org/wiki/Adam_Smith)和[大卫·李嘉图](https://en.wikipedia.org/wiki/David_Ricardo)的工作推翻，他们展示了如何通过自由化国内和国际贸易来提高福利。

经济学中有很多不同的方式来表达这一思想。

本讲座讨论了最简单的方式之一：如何通过自由价格调整来最大化单一商品市场中的社会福利测量。

### 主题和基础设施

在本讲座中，我们将遇到的关键基础设施概念有：

* 逆需求曲线
* 逆供给曲线
* 消费者剩余
* 生产者剩余
* 积分
* 作为消费者和生产者剩余之和的社会福利
* 均衡数量与社会福利最优关系

在我们的讲解中，我们将使用以下Python导入。

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
```
图像输入功能：启用

## 消费者剩余

在研究供需模型之前，我们将了解一些关于（a）消费者和生产者剩余以及（b）积分的背景知识。

（如果您对这两个主题都很熟悉，您可以直接跳到{ref}`下一节 <integration>`。）

### 一个离散例子

关于消费者剩余，假设我们有一种商品和10个消费者。

这10个消费者有不同的偏好；特别是，他们愿意支付一单位商品的金额不同。

假设10个消费者每人的支付意愿如下：

| 消费者         | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10  |
|----------------|----|----|----|----|----|----|----|----|----|-----|
| 支付意愿 | 98 | 72 | 41 | 38 | 29 | 21 | 17 | 12 | 11 | 10  |

（我们按支付意愿的降序排列消费者。）

如果 $p$ 是商品的价格，$w_i$ 是消费者 $i$ 的支付意愿，那么当 $w_i \geq p$ 时，$i$ 就会购买。

```{note}
如果 $p=w_i$，消费者对购买与不购买无所谓；我们随意假设他们会购买。
```

第 $i$ 个消费者的**消费者剩余**是 $\max\{w_i - p, 0\}$

* 如果 $w_i \geq p$，则消费者购买并获得剩余 $w_i - p$
* 如果 $w_i < p$，则消费者不购买并获得剩余 $0$

例如，如果价格是 $p=40$，则消费者1获得剩余 $98-40=58$。

下方的柱状图显示了当 $p=25$ 时每个消费者的剩余。

每个柱状图的总高是消费者 $i$ 的支付意愿。

一些柱子中的橙色部分显示的是消费者剩余。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "支付意愿（离散）"
    name: wpdisc
---
fig, ax = plt.subplots()
consumers = range(1, 11) # 消费者 1,..., 10
# 每个消费者的支付意愿
wtp = (98, 72, 41, 38, 29, 21, 17, 12, 11, 10)
price = 25
ax.bar(consumers, wtp, label="消费者剩余", color="darkorange", alpha=0.8)
ax.plot((0, 12), (price, price), lw=2, label="价格 $p$")
ax.bar(consumers, [min(w, price) for w in wtp], color="black", alpha=0.6)
ax.set_xlim(0, 12)
ax.set_xticks(consumers)
ax.set_ylabel("支付意愿, 价格")
ax.set_xlabel("消费者, 数量")
ax.legend()
plt.show()
```

总消费者剩余是

$$ 
\sum_{i=1}^{10} \max\{w_i - p, 0\}
= \sum_{w_i \geq p} (w_i - p)
$$

由于消费者剩余 $\max\{w_i-p,0\}$ 是消费者 $i$ 的交易收益的度量（即，商品价值超过消费者支付金额的程度），将总消费者剩余视为消费者福利的度量是合理的。

之后我们将进一步探讨这个想法，考虑不同的价格如何导致消费者和生产者获得不同的福利结果。

### 关于数量的评论

请注意，在图中，横轴标注为“消费者，数量”。

我们在这里添加了“数量”，因为我们可以从这个轴上读取出售的单位数，假设目前有卖家愿意以当前市场价格 $p$ 向消费者提供任意数量的商品。

在这个例子中，消费者1到5购买，售出的数量是5。

下面我们取消卖家会在任意价格提供任意数量的假设，并研究这将如何改变结果。

### 连续近似

通常假设有“非常大量”的消费者是很方便的，这样支付意愿就成为一条连续的曲线。

和之前一样，纵轴表示支付意愿，而横轴表示数量。

这种曲线被称为**逆需求曲线**。

以下提供了一个例子，展示了逆需求曲线和设定价格。

逆需求曲线由以下公式给出

$$
p = 100 e^{-q} 
$$

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "支付意愿（连续）"
    name: wpcont
---
def inverse_demand(q):
    return 100 * np.exp(- q)

# 构建网格以在不同的 q 值处评估函数
q_min, q_max = 0, 5
q_grid = np.linspace(q_min, q_max, 1000)

# 绘制逆需求曲线
fig, ax = plt.subplots()
ax.plot((q_min, q_max), (price, price), lw=2, label="价格")
ax.plot(q_grid, inverse_demand(q_grid), 
        color="orange", label="逆需求曲线")
ax.set_ylabel("支付意愿, 价格")
ax.set_xlabel("数量")
ax.set_xlim(q_min, q_max)
ax.set_ylim(0, 110)
ax.legend()
plt.show()
```

通过类比离散情况，需求曲线下方和价格上方的面积称为**消费者剩余**，即消费者总贸易收益的度量。

消费者剩余在下图中用阴影表示。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "支付意愿（连续）与消费者剩余"
    name: wpcont_cs
---
# solve for the value of q where demand meets price
q_star = np.log(100) - np.log(price)

fig, ax = plt.subplots()
ax.plot((q_min, q_max), (price, price), lw=2, label="价格")
ax.plot(q_grid, inverse_demand(q_grid), 
        color="orange", label="逆需求曲线")
small_grid = np.linspace(0, q_star, 500)
ax.fill_between(small_grid, np.full(len(small_grid), price),
                inverse_demand(small_grid), color="orange",
                alpha=0.5, label="消费者剩余")
ax.vlines(q_star, 0, price, ls="--")
ax.set_ylabel("支付意愿, 价格")
ax.set_xlabel("数量")
ax.set_xlim(q_min, q_max)
ax.set_ylim(0, 110)
ax.text(q_star, -10, "$q^*$")
ax.legend()
plt.show()
```

这样计算出来的消费者剩余表示为

$$
\int_0^{q^*} (p(q) - p) \, dq
$$

其中 $q^*$ 是逆需求曲线与价格相交的地方。

到目前为止，对于我们定义消费者剩余的方式，您可能会有以下两个问题：

1. 消费者剩余度量的合理性。

2. 如何在离散情况下使用积分计算消费者剩余。

关于问题1，很容易证明在一定条件下，消费者剩余代表了增加到消费者总收入中的价值，假设每个消费者从市场中消费数量$q^*$的商品。

关于问题2，注意到当消费者数量足够大时，单个消费者的支付意愿按大小顺序非常接近。

相应地，通过将支付意愿与消费者数量的累积均值匹配和插值，我们可以逼近离散支付意愿分布为一条连续需求曲线。

如图连续情况下的消费者剩余是由同样价格设定下所有消费者剩余的数值和表示的。

### 数值求解

事实上，如果需求曲线可以用解析项表示，就可以用数值积分的方式来计算消费剩余。

```{code-cell} ipython3
import scipy.integrate as spi

# 计算消费者剩余
integral, error = spi.quad(lambda q: inverse_demand(q) - price, 0., q_star)
integral
```

数值结果看起来合理。

然而，如果想要更加准确的结果，该如何处理呢？

我们可以细分我们的网格区域，在每个子区域内进行维积分，并求和得到总消费者剩余。

下面提供了一个细分的工作示例。

```{code-cell} ipython3
def finer_partition_integrate(f, a, b, num_intervals):
    x = np.linspace(a, b, num_intervals+1)
    h = x[1] - x[0]
    integral = sum(f(x[i]) * h for i in range(num_intervals))
    return integral

# 计算消费者剩余
finer_partition_integrate(lambda q: inverse_demand(q) - price, 0.0, q_star, 10000)
```

这样得出的数值结果精度会更高，可以用户可控的粒度以保证可接受的累计误差比例。

## 生产者剩余

我们现在考虑生产者剩余，讨论与商家或供应商的支付意愿相关的概念。

设 $v_i$ 是生产者 $i$ 愿意出售商品的价格。

当价格为 $p$ 时，生产者 $i$ 的生产者剩余是 $\max\{p - v_i, 0\}$。

例如，一个愿意以 \$10 售卖的生产者，以 \$20 的价格售出，获得了 \$10 的剩余。

总生产者剩余为

$$
\sum_{i=1}^{10} \max\{p - v_i, 0\}
= \sum_{p \geq v_i} (p - v_i)
$$

与消费者情况类似，如果我们将生产者愿意出售的价格近似成一条连续曲线来进行分析会更有帮助。

这条曲线被称为**逆供给曲线**

我们展示一个例子，其中逆供给曲线为

$$
p = 2 q^2
$$

阴影部分是这个连续模型中的总生产者剩余。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "愿意出售的价格（连续）与生产者剩余"
    name: wscont
---
def inverse_supply(q):
    return 2 * q**2

# solve for the value of q where supply meets price
q_star = (price / 2)**(1/2)

# plot the inverse supply curve
fig, ax = plt.subplots()
ax.plot((q_min, q_max), (price, price), lw=2, label="价格")
ax.plot(q_grid, inverse_supply(q_grid), 
        color="green", label="逆供给曲线")
small_grid = np.linspace(0, q_star, 500)
ax.fill_between(small_grid, inverse_supply(small_grid), 
                np.full(len(small_grid), price), 
                color="green",
                alpha=0.5, label="生产者剩余")
ax.vlines(q_star, 0, price, ls="--")
ax.set_ylabel("愿意出售的价格, 价格")
ax.set_xlabel("数量")
ax.set_xlim(q_min, q_max)
ax.set_ylim(0, 60)
ax.text(q_star, -10, "$q^*$")
ax.legend()
plt.show()
```

生产者剩余可以用以下公式表示

$$
\int_0^{q^*} (p - \sigma(q)) \, dq
$$

其中 $ \sigma $ 是逆供给曲线，并且 $ q^* $ 是供给与价格相交的数量。

通过专门的数值积分工具，我们可以计算出该区域的面积。例如：

```{code-cell} ipython3
# 计算生产者剩余
integral, error = spi.quad(lambda q: price - inverse_supply(q), 0., q_star)
integral
```

再次，我们可以用以下示例通过网格细分改进这个结果

```{code-cell} ipython3
# 计算生产者剩余
finer_partition_integrate(lambda q: price - inverse_supply(q), 0.0, q_star, 10000)
```

## 总福利

我们现在回到逆供给和需求曲线都存在的情况。

除非出现特殊情况，逆供给曲线和逆需求曲线一般会在某个价格水平 $ p^* $ 交汇。

交汇点以 $ (q^*, p^*) $ 表示。

这里 $ q^* $ 表示最优价格水平 $ p^* $ 时的平衡数量。

例如，考虑以下供需曲线

* 需求
$$
p_d(q) = 100 e^{-q}
$$ 
* 供给
$$
p_s(q) = 2 q^2
$$

在这种情况下，平衡数量和价格如图所示

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(q_grid, inverse_demand(q_grid), 
        color="orange", label="逆需求曲线")
ax.plot(q_grid, inverse_supply(q_grid), 
        color="green", label="逆供给曲线")

# 目前我们是假设之前已计算出均衡
ax.plot((q_star, q_star), (0, price), "k--")
ax.plot((0, q_max), (price, price), "k--")
ax.legend()
ax.set_xlim(0, q_max)
ax.set_ylim(0, 110)
ax.set_ylabel("支付意愿, 价格")
ax.text(q_star, -10, "$q^*$")
ax.text(-0.2, price, "$p^*$")
plt.show()
```

生产者和消费者剩余用阴影部分表示。

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(q_grid, inverse_demand(q_grid), 
        color="orange", label="逆需求曲线")
ax.plot(q_grid, inverse_supply(q_grid), 
        color="green", label="逆供给曲线")
q_star = np.log(100) / 2
small_grid = np.linspace(0, q_star, 500)

ax.fill_between(small_grid, np.full(len(small_grid), price), 
                inverse_demand(small_grid), color="orange",
                alpha=0.5, label="消费者剩余")
ax.fill_between(small_grid, inverse_supply(small_grid), 
                np.full(len(small_grid), price), 
                color="green",
                alpha=0.5, label="生产者剩余")
ax.plot((q_star,q_star), (0, price), 'k--')
ax.plot((0, q_max), (price, price), 'k--')
ax.legend()
ax.set_xlim(0, q_max)
ax.set_ylim(0, 110)
ax.set_ylabel("支付意愿, 价格")
ax.text(q_star, -10, "$q^*$")
ax.text(-0.2, price, "$p^*$")
plt.show()
```

将社会总福利定义为消费者和生产者剩余之和

$$
\int_0^{q^*} p_d(q) dq - \int_0^{q^*} p_s(q) dq
$$

```{code-cell} ipython3
# 计算消费者剩余
consumer_surplus, error = spi.quad(lambda q: inverse_demand(q), 0., q_star)
# 从总和中减去价格乘以最优数量
consumer_surplus -= price * q_star
consumer_surplus
```

```{code-cell} ipython3
# 计算生产者剩余
producer_surplus, error = spi.quad(lambda q: price, 0., q_star)
# 减去逆供给
producer_surplus -= spi.quad(lambda q: inverse_supply(q), 0., q_star)[0]
producer_surplus
```

```{code-cell} ipython3
# 计算总福利
consumer_surplus + producer_surplus
```

## 积分规则

有许多关于计算积分的规则，不同的规则适用于不同的函数 $f$。

许多这些规则都与数学中最美丽和最强大的结果之一有关：[微积分基本定理](https://en.wikipedia.org/wiki/Fundamental_theorem_of_calculus)。

我们不会试图在这里讨论这些概念，部分原因是这个主题太大，部分原因是你在本讲座中只需要知道一个规则，如下所述。

如果 $f(x) = c + \mathrm{d} x$，则

$$ 
\int_a^b f(x) \mathrm{d} x = c (b - a) + \frac{d}{2}(b^2 - a^2) 
$$

实际上，这个规则是如此简单，可以通过基本几何来计算——你可以尝试通过绘制 $f$ 并计算 $a$ 和 $b$ 之间曲线下的面积来验证。

接下来内容中我们会反复使用此规则。

## 供需

现在我们将供给和需求结合起来。

这引导我们到重要的市场均衡概念，然后讨论均衡和福利。

在大部分讨论中，我们假设逆需求和供给曲线是**仿射**函数的数量。

```{note}
"仿射" 意味着 "线性加常数"，[这里](https://math.stackexchange.com/questions/275310/what-is-the-difference-between-linear-and-affine-function)是一个关于它很好的讨论。
```

当我们在后续讲座中研究多种消费品的模型时，我们也会假设仿射的逆供给和需求函数。

这样做是为了简化解释，使我们能够使用线性代数中的少量工具，即矩阵乘法和矩阵求逆。

我们研究的是买卖双方以某个价格 $p$ 交换某种商品数量 $q$ 的市场。

数量 $q$ 和价格 $p$ 都是标量。

我们假设该商品的逆需求和供给曲线为：

$$
p = d_0 - d_1 q, \quad d_0, d_1 > 0
$$

$$
p = s_0 + s_1 q , \quad s_0, s_1 > 0
$$

我们称它们为逆需求和供给曲线，因为价格在等式的左侧，而不是像在直接需求或供给函数中那样在右侧。

我们可以使用一个 [namedtuple](https://docs.python.org/3/library/collections.html#collections.namedtuple) 来存储单一商品市场的参数。

```{code-cell} ipython3
Market = namedtuple('Market', ['d_0', # 需求截距
                               'd_1', # 需求斜率
                               's_0', # 供给截距
                               's_1'] # 供给斜率
                   )
```

下面的函数创建一个带有默认值的 Market namedtuple 实例。

```{code-cell} ipython3
def create_market(d_0=1.0, d_1=0.6, s_0=0.1, s_1=0.4):
    return Market(d_0=d_0, d_1=d_1, s_0=s_0, s_1=s_1)
```

## 图形和均衡

考虑以下函数，它绘制市场的供需曲线。

```{code-cell} ipython3
def plot_market(m):
    q_max = (m.d_0 - m.s_0) / min(m.d_1, m.s_1)
    q_grid = np.linspace(0, q_max, 100)
    fig, ax = plt.subplots()
    ax.plot(q_grid, m.d_0 - m.d_1 * q_grid, lw=2, alpha=0.6, label="逆需求曲线")
    ax.plot(q_grid, m.s_0 + m.s_1 * q_grid, lw=2, alpha=0.6, label="逆供给曲线")
    ax.legend()
    ax.set_xlim(0, q_max)
    plt.show()

m = create_market()
plot_market(m)
```

### 均衡分析

我们现在想要知道均衡价格和数量。

这意味着我们想要知道两条曲线相交的位置。

为了回答这个问题，我们将两条曲线相等

$$
d_0 - d_1 q = s_0 + s_1 q
$$

通过一些代数变换，我们得到

$$
q^* = (d_0 - s_0) / (s_1 + d_1)
$$

然后我们可以用逆供给或需求曲线计算得到对应的 $p$

$$
p^* = d_0 - d_1 q^*
$$

我们可以计算出一个市场的均衡价格和数量如下

```{code-cell} ipython3
def compute_equilibrium(m):
    q_star = (m.d_0 - m.s_0) / (m.d_1 + m.s_1)
    p_star = m.d_0 - m.d_1 * q_star
    return p_star, q_star

price, quantity = compute_equilibrium(m)
(price, quantity)
```

一个简单的函数可以将需求曲线、供给曲线和范围进行绘制

```{code-cell} ipython3
def plot_market_with_equilibrium(m):
    price, quantity = compute_equilibrium(m)
    q_max = (m.d_0 - m.s_0) / min(m.d_1, m.s_1)
    q_grid = np.linspace(0, q_max, 100)
    fig, ax = plt.subplots()
    ax.plot(q_grid, m.d_0 - m.d_1 * q_grid, lw=2, alpha=0.6, label="逆需求曲线")
    ax.plot(q_grid, m.s_0 + m.s_1 * q_grid, lw=2, alpha=0.6, label="逆供给曲线")
    ax.plot((quantity, quantity), (0, price), 'k--')
    ax.plot((0, quantity), (price, price), 'k--')
    ax.legend()
    ax.set_xlim(0, q_max)
    plt.show()
plot_market_with_equilibrium(m)
```

## 福利

消费者剩余，如阴影区域所示

$$
\int_0^{q^*} (d_0 - d_1 q - p^*) d q 
$$

生产者剩余，如阴影区域所示

$$
\int_0^{q^*} (p^* - (s_0 + s_1 q)) d q 
$$

社会总福利是消费者和生产者剩余之和

```{code-cell} ipython3
def compute_welfare(m):
    p_star, q_star = compute_equilibrium(m)
    d = m.d_0 - m.d_1 * np.linspace(0, q_star, 1000)
    s = m.s_0 + m.s_1 * np.linspace(0, q_star, 1000)
    consumer_surplus = np.trapz(d - p_star, dx=q_star/1000)
    producer_surplus = np.trapz(p_star - s, dx=q_star/1000)
    total_welfare = consumer_surplus + producer_surplus
    return consumer_surplus, producer_surplus, total_welfare

consumer_surplus, producer_surplus, total_welfare = compute_welfare(m)
(consumer_surplus, producer_surplus, total_welfare)
```

### 小结

综上所述，我们探讨了如何计算均衡价格和数量，并研究了消费者剩余和生产者剩余。

我们还展示了如何用数值方法计算并可视化这些概念。

我们现在可以总结一下上述步骤，并定义一些可以直接调用的辅助函数：

1. 创建一个 `Market` 实例
2. 计算均衡价格和数量
3. 绘制市场供需曲线及其均衡
4. 计算社会总福利

以下是完整的示例代码：

```{code-cell} ipython3
# 定义 Market 类以及创建市场实例的函数
Market = namedtuple('Market', ['d_0', 'd_1', 's_0', 's_1'])

def create_market(d_0=1.0, d_1=0.6, s_0=0.1, s_1=0.4):
    return Market(d_0=d_0, d_1=d_1, s_0=s_0, s_1=s_1)

# 定义绘制市场供需曲线的函数
def plot_market(m):
    q_max = (m.d_0 - m.s_0) / min(m.d_1, m.s_1)
    q_grid = np.linspace(0, q_max, 100)
    fig, ax = plt.subplots()
    ax.plot(q_grid, m.d_0 - m.d_1 * q_grid, lw=2, alpha=0.6, label="逆需求曲线")
    ax.plot(q_grid, m.s_0 + m.s_1 * q_grid, lw=2, alpha=0.6, label="逆供给曲线")
    ax.legend()
    ax.set_xlim(0, q_max)
    plt.show()

# 定义计算均衡价格和数量的函数
def compute_equilibrium(m):
    q_star = (m.d_0 - m.s_0) / (m.d_1 + m.s_1)
    p_star = m.d_0 - m.d_1 * q_star
    return p_star, q_star

# 定义绘制带有均衡点的市场供需曲线的函数
def plot_market_with_equilibrium(m):
    price, quantity = compute_equilibrium(m)
    q_max = (m.d_0 - m.s_0) / min(m.d_1, m.s_1)
    q_grid = np.linspace(0, q_max, 100)
    fig, ax = plt.subplots()
    ax.plot(q_grid, m.d_0 - m.d_1 * q_grid, lw=2, alpha=0.6, label="逆需求曲线")
    ax.plot(q_grid, m.s_0 + m.s_1 * q_grid, lw=2, alpha=0.6, label="逆供给曲线")
    ax.plot((quantity, quantity), (0, price), 'k--')
    ax.plot((0, quantity), (price, price), 'k--')
    ax.legend()
    ax.set_xlim(0, q_max)
    plt.show()

# 定义计算消费者剩余、生产者剩余和总福利的函数
def compute_welfare(m):
    p_star, q_star = compute_equilibrium(m)
    d = m.d_0 - m.d_1 * np.linspace(0, q_star, 1000)
    s = m.s_0 + m.s_1 * np.linspace(0, q_star, 1000)
    consumer_surplus = np.trapz(d - p_star, dx=q_star/1000)
    producer_surplus = np.trapz(p_star - s, dx=q_star/1000)
    total_welfare = consumer_surplus + producer_surplus
    return consumer_surplus, producer_surplus, total_welfare

# 示例使用
m = create_market()
plot_market_with_equilibrium(m)
consumer_surplus, producer_surplus, total_welfare = compute_welfare(m)
(consumer_surplus, producer_surplus, total_welfare)
```

此 `market` 可以被我们的 `inverse_demand` 和 `inverse_supply` 函数所使用。

```{code-cell} ipython3
def inverse_demand(q, model):
    # 计算逆需求
    return model.d_0 - model.d_1 * q

def inverse_supply(q, model):
    # 计算逆供给
    return model.s_0 + model.s_1 * q
```

上述代码块定义了逆需求和逆供给函数，它们可以使用`Market`实例进行计算。

下面是使用 `market` 变量绘制供需曲线的代码示例：

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Supply and demand"
    name: supply_demand
---
market = create_market()

grid_min, grid_max, grid_size = 0, 1.5, 200
q_grid = np.linspace(grid_min, grid_max, grid_size)
supply_curve = inverse_supply(q_grid, market)
demand_curve = inverse_demand(q_grid, market)

fig, ax = plt.subplots()
ax.plot(q_grid, supply_curve, label='供给', color='green')
ax.plot(q_grid, demand_curve, label='需求', color='orange')
ax.legend(loc='upper center', frameon=False)
ax.set_ylim(0, 1.2)
ax.set_xticks((0, 1))
ax.set_yticks((0, 1))
ax.set_xlabel('数量')
ax.set_ylabel('价格')
plt.show()
```

这段代码首先创建一个市场实例，然后生成一个数量网格，计算相对应的供需曲线，并绘制结果。

在上面的图中，**均衡**价格-数量对出现在供给和需求曲线的交点处。

### 消费者剩余

设给定某个数量 $q$ 并设 $p := d_0 - d_1 q$ 为逆需求曲线上相应的价格。

我们将**消费者剩余** $S_c(q)$ 定义为逆需求曲线下的面积减去 $p q$：

$$
S_c(q) := 
\int_0^{q} (d_0 - d_1 x) \mathrm{d} x - p q 
$$ (eq:cstm_spls)

下图展示了这一点

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "供给和需求（消费者剩余）"
    name: supply_demand_cs
tags: [hide-input]
---

q = 1.25
p = inverse_demand(q, market)
ps = np.ones_like(q_grid) * p

fig, ax = plt.subplots()
ax.plot(q_grid, demand_curve, label='需求', color='orange')
ax.fill_between(q_grid[q_grid <= q],
                demand_curve[q_grid <= q],
                ps[q_grid <= q],
                label='消费者剩余',
                color="orange", 
                alpha=0.5)
ax.vlines(q, 0, p, linestyle="dashed", color='black', alpha=0.7)
ax.hlines(p, 0, q, linestyle="dashed", color='black', alpha=0.7)

ax.legend(loc='upper center', frameon=False)
ax.set_ylim(0, 1.2)
ax.set_xticks((q,))
ax.set_xticklabels(("$q$",))
ax.set_yticks((p,))
ax.set_yticklabels(("$p$",))
ax.set_xlabel('数量')
ax.set_ylabel('价格')
plt.show()
```

这个绘图展示了消费者剩余的阴影区域，它表示消费者在给定数量和价格下的福利增益。

消费者剩余提供了数量 $q$ 对应的总消费者福利的度量。

其思想是逆需求曲线 $d_0 - d_1 q$ 显示了消费者在给定数量 $q$ 下，愿意为额外增量支付的价格。

愿意支付的价格与实际支付的价格之间的差额就是消费者剩余。

值 $S_c(q)$ 是当购买总量为 $q$，购买价格为 $p$ 时，这些剩余的“总和”（即积分）。

在消费者剩余定义 {eq}`eq:cstm_spls` 中评估积分可得

$$
S_c(q) 
= d_0 q - \frac{1}{2} d_1 q^2 - p q
$$

### 生产者剩余

设给定某个数量 $q$ 并设 $p := s_0 + s_1 q$ 为逆供给曲线上相应的价格。

我们将**生产者剩余**定义为 $p q$ 减去逆供给曲线下的面积

$$
S_p(q) 
:= p q - \int_0^{q} (s_0 + s_1 x) \mathrm{d} x 
$$ (eq:pdcr_spls)

下图展示了这一点

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "供给和需求（生产者剩余）"
    name: supply_demand_ps
tags: [hide-input]
---

q = 0.75
p = inverse_supply(q, market)
ps = np.ones_like(q_grid) * p

fig, ax = plt.subplots()
ax.plot(q_grid, supply_curve, label='供给', color='green')
ax.fill_between(q_grid[q_grid <= q],
                supply_curve[q_grid <= q],
                ps[q_grid <= q],
                label='生产者剩余',
                color="green",
                alpha=0.5)
ax.vlines(q, 0, p, linestyle="dashed", color='black', alpha=0.7)
ax.hlines(p, 0, q, linestyle="dashed", color='black', alpha=0.7)

ax.legend(loc='upper center', frameon=False)
ax.set_ylim(0, 1.2)
ax.set_xticks((q,))
ax.set_xticklabels(("$q$",))
ax.set_yticks((p,))
ax.set_yticklabels(("$p$",))
ax.set_xlabel('数量')
ax.set_ylabel('价格')
plt.show()
```

生产者剩余提供了数量 $q$ 对应的总生产者福利的度量。

其思想是逆供给曲线 $s_0 + s_1 q$ 显示了生产者在给定数量 $q$ 下，愿意出售的价格。

愿意出售的价格与实际收到的价格之间的差额就是生产者剩余。

值 $S_p(q)$ 是这些剩余的“总和”（即积分）。

在生产者剩余定义 {eq}`eq:pdcr_spls` 中评估积分可得

$$
S_p(q) = p q - s_0 q -  \frac{1}{2} s_1 q^2
$$

### 社会福利

有时经济学家通过一个**福利标准**来衡量社会福利，该标准等于消费者剩余与生产者剩余之和，假设消费者和生产者支付相同的价格：

$$
W(q)
= \int_0^q (d_0 - d_1 x) dx - \int_0^q (s_0 + s_1 x) \mathrm{d} x  
$$

评估积分可得

$$
W(q) = (d_0 - s_0) q -  \frac{1}{2} (d_1 + s_1) q^2
$$

以下是一个 Python 函数，它评估给定数量 $q$ 和一组固定参数下的社会福利。

```{code-cell} ipython3
def W(q, market):
    # Compute and return welfare
    return (market.d_0 - market.s_0) * q - 0.5 * (market.d_1 + market.s_1) * q**2
```

这个函数计算给定市场条件下的社会总福利。

下图绘制了社会福利 $W$ 作为数量 $q$ 的函数。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Welfare"
    name: wf
tags: [hide-input]
---

q_vals = np.linspace(0, 1.78, 200)
fig, ax = plt.subplots()
ax.plot(q_vals, W(q_vals, market), label='welfare', color='brown')
ax.legend(frameon=False)
ax.set_xlabel('quantity')
plt.show()
```

让我们假设现在有一个社会规划者的任务是最大化社会福利。

要计算最大化福利准则的数量，我们对 $q$ 求导数并将导数设置为零。

$$
\frac{\mathrm{d} W(q)}{\mathrm{d} q} = d_0 - s_0 - (d_1 + s_1) q  = 0
$$

求解 $q$ 得到：

$$
q = \frac{ d_0 - s_0}{s_1 + d_1}
$$ (eq:old1)

记住方程 {eq}`eq:old1` 给出的数量 $q$，这是社会规划者为了最大化消费者剩余加生产者剩余而选择的。

我们将它与竞争性均衡中通过使供给等于需求所产生的数量进行比较。

### 竞争性均衡

我们可以通过使需求价格等于供给价格来实现相同的效果，而不是使供应和需求数量相等：

$$
p =  d_0 - d_1 q = s_0 + s_1 q 
$$

如果我们求解上述等式中的第二等式来求 $q$，我们得到：

$$
q = \frac{ d_0 - s_0}{s_1 + d_1}
$$ (eq:equilib_q)

这是竞争性均衡的数量。

注意，均衡数量等于方程 {eq}`eq:old1` 给出的相同 $q$。

方程 {eq}`eq:old1` 所确定的数量使供给等于需求的结果带来了一个关键发现：

* 竞争性均衡数量最大化了我们的福利准则

这是[第一福利定理](https://en.wikipedia.org/wiki/Fundamental_theorems_of_welfare_economics) 的一个版本，

它还带来了一个有用的竞争性均衡计算策略：

* 在求解福利问题以获得最优数量后，我们可以从供给价格或需求价格的任一处读出竞争性均衡价格，在竞争性均衡数量处。

## 推广

在{doc}`后续讲座 <supply_demand_multiple_goods>`中，我们将从其他对象中导出上述需求曲线和供给曲线的推广。

我们的推广将把对单一商品市场的分析扩展到对 $n$ 个商品的 $n$ 个同时市场的分析。

另外，

* 我们将从一个在预算约束下最大化**效用函数**的消费者问题中导出**需求曲线**。

* 我们将从一个价格接受者的生产者最大化其利润减去由**成本函数**描述的总成本的问题中导出**供给曲线**。

## 练习

假设现在逆需求和供给曲线被修改为

$$
p = i_d(q) := d_0 - d_1 q^{0.6} 
$$

$$
p = i_s(q) := s_0 + s_1 q^{1.8} 
$$

所有参数依然是正值，如之前所述。

```{exercise}
:label: isd_ex1

使用之前参数值保存的同一个 `Market` namedtuple，但使新的 `inverse_demand` 和 `inverse_supply` 函数匹配这些新定义。

然后绘制逆需求和供给曲线 $i_d$ 和 $i_s$。

```

```{solution-start} isd_ex1
:class: dropdown
```

让我们更新上述定义的 `inverse_demand` 和 `inverse_supply` 函数。

```{code-cell} ipython3
def inverse_demand(q, model):
    return model.d_0 - model.d_1 * q**0.6

def inverse_supply(q, model):
    return model.s_0 + model.s_1 * q**1.8
```

接下来，我们将绘制供需曲线。我们将重新绘制图表，以便适应新的需求和供给函数形式。

```{code-cell} ipython3
market = create_market()

grid_min, grid_max, grid_size = 0, 1.5, 200
q_grid = np.linspace(grid_min, grid_max, grid_size)
supply_curve = inverse_supply(q_grid, market)
demand_curve = inverse_demand(q_grid, market)

fig, ax = plt.subplots()
ax.plot(q_grid, supply_curve, label='供给', color='green')
ax.plot(q_grid, demand_curve, label='需求', color='orange')
ax.legend(loc='upper center', frameon=False)
ax.set_ylim(0, 1.2)
ax.set_xticks((0, 1))
ax.set_yticks((0, 1))
ax.set_xlabel('数量')
ax.set_ylabel('价格')
plt.show()
```

如上图所示，我们按照新的定义更新了供需曲线。
```{solution-end}
```


```{exercise}
:label: isd_ex2

如前所述，消费剩余在数量 $q$ 时是需求曲线下的面积减去价格乘以数量：

$$
S_c(q) = \int_0^{q} i_d(x) \mathrm{d} x - p q 
$$

此处 $p$ 设置为 $i_d(q)$

生产者剩余是价格乘以数量减去逆供给曲线下的面积：

$$
S_p(q) 
= p q - \int_0^q i_s(x) \mathrm{d} x 
$$

此处 $p$ 设置为 $i_s(q)$。

社会福利是假设买卖双方价格相同的情况下，消费者和生产者剩余的总和：

$$
W(q)
= \int_0^q i_d(x) \mathrm{d} x - \int_0^q i_s(x) \mathrm{d} x  
$$

求积分并编写一个函数以数值方式计算给定 $q$ 的该量。

绘制福利作为 $q$ 的函数。
```

```{solution-start} isd_ex2
:class: dropdown
```

计算积分得

$$
W(q) 
= d_0 q - \frac{d_1 q^{1.6}}{1.6}
    - \left( s_0 q + \frac{s_1 q^{2.8}}{2.8} \right)
$$

以下是一个计算此值的 Python 函数：

```{code-cell} ipython3
def W(q, market):
    # Compute and return welfare
    S_c = market.d_0 * q - market.d_1 * q**1.6 / 1.6
    S_p = market.s_0 * q + market.s_1 * q**2.8 / 2.8
    return S_c - S_p
```

接下来，我们绘制福利 $W$ 作为数量 $q$ 的函数。

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(q_vals, W(q_vals, market), label='welfare', color='brown')
ax.legend(frameon=False)
ax.set_xlabel('quantity')
plt.show()
```

```{solution-end}
```

````{exercise}
:label: isd_ex3

由于非线性，新的福利函数不易用铅笔和纸来最大化。

请使用 `scipy.optimize.minimize_scalar` 来最大化它。

```{seealso}
我们的 [SciPy](https://python-programming.quantecon.org/scipy.html) 讲座中关于 [优化](https://python-programming.quantecon.org/scipy.html#optimization) 的一节是了解更多的有用资源。
```

````

```{solution-start} isd_ex3
:class: dropdown
```

```{code-cell} ipython3
from scipy.optimize import minimize_scalar

def objective(q):
    return -W(q, market)

result = minimize_scalar(objective, bounds=(0, 10))
print(result.message)
```

```{code-cell} ipython3
maximizing_q = result.x
print(f"{maximizing_q: .5f}")
```

```{solution-end}
```

````{exercise}
:label: isd_ex4

现在通过找到等于供给和需求的价格来计算均衡数量。

你可以通过找到超额需求函数的根来数值计算：

$$
e_d(q) := i_d(q) - i_s(q) 
$$

你可以用 `scipy.optimize.newton` 来计算这个根。

```{seealso}
我们的 [SciPy](https://python-programming.quantecon.org/scipy.html) 讲座中关于 [根和固定点](https://python-programming.quantecon.org/scipy.html#roots-and-fixed-points) 的一节是了解更多的有用资源。
```

用接近1.0的初始猜测值来初始化 `newton`。

（类似的初始条件会给出相同的结果。）

你会发现均衡价格与福利最大化价格是一致的，符合首要福利理论。
````

```{solution-start} isd_ex4
:class: dropdown
```

```{code-cell} ipython3
from scipy.optimize import newton

def excess_demand(q):
    return inverse_demand(q, market) - inverse_supply(q, market)

equilibrium_q = newton(excess_demand, 0.99)
print(f"{equilibrium_q: .5f}")
```

```{solution-end}
```