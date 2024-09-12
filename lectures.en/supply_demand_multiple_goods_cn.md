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

(supply_demand_multiple_goods)=
# 多种商品的供需

## 概述

在{doc}`<intro_supply_demand>`的前一篇讲座中，我们研究了一种单一消费品市场的供给、需求和福利。

在本讲座中，我们研究了一个有 $n$ 种商品和 $n$ 个相应价格的环境。

我们在本讲座中会遇到的关键基础概念包括

* 逆需求曲线
* 财富的边际效用
* 逆供给曲线
* 消费者剩余
* 生产者剩余
* 作为消费者和生产者剩余之和的社会福利
* 竞争均衡

我们将提供一个由以下经济学家们制定的[第一福利理论](https://en.wikipedia.org/wiki/Fundamental_theorems_of_welfare_economics)的版本：

* [列昂·瓦尔拉斯](https://en.wikipedia.org/wiki/L%C3%A9on_Walras)
* [弗朗西斯·伊西德罗·艾奇沃斯](https://en.wikipedia.org/wiki/Francis_Ysidro_Edgeworth)
* [维尔弗雷多·帕累托](https://en.wikipedia.org/wiki/Vilfredo_Pareto)

这些关键思想的重要扩展由以下经济学家获得：

* [阿巴·勒纳](https://en.wikipedia.org/wiki/Abba_P._Lerner)
* [哈罗德·霍特林](https://en.wikipedia.org/wiki/Harold_Hotelling)
* [保罗·萨缪尔森](https://en.wikipedia.org/wiki/Paul_Samuelson)
* [肯尼思·阿罗](https://en.wikipedia.org/wiki/Kenneth_Arrow)
* [杰拉德·德布鲁](https://en.wikipedia.org/wiki/G%C3%A9rard_Debreu)

我们将描述两个经典的福利定理：

* **第一福利理论：** 对于消费者之间的既定财富分配，竞争性均衡的商品分配解决了社会规划问题。

* **第二福利理论：** 解决社会规划问题的商品分配可以通过适当的初始财富分配在竞争均衡中得到支持。

像往常一样，我们先导入一些Python模块。

```{code-cell} ipython3
# 导入一些包
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
```
图像输入功能：已启用

## 线性代数公式

我们将应用以下线性代数公式

* 对每个向量求内积的微分
* 对向量与矩阵的乘积求微分
* 对向量的二次型求微分

其中 $a$ 是一个 $n \times 1$ 的向量，$A$ 是一个 $n \times n$ 的矩阵，$x$ 是一个 $n \times 1$ 的向量:

$$
\frac{\partial a^\top x }{\partial x} = \frac{\partial x^\top a }{\partial x} = a
$$

$$
\frac{\partial A x} {\partial x} = A
$$

$$
\frac{\partial x^\top A x}{\partial x} = (A + A^\top)x
$$

## 从效用函数到需求曲线

我们对消费者的研究将使用以下基本模型

* $\Pi$ 是一个 $m \times n$ 的矩阵,
* $b$ 是一个 $m \times 1$ 的幸福点向量,
* $e$ 是一个 $n \times 1$ 的禀赋向量, 和

+++

我们将分析内生对象 $c$ 和 $p$，其中

* $c$ 是一个消费各种商品的 $n \times 1$ 的向量,
* $p$ 是一个价格的 $n \times 1$ 的向量

+++

矩阵 $\Pi$ 描述了消费者愿意用一种商品替代每一种其他商品的情况。

我们假设 $\Pi$ 的列是线性独立的，这意味着 $\Pi^\top \Pi$ 是一个正定矩阵。

* 由此可得 $\Pi^\top \Pi$ 有逆矩阵。

我们将在下文看到 $(\Pi^\top \Pi)^{-1}$ 是受补偿的需求曲线对价格向量的斜率矩阵:

$$
    \frac{\partial c } {\partial p} = (\Pi^\top \Pi)^{-1}
$$

消费者作为价格接受者面临 $p$，并选择 $c$ 以最大化效用函数

$$
    - \frac{1}{2} (\Pi c -b) ^\top (\Pi c -b )
$$ (eq:old0)

在预算约束的条件下

$$
    p^\top (c -e ) = 0
$$ (eq:old2)

我们将通过指定一些 $\Pi$ 和 $b$ 的例子，以说明通常发生的情况是

$$
    \Pi c \ll b
$$ (eq:bversusc)

这意味着消费者所拥有的每一种商品的数量都远少于他所想要的数量。

{eq}`eq:bversusc` 中的偏差将最终确保我们竞争均衡价格是正的。

+++

### 由受限效用最大化隐含的需求曲线

现在，我们假设预算约束是 {eq}`eq:old2`。

因此我们要推导的是一个 **马歇尔需求曲线**。

我们的目标是最大化 {eq:old0} 在 {eq:old2} 的约束下。

形成Lagrangian

$$ L = - \frac{1}{2} (\Pi c -b)^\top (\Pi c -b ) + \mu [p^\top (e-c)] $$

其中 $\mu$ 是一个拉格朗日乘数，通常被称为 **财富的边际效用**。

消费者选择 $c$ 最大化 $L$，并选择 $\mu$ 最小化它。

$c$ 的一阶条件为

$$
    \frac{\partial L} {\partial c}
    = - \Pi^\top \Pi c + \Pi^\top b - \mu p = 0
$$

因此，给定 $\mu$，消费者选择

$$
    c = (\Pi^\top \Pi )^{-1}(\Pi^\top b -  \mu p )
$$ (eq:old3)

将 {eq}`eq:old3` 代入预算约束 {eq}`eq:old2` 并求解 $\mu$ 得到

$$
    \mu(p,e) = \frac{p^\top ( \Pi^\top \Pi )^{-1} \Pi^\top b - p^\top e}{p^\top (\Pi^\top \Pi )^{-1} p}.
$$ (eq:old4)

方程 {eq}`eq:old4` 表示了财富的边际效用如何依赖于禀赋向量 $e$ 和价格向量 $p$。

```{note}
方程 {eq}`eq:old4` 是强加了 $p^\top (c - e) = 0$ 的结果。

我们可以将 $\mu$ 作为参数，并使用 {eq}`eq:old3` 和预算约束 {eq}`eq:old2p` 来求解财富。

我们使用哪种方式决定了我们是在构建**马歇尔**需求曲线还是**希克斯**需求曲线。
```

## 禀赋经济

我们现在研究一个纯交换经济，或者有时称为禀赋经济。

考虑一个单消费者、多种商品的无生产经济。

唯一的商品来源是单个消费者的禀赋向量 $e$。

一个竞争均衡价格向量使消费者选择 $c=e$。

这意味着均衡价格向量满足

$$
p = \mu^{-1} (\Pi^\top b - \Pi^\top \Pi e)
$$

在我们将预算约束以 {eq}`eq:old2` 形式强加的情况下，我们可以通过设置财富的边际效用 $\mu =1$（或任何其他值）来规范价格向量。

这相当于选择一个共同单位（或记账单位）来表示所有商品的价格。

（将所有价格翻倍不会影响数量或相对价格。）

我们将设置 $\mu=1$。

```{exercise}
:label: sdm_ex1

验证在 {eq}`eq:old3` 中设置 $\mu=1` 意味着公式 {eq}`eq:old4` 得到满足。

```

```{exercise}
:label: sdm_ex2

验证在 {eq}`eq:old3` 中设置 $\mu=2` 也意味着公式 {eq}`eq:old4` 得到满足。

```

这是一个计算我们经济的竞争均衡的类。

```{code-cell} ipython3
class ExchangeEconomy:
    
    def __init__(self, 
                 Π, 
                 b, 
                 e,
                 thres=1.5):
        """
        设置一个交换经济的环境

        参数：
            Π (np.array): 替代的共享矩阵
            b (list): 消费者的幸福点
            e (list): 消费者的禀赋
            thres (float): 检查 p >> Π e 条件的阈值
        """

        # 检查非饱和
        if np.min(b / np.max(Π @ e)) <= thres:
            raise Exception('将幸福点设置得更远')

        self.Π, self.b, self.e = Π, b, e

    
    def competitive_equilibrium(self):
        """
        计算竞争均衡价格和分配
        """
        Π, b, e = self.Π, self.b, self.e

        # 计算 μ=1 的价格向量
        p = Π.T @ b - Π.T @ Π @ e
        
        # 计算消费向量
        slope_dc = inv(Π.T @ Π)
        Π_inv = inv(Π)
        c = Π_inv @ b - slope_dc @ p

        if any(c < 0):
            print('分配: ', c)
            raise Exception('负分配: 均衡不存在')

        return p, c
```

让我们给出两个特殊例子，以展示如何应用这些知识。

### 例 1

我们假设一个消费者在两种商品上有以下效用函数：

$$
- \frac{1}{2} [(c_1 - 10)^2 + (c_2 - 20)^2]
$$

这里 $c_1$ 和 $c_2$ 分别代表商品 1 和商品 2 的消费量。消费者有以下禀赋：

$$
e_1 = 1, \quad e_2 = 2
$$

我们可以使用以下矩阵和向量来表示我们的模型：

$$
\Pi = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad b = \begin{bmatrix} 10 \\ 20 \end{bmatrix}, \quad e = \begin{bmatrix} 1 \\ 2 \end{bmatrix}
$$

我们可以用我们的类来计算竞争均衡。

```python
Π = np.array([[1, 0], [0, 1]])
b = np.array([10, 20])
e = np.array([1, 2])

economy = ExchangeEconomy(Π=b, e=e)
prices, allocation = economy.competitive_equilibrium()

print("价格:", prices)
print("分配:", allocation)
```

### 例 2

我们假设另一个消费者在两种商品上有以下效用函数：

$$
- \frac{1}{2} [(c_1 - 5)^2 + (c_2 - 15)^2]
$$

这里 $c_1$ 和 $c_2$ 分别代表商品 1 和商品 2 的消费量。消费者有以下禀赋：

$$
e_1 = 3, \quad e_2 = 4
$$

我们可以用以下矩阵和向量来表示我们的模型：

$$
\Pi = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad b = \begin{bmatrix} 5 \\ 15 \end{bmatrix}, \quad e = \begin{bmatrix} 3 \\ 4 \end{bmatrix}
$$

我们可以用我们的类来计算竞争均衡。

```python
Π = np.array([[1, 0], [0, 1]])
b = np.array([5, 15])
e = np.array([3, 4])

economy = ExchangeEconomy(Π=b, e=e)
prices, allocation = economy.competitive_equilibrium()

print("价格:", prices)
print("分配:", allocation)
```

## 插播话题：马歇尔和希克斯需求曲线

有时我们会在消费的禀赋向量 $e$ 是消费者的唯一收入来源的情况下使用预算约束 {eq}`eq:old2`。

其他时候，我们假设消费者有另一种收入来源（正或负）并写出他的预算约束为

$$
p ^\top (c -e ) = w
$$ (eq:old2p)

其中 $w$ 以"美元"（或其他**记账单位**）衡量，价格向量的分量 $p_i$ 以每单位商品 $i$ 的美元计。

无论消费者的预算约束是 {eq}`eq:old2` 还是 {eq}`eq:old2p`，以及我们是否将 $w$ 作为一个自由参数或作为一个内生变量，都将影响消费者的边际财富效用。

因此，我们如何设置 $\mu$ 决定了我们是在构建

* **马歇尔**需求曲线，例如当我们使用 {eq}`eq:old2` 并用上述方程 {eq}`eq:old4` 求解 $\mu$ 时，或
* **希克斯**需求曲线，例如当我们将 $\mu$ 视为一个固定参数并从 {eq}`eq:old2p` 中求解 $w$ 时。

马歇尔和希克斯需求曲线考虑不同的心理实验：

对于马歇尔需求曲线，假设的价格向量变化具有**替代**和**收入**效应

* 收入效应是价格向量变化关联的 $p^\top e$ 的变化

对于希克斯需求曲线，假设的价格向量变化只有**替代**效应

* 价格向量的变化不会改变 $p^\top e + w$，因为我们冻结 $\mu$ 并求解 $w$

有时希克斯需求曲线被称为**补偿的**需求曲线，以强调通过调整消费者的财富 $w$ 来解除价格变化关联的收入（或财富）效应。

我们将在下文更详细地讨论这些不同的需求曲线。

+++

## 动态性和风险作为特殊情况

我们的 $n$-种商品纯交换模型的特殊情况可以表示以下内容：

* **动态性** — 通过在不同商品上标注不同的日期
* **风险** — 通过解释商品的交付取决于一种已知概率分布描述的世界状态

让我们展示如何实现这些内容。

### 动态性

假设我们要表示一个效用函数

$$
  - \frac{1}{2} [(c_1 - b_1)^2 + \beta (c_2 - b_2)^2]
$$

其中 $\beta \in (0,1)$ 是一个折现因子，$c_1$ 是时间1的消费，$c_2$ 是时间2的消费。

为了将其用我们的二次效用函数 {eq}`eq:old0` 表示，设置

$$
\Pi = \begin{bmatrix} 1 & 0 \cr
         0 & \sqrt{\beta} \end{bmatrix}
$$

$$
e = \begin{bmatrix} e_1 \cr e_2 \end{bmatrix}
$$

和

$$
b = \begin{bmatrix} b_1 \cr \sqrt{\beta} b_2
\end{bmatrix}
$$

预算约束 {eq}`eq:old2` 变为

$$
p_1 c_1 + p_2 c_2 = p_1 e_1 + p_2 e_2
$$

左侧是消费的**折现现值**。

右侧是消费者禀赋的**折现现值**。

### 事件和保险合约

假设我们有以下效用函数：

$$
- \frac{1}{2} [(c_1 - b_1)^2 + \beta (c_2 - b_2)^2 + \beta c_3 - b_3]^2
$$

其中 $c_1$ 是时间1的消费，$c_2$ 是时间2在事件A中的消费，$c_3$ 是时间2在事件B中的消费。

我们有两个正的概率 $0 < \pi_A, \pi_B < 1$，且 $\pi_A + \pi_B = 1$，承担两个事件中的一个会在时间2发生。

为了将其用我们的二次效用函数 {eq}`eq:old0` 表示，我们设置

$$
\Pi = \begin{bmatrix}
1 &  0 & 0\\
0 & \sqrt{\beta p_A} & 0 \\
0 &  0 & \sqrt{\beta p_B}
\end{bmatrix}
$$

$$
e = \begin{bmatrix}
e_1 \\
e_2 \\
e_3
\end{bmatrix}
$$

和

$$
b = \begin{bmatrix}
b_1 \\
\sqrt{\beta p_A} b_2 \\
\sqrt{\beta p_B} b_3
\end{bmatrix}
$$

预算约束 {eq}`eq:old2` 变为

$$
p_1 (c_1 - e_1) + p_2 (c_2 - e_2) + p_3 (c_3 - e_3) = 0
$$

左侧是消费者由于对 $c_1, c_2, c_3$ 消费的选择改变而获得（或支付）的收入。

右侧是消费者能够从出售禀赋 $e$ 中获取的收入。

```{note}
在这种情况下，所有的未使用预算的消费品被假定以相等的单位形式转化为普通的账户单位。
注意，$z_i ≥ 0$ 制约条件被隐含。
```

我们将在这次讲座中首先回顾线性代数的一些概念，然后再讨论货币单位变化和不同需求曲线的例子。

### 风险与状态依赖的索赔

我们在一个**静态**环境中研究风险，这意味着只有一个时期。

**风险**意味着结果事先未知，但它是由已知的概率分布所控制的。

例如，当我们说消费者面对**风险**时，具体意味着

  * 存在两种自然状态，$1$ 和 $2$。

  * 消费者知道状态 $1$ 发生的概率是 $\lambda$。

  * 消费者知道状态 $2$ 发生的概率是 $(1-\lambda)$。

在结果实现之前，消费者的**期望效用**是

$$
- \frac{1}{2} [\lambda (c_1 - b_1)^2 + (1-\lambda)(c_2 - b_2)^2]
$$

其中

* $c_1$ 是状态 $1$ 的消费
* $c_2$ 是状态 $2$ 的消费

为了捕捉这些偏好，我们设定

$$
\Pi = \begin{bmatrix} \sqrt{\lambda} & 0 \cr
                     0  & \sqrt{1-\lambda} \end{bmatrix}
$$

$$
e = \begin{bmatrix} e_1 \cr e_2 \end{bmatrix}
$$

+++

$$
b = \begin{bmatrix} \sqrt{\lambda}b_1 \cr \sqrt{1-\lambda}b_2 \end{bmatrix}
$$

消费者的禀赋向量是

$$
c = \begin{bmatrix} c_1 \cr c_2 \end{bmatrix}
$$

一个价格向量是

$$
p = \begin{bmatrix} p_1 \cr p_2 \end{bmatrix}
$$

其中 $p_i$ 是状态 $i \in \{1, 2\}$ 下消费品的一个单位的价格。

这些状态依赖的交易品通常被称为**阿罗证券**。

在世界的随机状态 $i$ 实现之前，消费者出售他/她的状态依赖的禀赋组合，并购买状态依赖的消费组合。

交易这些状态依赖的商品是经济学家建模**保险**的一种方式。

+++

我们使用上述技巧将 $c_1, c_2$ 解释为状态依赖的消费品“阿罗证券”。

+++

以下是风险经济的一个实例：

```{code-cell} ipython3
prob = 0.2

Π = np.array([[np.sqrt(prob), 0],
              [0, np.sqrt(1 - prob)]])

b = np.array([np.sqrt(prob) * 5, np.sqrt(1 - prob) * 5])

e = np.array([1, 1])

risk = ExchangeEconomy(Π, b, e)
p, c = risk.competitive_equilibrium()

print('竞争均衡价格向量:', p)
print('竞争均衡分配:', c)
```

## 比较静态分析

回顾风险经济例子中的竞争性均衡：

```{code-cell} ipython3
prob = 0.2

Π = np.array([[np.sqrt(prob), 0],
              [0, np.sqrt(1 - prob)]])

b = np.array([np.sqrt(prob) * 5, np.sqrt(1 - prob) * 5])

e = np.array([1, 1])

risk = ExchangeEconomy(Π, b, e)
p, c = risk.competitive_equilibrium()

print('竞争均衡价格向量:', p)
print('竞争均衡分配:', c)
```

```{exercise}
:label: sdm_ex3

考虑上述情况。

请通过数值研究以下情况如何影响均衡价格和分配：

* 消费者变穷了，
* 他们对第一种商品的喜爱程度更高，或者
* 状态 $1$ 发生的概率更高。

提示。对每种情况选择与实例不同的参数 $e, b \text{ 或 } \lambda$。

```

+++

```{solution-start} sdm_ex3
:class: dropdown
```

首先考虑当消费者变穷时。

这里我们只需要减少禀赋。

```{code-cell} ipython3
risk.e = np.array([0.5, 0.5])

p, c = risk.competitive_equilibrium()

print('竞争均衡价格向量:', p)
print('竞争均衡分配:', c)
```

当消费者变得更穷时：

```{code-cell} ipython3
prob = 0.2

Π = np.array([[np.sqrt(prob), 0],
              [0, np.sqrt(1 - prob)]])

b = np.array([np.sqrt(prob) * 5, np.sqrt(1 - prob) * 5])

e = np.array([0.5, 0.5])

risk = ExchangeEconomy(Π, b, e)
p, c = risk.competitive_equilibrium()

print('竞争均衡价格向量:', p)
print('竞争均衡分配:', c)
```

我们观察到竞争均衡中的分配减少了。

+++

接下来考虑消费者更喜欢第一种商品的情况。

```{code-cell} ipython3
risk.b = np.array([np.sqrt(prob) * 6, np.sqrt(1 - prob) * 5])
p, c = risk.competitive_equilibrium()

print('竞争均衡价格向量:', p)
print('竞争均衡分配:', c)
```

消费者对第一种商品的更高偏好增加了第一个状态的需求，从而改变均衡价格和分配。

+++

最后，假设状态 $1$ 发生的概率更高。

```{code-cell} ipython3
prob = 0.8

risk.Π = np.array([[np.sqrt(prob), 0],
                   [0, np.sqrt(1 - prob)]])

risk.b = np.array([np.sqrt(prob) * 5, np.sqrt(1 - prob) * 5])
p, c = risk.competitive_equilibrium()

print('竞争均衡价格向量:', p)
print('竞争均衡分配:', c)
```

状态 $1$ 发生的更高概率增加了对第一个状态的需求，这也改变均衡价格和分配。

```{solution-end}
```


让我们总结一下我们的很多论点。

## 总结

我们研究了一个具有多个商品的市场，并描述了它的竞争均衡。

我们还讨论了不同的需求曲线，以及它们如何受到消费者选择的参数影响。

我们通过分析几种特殊情况，如时间相关的效用函数和风险相关的效用函数，描述了如何应用这些基本概念。

最后，我们进行了比较静态分析，探讨了改变一些参数如何影响均衡价格和分配。

这些基础知识在经济学的更广阔领域中广泛应用，并且是理解更复杂的市场动力学的重要组成部分。

```

请定义一个函数，该函数绘制需求和供给曲线，并标记出剩余和均衡。

```{code-cell} ipython3
:tags: [hide-input]

def plot_competitive_equilibrium(PE):
    """
    绘制需求和供给曲线，标记生产者/消费者剩余和均衡点

    参数:
        PE (class): 初始化的生产经济类
    """
    # 获取单一值
    J, h, Π, b, μ = PE.J.item(), PE.h.item(), PE.Π.item(), PE.b.item(), PE.μ
    H = J

    # 计算竞争均衡
    c, p = PE.competitive_equilibrium()
    c, p = c.item(), p.item()

    # 逆供给/需求曲线
    supply_inv = lambda x: h + H * x
    demand_inv = lambda x: 1 / μ * (Π * b - Π * Π * x)

    xs = np.linspace(0, 2 * c, 100)
    ps = np.ones(100) * p
    supply_curve = supply_inv(xs)
    demand_curve = demand_inv(xs)

    # 绘制
    plt.figure()
    plt.plot(xs, supply_curve, label='供给', color='#020060')
    plt.plot(xs, demand_curve, label='需求', color='#600001')

    plt.fill_between(xs[xs <= c], demand_curve[xs <= c], ps[xs <= c], label='消费者剩余', color='#EED1CF')
    plt.fill_between(xs[xs <= c], supply_curve[xs <= c], ps[xs <= c], label='生产者剩余', color='#E6E6F5')

    plt.vlines(c, 0, p, linestyle="dashed", color='black', alpha=0.7)
    plt.hlines(p, 0, c, linestyle="dashed", color='black', alpha=0.7)
    plt.scatter(c, p, zorder=10, label='竞争均衡', color='#600001')

    plt.legend(loc='upper right')
    plt.margins(x=0, y=0)
    plt.ylim(0)
    plt.xlabel('数量')
    plt.ylabel('价格')
    plt.show()
```



用一个商品构建一个生产经济的例子。

现在让我们构建一个有一个商品的生产经济的例子。

为此我们

* 指定一个单一的**人**和一个**成本曲线**，以便我们可以复制我们开始的简单单商品供需例子

* 计算均衡价格 $p$ 和均衡消费量 $c$ 以及消费者和生产者剩余

* 绘制两者的图表

* 做一个实验，改变 $b$ 并观察 $p$ 和 $c$ 会发生什么。

```{code-cell} ipython3
Π = np.array([[1]])  # 矩阵现在是单一矩阵
b = np.array([10])
h = np.array([0.5])
J = np.array([[1]])
μ = 1

PE = ProductionEconomy(Π, b, h, J, μ)
c, p = PE.competitive_equilibrium()

print('竞争均衡价格:', p.item())
print('竞争均衡分配:', c.item())

# 绘制
plot_competitive_equilibrium(PE)
```

在我们继续之前，让我们重新计算和绘制消费者和生产者剩余。

```{code-cell} ipython3
c_surplus, p_surplus = PE.compute_surplus()

print('消费者剩余:', c_surplus.item())
print('生产者剩余:', p_surplus.item())
```

现在我们增大 \(b\)

```{code-cell} ipython3
# 增大消费者的需求
b = np.array([20])
PE = ProductionEconomy(Π, b, h, J, μ)

c, p = PE.competitive_equilibrium()
print('更新后的竞争均衡价格:', p.item())
print('更新后的竞争均衡分配:', c.item())

plot_competitive_equilibrium(PE)

c_surplus, p_surplus = PE.compute_surplus()
print('更新后的消费者剩余:', c_surplus.item())
print('更新后的生产者剩余:', p_surplus.item())
```

我们可以看到，需求增加导致均衡价格和分配都提高了。消费者和生产者的剩余也相应地提高了。通过这种比较静态分析，我们可以深入了解市场参数变化对均衡的影响。

以上就是一个用单一商品构建生产经济的完整例子，我们计算并绘制了均衡价格、均衡分配以及消费者和生产者剩余，并进行了参数变化的实验，观察到市场均衡状态的调整。

让我们给消费者一个更低的福利权重，通过提高 $\mu$。

```{code-cell} ipython3
PE.μ = 2
c, p = PE.competitive_equilibrium()

print('竞争均衡价格:', p.item())
print('竞争均衡分配:', c.item())

# 绘制
plot_competitive_equilibrium(PE)
```

这次，消费者的边际效用增加了，导致价格和分配均发生了变化。

最后，我们将总结通过供需曲线分析多种商品市场的关键要点。

## 总结

在这节课中，我们研究了多种商品的供需市场，讨论了逆需求曲线、财富的边际效用、消费者剩余和生产者剩余等核心概念，并介绍了竞争均衡的概念。

我们通过分析特定的例子，如动态性和风险如何影响经济，以及通过比较静态分析展示了参数变化如何影响市场均衡。

通过这些分析，我们了解了供需模型的基本结构和应用，并讨论了价格变化如何通过不同的消费者和生产者剩余反映在经济福利中。

这些概念在经济学中的应用非常广泛，不仅适用于单一市场，还适用于更复杂的多商品市场和不同的经济环境。理解这些基本原理将有助于我们在更复杂的经济分析中应用这些工具和方法。

现在我们改变幸福点，使消费者从消费中获得更多的效用。

```{code-cell} ipython3
PE.μ = 1
PE.b = PE.b * 1.5
c, p = PE.competitive_equilibrium()

print('竞争均衡价格:', p.item())
print('竞争均衡分配:', c.item())

# 绘制
plot_competitive_equilibrium(PE)
```

这次，通过提高幸福点，我们能够看到需求增加，导致均衡价格和分配都升高。

我们总结了通过需求和供给曲线分析多种商品市场的关键要点，并了解了均衡价格和数量如何受到模型参数变化的影响。

具有这些知识，您将能够在更复杂的经济情境中应用这些原理来分析市场动态和经济福利。

```{code-cell} ipython3
PE.b = np.array([12, 10])

c, p = PE.competitive_equilibrium()

print('竞争均衡价格:', p)
print('竞争均衡分配:', c)
```

```{code-cell} ipython3
# 更新后的幸福点
PE.b = np.array([12, 10])

c, p = PE.competitive_equilibrium()

print('竞争均衡价格:', p)
print('竞争均衡分配:', c)
```

您通过这些例子可以看到，不同的模型参数（例如幸福点、禀赋、边际效用）对竞争均衡价格和分配有显著影响。希望这些示例和总结能帮助您更好地理解和分析多商品市场中的供需均衡和经济福利。

```{code-cell} ipython3
PE.b = np.array([12, 10])
c, p = PE.competitive_equilibrium()

print('竞争均衡价格:', p)
print('竞争均衡分配:', c)
```

以上例子展示了如何在参数发生变化时计算和绘制多商品生产经济的竞争均衡价格和分配。通过这些例子，您可以直观地看到幸福点、禀赋、边际效用等参数对市场均衡的影响。

这些示例和总结有助于您更好地理解和分析多商品市场中的供需关系和经济福利。

希望这些内容能够帮助您应用供需模型进行复杂的经济分析，并了解市场动态中的关键因素和它们的相互作用。

定义一个函数，该函数绘制需求、边际成本和边际收益曲线，并标记出剩余和均衡。

```{code-cell} ipython3
:tags: [hide-input]

def plot_monopoly(M):
    """
    绘制需求曲线、边际生产成本和收益、剩余和垄断供应经济中的均衡点

    参数:
        M (class): 一个继承自 ProductionEconomy 的垄断类
    """
    # 获取单一值
    J, h, Π, b, μ = M.J.item(), M.h.item(), M.Π.item(), M.b.item(), M.μ
    H = J

    # 计算竞争均衡
    c, p = M.competitive_equilibrium()
    q, pm = M.equilibrium_with_monopoly()
    c, p, q, pm = c.item(), p.item(), q.item(), pm.item()

    # 计算

    # 逆供给/需求曲线
    marg_cost = lambda x: h + H * x
    marg_rev = lambda x: -2 * 1 / μ * Π * Π * x + 1 / μ * Π * b
    demand_inv = lambda x: 1 / μ * (Π * b - Π * Π * x)

    xs = np.linspace(0, 2 * c, 100)
    pms = np.ones(100) * pm
    marg_cost_curve = marg_cost(xs)
    marg_rev_curve = marg_rev(xs)
    demand_curve = demand_inv(xs)

    # 绘制
    plt.figure()
    plt.plot(xs, marg_cost_curve, label='边际成本', color='#020060')
    plt.plot(xs, marg_rev_curve, label='边际收益', color='#E55B13')
    plt.plot(xs, demand_curve, label='需求', color='#600001')

    plt.fill_between(xs[xs <= q], demand_curve[xs <= q], pms[xs <= q], label='消费者剩余', color='#EED1CF')
    plt.fill_between(xs[xs <= q], marg_cost_curve[xs <= q], pms[xs <= q], label='生产者剩余', color='#E6E6F5')

    plt.vlines(c, 0, p, linestyle="dashed", color='black', alpha=0.7)
    plt.hlines(p, 0, c, linestyle="dashed", color='black', alpha=0.7)
    plt.scatter(c, p, zorder=10, label='竞争均衡', color='#600001')

    plt.vlines(q, 0, pm, linestyle="dashed", color='black', alpha=0.7)
    plt.hlines(pm, 0, q, linestyle="dashed", color='black', alpha=0.7)
    plt.scatter(q, pm, zorder=10, label='垄断均衡', color='#E55B13')

    plt.legend(loc='upper right')
    plt.margins(x=0, y=0)
    plt.ylim(0)
    plt.xlabel('数量')
    plt.ylabel('价格')
    plt.show()
```

## 垄断

我们将展示如何在多种商品经济的竞争性均衡和垄断均衡之间进行比较。

```{code-cell} ipython3
Π = np.array([[1, 0],
              [0, 1.2]])

b = np.array([10, 10])

h = np.array([0.5, 0.5])

J = np.array([[1, 0.5],
              [0.5, 1]])
μ = 1

M = Monopoly(Π, b, h, J, μ)
c, p = M.competitive_equilibrium()
q, pm = M.equilibrium_with_monopoly()

print('竞争均衡价格:', p)
print('竞争均衡分配:', c)
print('垄断均衡供应商价格:', pm)
print('垄断均衡供应商分配:', q)
```

## 单一商品的例子

```{code-cell} ipython3
Π = np.array([[1]])  # 矩阵现在是单一矩阵
b = np.array([10])
h = np.array([0.5])
J = np.array([[1]])
μ = 1

M = Monopoly(Π, b, h, J, μ)
c, p = M.competitive_equilibrium()
q, pm = M.equilibrium_with_monopoly()

print('竞争均衡价格:', p.item())
print('竞争均衡分配:', c.item())

print('垄断均衡供应商价格:', pm.item())
print('垄断均衡供应商分配:', q.item())

# 绘制
plot_monopoly(M)
```

## 多种商品的福利最大化问题

我们的福利最大化问题 -- 有时也称为社会计划问题 -- 是选择 $c$ 以最大化

$$
- \frac{1}{2} \mu^{-1}(\Pi c -b) ^\top (\Pi c -b )
$$

减去逆供给曲线下的面积，即

$$
h c +  \frac{1}{2} c^\top J c
$$

所以福利标准是

$$
- \frac{1}{2} \mu^{-1}(\Pi c -b)^\top (\Pi c -b ) -h c 
   -  \frac{1}{2} c^\top J c
$$

在这个公式中，$\mu$ 是一个描述计划者如何权衡外部供应商和我们代表性消费者利益的参数。

关于 $c$ 的一阶条件是

$$
- \mu^{-1} \Pi^\top \Pi c + \mu^{-1}\Pi^\top b - h -  H c = 0
$$

这意味着 {eq}`eq:old5p`。

因此，对于单一商品的情况，具有多种商品的竞争均衡数量向量解决了规划问题。

（这是第一福利定理的另一种版本。）

我们可以从以下任一方式推导出一个竞争均衡价格向量：

  * 逆需求曲线，或
  * 逆供给曲线