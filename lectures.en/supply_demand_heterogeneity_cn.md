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

(supply_demand_heterogeneity)=
# 市场均衡与异质性

## 概述

在 {doc}`上一讲 <supply_demand_multiple_goods>` 中，我们研究了多商品经济中的竞争均衡。

尽管研究结果很有启发性，但我们使用了一个强有力的简化假设：经济中的所有主体都是相同的。

在现实世界中，家庭、公司和其他经济主体在很多方面都是不同的。

在本讲中，我们通过允许消费者的偏好和禀赋不同来引入异质性。

我们将研究在这种情况下的竞争均衡。

我们还将展示如何构建“代表性消费者”。

以下是一些导入：

```{code-cell} ipython3
import numpy as np
from scipy.linalg import inv
```
Image input capabilities: Enabled

## 一个简单的例子

让我们研究一个没有生产的**纯粹交换**经济的简单例子。

有两个消费者，他们在禀赋向量 $e_i$ 和他们的幸福点向量 $b_i$ 上有所不同，对于 $i=1,2$。

总禀赋是 $e_1 + e_2$。

一个竞争均衡要求

$$
c_1 + c_2 = e_1 + e_2
$$

假设需求曲线为

$$
    c_i = (\Pi^\top \Pi )^{-1}(\Pi^\top b_i -  \mu_i p )
$$

竞争均衡需要

$$
e_1 + e_2 =
    (\Pi^\top \Pi)^{-1}(\Pi^\top (b_1 + b_2) - (\mu_1 + \mu_2) p )
$$

这在进行一两行线性代数操作后，意味着

$$
(\mu_1 + \mu_2) p = \Pi^\top(b_1+ b_2) - \Pi^\top \Pi (e_1 + e_2)
$$ (eq:old6)

我们可以通过设定 $\mu_1 + \mu_2 =1$ 来规范价格，然后求解

$$
\mu_i(p,e) = \frac{p^\top (\Pi^{-1} b_i - e_i)}{p^\top (\Pi^\top \Pi )^{-1} p}
$$ (eq:old7)

其中 $i = 1,2$。

```{exercise-start}
:label: sdh_ex1
```

展示在上一节的两消费者经济中计算出的竞争均衡价格向量（经过正标量归一化）将在只有一个消费者的经济中仍然成立，其中单个**代表性消费者**具有效用函数

$$
-.5 (\Pi c -b) ^\top (\Pi c -b )
$$

以及禀赋向量 $e$，其中

$$
b = b_1 + b_2
$$

和

$$
e = e_1 + e_2 .
$$

```{exercise-end}
```

## 纯粹交换经济

让我们进一步探索一个有 $n$ 种商品和 $m$ 个人的纯粹交换经济。

### 竞争均衡

我们将计算一个竞争均衡。

为了计算一个纯粹交换经济的竞争均衡，我们使用以下事实：

- 在竞争均衡中的相对价格与特殊单消费者或代表性消费者经济中的偏好 $\Pi$ 和 $b=\sum_i b_i$ 以及禀赋 $e = \sum_i e_{i}$ 是相同的。

我们可以使用以下步骤来计算一个竞争均衡：

- 首先我们通过将 $\mu = 1$ 进行规范化来求解单个代表性消费者经济。然后，我们通过使用第一个消费品作为计量标准重新规范价格向量。

- 接下来我们使用竞争均衡价格计算每个消费者的边际财富效用：

$$
\mu_{i}=\frac{-W_{i}+p^{\top}\left(\Pi^{-1}b_{i}-e_{i}\right)}{p^{\top}(\Pi^{\top}\Pi)^{-1}p}$$

- 最后我们通过使用需求曲线计算一个竞争均衡分配：

$$
c_{i}=\Pi^{-1}b_{i}-(\Pi^{\top}\Pi)^{-1}\mu_{i}p 
$$

### 设计一些 Python 代码

下面我们将构建一个 Python 类，具有以下属性：

* **偏好** 以以下形式

    * 一个 $n \times n$ 的正定矩阵 $\Pi$
    * 一个 $n \times 1$ 的幸福点向量 $b$

* **禀赋** 以以下形式

    * 一个 $n \times 1$ 的向量 $e$
    * 一个标量“财富” $W$ 默认值为 0

该类将包括一个测试，以确保 $b \gg \Pi e$ 并在其被违反时引发异常（在我们必须指定的某个阈值水平）。

构建类时，我们将实现方法来计算竞争均衡的价格和分配。

### 类和方法定义

以下代码定义了 `ExchangeEconomy` 类及其方法：

```{code-cell} ipython3
class ExchangeEconomy:
    def __init__(self, 
                 Π, 
                 bs, 
                 es, 
                 Ws=None, 
                 thres=1.5):
        """
        设置交换经济环境

        参数:
            Π (np.array): 替代矩阵
            bs (list): 所有消费者的幸福点
            es (list): 所有消费者的禀赋
            Ws (list): 所有消费者的财富
            thres (float): 检测 b >> Pi e 是否违反的阈值
        """
        n, m = Π.shape[0], len(bs)

        # 检查非饱和性
        for b, e in zip(bs, es):
            if np.min(b / np.max(Π @ e)) <= thres:
                raise Exception('请将幸福点设置得更远')

        if Ws == None:
            Ws = np.zeros(m)
        else:
            if sum(Ws) != 0:
                raise Exception('无效的财富分配')

        self.Π, self.bs, self.es, self.Ws, self.n, self.m = Π, bs, es, Ws, n, m

    def competitive_equilibrium(self):
        """
        计算竞争均衡价格和分配
        """
        Π, bs, es, Ws = self.Π, self.bs, self.es, self.Ws
        n, m = self.n, self.m
        slope_dc = inv(Π.T @ Π)
        Π_inv = inv(Π)

        # 总体
        b = sum(bs)
        e = sum(es)

        # 使用 mu=1 计算价格向量并重新规范化
        p = Π.T @ b - Π.T @ Π @ e
        p = p / p[0]

        # 计算财富的边际效用
        μ_s = []
        c_s = []
        A = p.T @ slope_dc @ p

        for i in range(m):
            μ_i = (-Ws[i] + p.T @ (Π_inv @ bs[i] - es[i])) / A
            c_i = Π_inv @ bs[i] - μ_i * slope_dc @ p
            μ_s.append(μ_i)
            c_s.append(c_i)

        for c_i in c_s:
            if any(c_i < 0):
                print('分配: ', c_s)
                raise Exception('负分配: 均衡不存在')

        return p, c_s, μ_s
```

这个类首先通过其 `__init__` 方法设置交换经济的环境，包括消费者的偏好和禀赋。`competitive_equilibrium` 方法则用于计算竞争均衡的价格和分配。

### 示例

我们可以创建一个 `ExchangeEconomy` 实例并计算竞争均衡：

```{code-cell} ipython3
# 设定参数
Π = np.array([[1, 0.5], [0.5, 1]])
bs = [np.array([10, 10]), np.array([20, 5])]
es = [np.array([5, 5]), np.array([5, 5])]
Ws = [0, 0]

# 创建交换经济实例
economy = ExchangeEconomy(Π, bs, es, Ws)

# 计算竞争均衡
p, c_s, μ_s = economy.competitive_equilibrium()

# 打印结果
print("价格向量 p: ", p)
print("均衡分配 c_s: ", c_s)
print("边际效用 μ_s: ", μ_s)
```

以上代码将计算并输出竞争均衡的价格向量、均衡分配以及边际效用。

## 实现

接下来我们使用上面定义的类 `ExchangeEconomy` 来研究

* 一个没有生产的两人经济，
* 一个动态经济，和
* 一个带有风险和箭证券的经济。

### 没有生产的两人经济

在这里我们研究不同 $b_i$ 和 $e_i$, $i \in \{1, 2\}$ 对竞争均衡 $p, c_1, c_2$ 的影响。

```{code-cell} ipython3
Π = np.array([[1, 0],
              [0, 1]])

bs = [np.array([5, 5]),  # 第一个消费者的幸福点
      np.array([5, 5])]  # 第二个消费者的幸福点

es = [np.array([0, 2]),  # 第一个消费者的禀赋
      np.array([2, 0])]  # 第二个消费者的禀赋

EE = ExchangeEconomy(Π, bs, es)
p, c_s, μ_s = EE.competitive_equilibrium()

print('竞争均衡价格向量:', p)
print('竞争均衡分配:', c_s)
```

会发生什么如果第一个消费者更喜欢第一个商品，第二个消费者更喜欢第二个商品？

```{code-cell} ipython3
EE.bs = [np.array([6, 5]),  # 第一个消费者的幸福点
         np.array([5, 6])]  # 第二个消费者的幸福点

p, c_s, μ_s = EE.competitive_equilibrium()

print('竞争均衡价格向量:', p)
print('竞争均衡分配:', c_s)
```

让第一个消费者变得更贫穷。

```{code-cell} ipython3
EE.es = [np.array([0.5, 0.5]),  # 第一个消费者的禀赋
         np.array([1, 1])]  # 第二个消费者的禀赋

p, c_s, μ_s = EE.competitive_equilibrium()

print('竞争均衡价格向量:', p)
print('竞争均衡分配:', c_s)
```

现在我们构建一个自给自足（即无交易）的均衡。

```{code-cell} ipython3
EE.bs = [np.array([4, 6]),  # 第一个消费者的幸福点
      np.array([6, 4])]  # 第二个消费者的幸福点

EE.es = [np.array([0, 2]),  # 第一个消费者的禀赋
      np.array([2, 0])]  # 第二个消费者的禀赋

p, c_s, μ_s = EE.competitive_equilibrium()

print('竞争均衡价格向量:', p)
print('竞争均衡分配:', c_s)
```

现在让我们在贸易前重新分配禀赋。

```{code-cell} ipython3
bs = [np.array([5, 5]),  # 第一个消费者的幸福点
      np.array([5, 5])]  # 第二个消费者的幸福点

es = [np.array([1, 1]),  # 第一个消费者的禀赋
      np.array([1, 1])]  # 第二个消费者的禀赋

Ws = [0.5, -0.5]
EE_new = ExchangeEconomy(Π, bs, es, Ws)
p, c_s, μ_s = EE_new.competitive_equilibrium()

print('竞争均衡价格向量:', p)
print('竞争均衡分配:', c_s)
```

### 一个动态经济

现在让我们使用上述技巧来研究一个动态经济，即一个有两个时期的经济。

```{code-cell} ipython3
beta = 0.95

Π = np.array([[1, 0],
              [0, np.sqrt(beta)]])

bs = [np.array([5, np.sqrt(beta) * 5])]

es = [np.array([1, 1])]

EE_DE = ExchangeEconomy(Π, bs, es)
p, c_s, μ_s = EE_DE.competitive_equilibrium()

print('竞争均衡价格向量:', p)
print('竞争均衡分配:', c_s)
```

### 带箭证券的风险经济

我们使用上述技巧，将 $c_1, c_2$ 解释为状态依赖的消费品的 "箭证券"。

```{code-cell} ipython3
prob = 0.7

Π = np.array([[np.sqrt(prob), 0],
              [0, np.sqrt(1 - prob)]])

bs = [np.array([np.sqrt(prob) * 5, np.sqrt(1 - prob) * 5]),
      np.array([np.sqrt(prob) * 5, np.sqrt(1 - prob) * 5])]

es = [np.array([1, 0]),
      np.array([0, 1])]

EE_AS = ExchangeEconomy(Π, bs, es)
p, c_s, μ_s = EE_AS.competitive_equilibrium()

print('竞争均衡价格向量:', p)
print('竞争均衡分配:', c_s)
```

## 推导一个代表性消费者

在这里我们研究的多消费者经济类中，事实证明存在一个单一的**代表性消费者**，其偏好和禀赋可以从单个消费者的偏好和禀赋列表中推导出来。

考虑一个初始化财富分布 $W_i$ 满足 $\sum_i W_{i}=0$ 的多消费者经济。

我们允许初始的财富再分配。

我们有以下对象：

- 需求曲线：
  
$$ 
c_{i}=\Pi^{-1}b_{i}-(\Pi^{\top}\Pi)^{-1}\mu_{i}p 
$$

- 财富的边际效用：
  
$$ 
\mu_{i}=\frac{-W_{i}+p^{\top}\left(\Pi^{-1}b_{i}-e_{i}\right)}{p^{\top}(\Pi^{\top}\Pi)^{-1}p}
$$

- 市场出清：
  
$$ 
\sum c_{i}=\sum e_{i}
$$

记总消费 $\sum_i c_{i}=c$ 和 $\sum_i \mu_i = \mu$。

市场出清要求

$$ 
\Pi^{-1}\left(\sum_{i}b_{i}\right)-(\Pi^{\top}\Pi)^{-1}p\left(\sum_{i}\mu_{i}\right)=\sum_{i}e_{i}
$$
经过几个步骤后，这将导致

$$
p=\mu^{-1}\left(\Pi^{\top}b-\Pi^{\top}\Pi e\right)
$$

其中

$$ 
\mu = \sum_i\mu_{i}=\frac{0 + p^{\top}\left(\Pi^{-1}b-e\right)}{p^{\top}(\Pi^{\top}\Pi)^{-1}p}.
$$

现在考虑上面指定的代表性消费者经济。

表示代表性消费者的财富边际效用为 $\tilde{\mu}$。

需求函数为

$$
c=\Pi^{-1}b-(\Pi^{\top}\Pi)^{-1}\tilde{\mu} p
$$

将其代入预算约束得

$$
\tilde{\mu}=\frac{p^{\top}\left(\Pi^{-1}b-e\right)}{p^{\top}(\Pi^{\top}\Pi)^{-1}p}
$$

在均衡中 $c=e$，所以

$$
p=\tilde{\mu}^{-1}(\Pi^{\top}b-\Pi^{\top}\Pi e)
$$

因此，我们已经验证了，在选择一个计量单位来表示绝对价格的情况下，我们的代表性消费者经济中的价格向量与多消费者经济中的价格向量是相同的。