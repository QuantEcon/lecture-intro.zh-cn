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

在{doc}`前一讲座<supply_demand_multiple_goods>`中，我们研究了具有多种商品的经济环境中的竞争均衡。

虽然研究结果具有启发性，但我们使用了一个强化的简化假设：经济中的所有代理人都是相同的。

实际上，家庭、公司及其他经济主体在许多方面都不相同。

在这次讲座中，我们通过允许消费者的偏好和禀赋不同，引入了消费者间的异质性。

我们将在这种设定下检验竞争均衡。

我们还将展示如何构建一个“代表性消费者”。

以下是一些导入内容：

```{code-cell} ipython3
import numpy as np
from scipy.linalg import inv
```

## 一个简单的例子

让我们学习一个没有生产的**纯交换**经济的简单例子。

有两个消费者，他们在禀赋向量 $e_i$ 和极乐点向量 $b_i$ 上有所不同，其中 $i=1,2$。

总的禀赋是 $e_1 + e_2$。

竞争均衡要求

$$
c_1 + c_2 = e_1 + e_2
$$

假设需求曲线为

$$
    c_i = (\Pi^{\top} \Pi)^{-1}(\Pi^{\top} b_i -  \mu_i p )
$$

那么竞争均衡还需要满足

$$
e_1 + e_2 =
    (\Pi^{\top} \Pi)^{-1}(\Pi^{\top} (b_1 + b_2) - (\mu_1 + \mu_2) p )
$$

通过一两行线性代数计算，可以推出

$$
(\mu_1 + \mu_2) p = \Pi^{\top}(b_1+ b_2) - \Pi^{\top} \Pi (e_1 + e_2)
$$ (eq:old6)

我们可以通过设定 $\mu_1 + \mu_2 =1$ 来规范价格，然后解方程

$$
\mu_i(p,e) = \frac{p^{\top} (\Pi^{-1} b_i - e_i)}{p^{\top} (\Pi^{\top} \Pi)^{-1} p}
$$ (eq:old7)

求得 $\mu_i, i = 1,2$。

```{exercise-start}
:label: sdh_ex1
```

证明，在正标量的归一化下，你在前面两消费者经济中计算出的同一竞争均衡价格向量，在只有一个代表性消费者的单一消费者经济中依旧适用，其中这个**代表性消费者**的效用函数为：

$$
-.5 (\Pi c -b) ^2 (\Pi c -b )
$$

其初始禀赋向量为 $e$，其中

$$
b = b_1 + b_2
$$

以及

$$
e = e_1 + e_2 .
$$

```{exercise-end}
```

## 纯交换经济

让我们进一步探讨一个有 $n$ 种商品和 $m$ 个人的纯交换经济。

### 竞争均衡

我们将计算一个竞争均衡。

为了计算纯交换经济的竞争均衡，我们使用以下事实：

- 竞争均衡中的相对价格与在特定单一人或代表性消费者经济中的相对价格相同，其中偏好为 $\Pi$，$b=\sum_i b_i$，并且禀赋 $e = \sum_i e_{i}$。

我们可以使用以下步骤来计算竞争均衡：

- 首先我们通过规范化 $\mu = 1$ 来解决单一代表性消费者经济问题。然后，我们使用第一个消费品作为计价单位来重新规范化价格向量。

- 接下来我们使用竞争均衡价格来计算每个消费者的财富边际效用：

$$
\mu_{i}=\frac{-W_{i}+p^{T}\left(\Pi^{-1}b_{i}-e_{i}\right)}{p^{T}(\Pi^{T}\Pi)^{-1}p}
$$

- 最后我们使用需求曲线来计算竞争均衡分配：

$$
c_{i}=\Pi^{-1}b_{i}-(\Pi^{T}\Pi)^{-1}\mu_{i}p
$$

### 设计一些 Python 代码

以下我们将构建一个 Python 类，具有以下属性：

 * **偏好** 以以下形式：
   
   * 一个 $n \times n$ 的正定矩阵 $\Pi$
   * 一个 $n \times 1$ 的幸福点向量 $b$

 * **禀赋** 以以下形式：
  
   * 一个 $n \times 1$ 的向量 $e$
   * 一个默认值为 $0$ 的标量 "财富" $W$

这个类将包括一个测试，确保 $b \gg \Pi e $，如果违反了这一规则（在我们需要指定的某一阈值水平上），则会引发一个异常。

 * **一个人** 以以下形式：

    * **偏好** 和 **禀赋**

 * **纯交换经济** 将包括：

    * $m$ 个 **人** 的集合

       * $m=1$ 对于我们的单代理经济
       * $m=2$ 对于我们描绘纯交换经济的示例

    * 一个平衡价格向量 $p$（以某种方式标准化）
    * 一个平衡配置 $c_1, c_2, \ldots, c_m$ -- $m$ 个维度为 $n \times 1$ 的向量的集合

现在让我们开始编码。

```{code-cell} ipython3
class ExchangeEconomy:
    def __init__(self, 
                 Π, 
                 bs, 
                 es, 
                 Ws=None, 
                 thres=1.5):
        """
        为交换经济设置环境

        参数:
            Π (np.array): 共用的替代矩阵
            bs (list): 所有消费者的幸福点
            es (list): 所有消费者的初始物资
            Ws (list): 所有消费者的财富
            thres (float): 设定一个阈值来测试 b >> Pi e 是否被违反
        """
        n, m = Π.shape[0], len(bs)

        # 检查不饱和条件
        for b, e in zip(bs, es):
            if np.min(b / np.max(Π @ e)) <= thres:
                raise Exception('设置更远的幸福点')

        if Ws == None:
            Ws = np.zeros(m)
        else:
            if sum(Ws) != 0:
                raise Exception('无效的财富分配')

        self.Π, self.bs, self.es, self.Ws, self.n, self.m = Π, bs, es, Ws, n, m

    def competitive_equilibrium(self):
        """
        计算竞争均衡的价格和分配
        """
        Π, bs, es, Ws = self.Π, self.bs, self.es, self.Ws
        n, m = self.n, self.m
        slope_dc = inv(Π.T @ Π)
        Π_inv = inv(Π)

        # 合计
        b = sum(bs)
        e = sum(es)

        # 以 mu=1 计算价格向量并重新标准化
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
                raise Exception('负分配：不存在平衡')

        return p, c_s, μ_s
```

## 实施

接下来我们使用上面定义的类 ``ExchangeEconomy`` 来研究

* 一个没有生产的两人经济体,
* 一个动态经济体, 和
* 一个有风险和箭头证券的经济体。

### 没有生产的两人经济体

在这里我们研究竞争均衡 $p, c_1, c_2$ 如何响应不同的 $b_i$ 和 $e_i$, $i \in \{1, 2\}$。

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
print('竞争均衡配置:', c_s)
```

如果第一个消费者更喜欢第一种商品，而第二个消费者更喜欢第二种商品会怎样？

```{code-cell} ipython3
EE.bs = [np.array([6, 5]),  # 第一个消费者的幸福点
         np.array([5, 6])]  # 第二个消费者的幸福点

p, c_s, μ_s = EE.competitive_equilibrium()

print('竞争均衡价格向量:', p)
print('竞争均衡配置:', c_s)
```

让第一个消费者变得更穷。

```{code-cell} ipython3
EE.es = [np.array([0.5, 0.5]),  # 第一个消费者的禀赋
         np.array([1, 1])]  # 第二个消费者的禀赋

p, c_s, μ_s = EE.competitive_equilibrium()

print('竞争均衡价格向量:', p)
print('竞争均衡配置:', c_s)
```

现在让我们构建一个自给自足（即无贸易）均衡。

```{code-cell} ipython3
EE.bs = [np.array([4, 6]),  # 第一个消费者的幸福点
      np.array([6, 4])]  # 第二个消费者的幸福点

EE.es = [np.array([0, 2]),  # 第一个消费者的禀赋
      np.array([2, 0])]  # 第二个消费者的禀赋

p, c_s, μ_s = EE.competitive_equilibrium()

print('竞争均衡价格向量:', p)
print('竞争均衡配置:', c_s)
```

现在让我们在贸易之前重新分配禀赋。

```{code-cell} ipython3
bs = [np.array([5, 5]),  # 第一个消费者的幸福点
      np.array([5, 5])]  # 第二个消费者的幸福点

es = [np.array([1, 1]),  # 第一个消费者的禀赋
      np.array([1, 1])]  # 第二个消费者的禀赋

Ws = [0.5, -0.5]
EE_new = ExchangeEconomy(Π, bs, es, Ws)
p, c_s, μ_s = EE_new.competitive_equilibrium()

print('竞争均衡价格向量:', p)
print('竞争均衡配置:', c_s)
```

### 动态经济

现在我们利用上述技巧来研究一个动态经济，这里涉及两个时期。

```{code-cell} ipython3
beta = 0.95

Π = np.array([[1, 0],
              [0, np.sqrt(beta)]])

bs = [np.array([5, np.sqrt(beta) * 5])]

es = [np.array([1, 1])]

EE_DE = ExchangeEconomy(Π, bs, es)
p, c_s, μ_s = EE_DE.competitive_equilibrium()

print('竞争均衡的价格向量:', p)
print('竞争均衡配置:', c_s)
```

### 带箭头证券的风险经济

我们使用上述技巧来解释 $c_1, c_2$ 为“箭头证券”，这些证券是对消费品的状态或情况下的索赔。

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
print('竞争均衡配置:', c_s)
```

## 推导代表性消费者

在我们这里研究的多消费者经济类中，事实上存在一个单一的**代表性消费者**，其偏好和禀赋可以从单独个体消费者的偏好和禀赋列表中推断出来。

考虑一个具有初始财富分配 $W_i$ 的多消费者经济，满足 $\sum_i W_{i}=0$

我们允许初始财富的重新分配。

我们有以下对象


- 需求曲线：
  
$$ 
c_{i}=\Pi^{-1}b_{i}-(\Pi^T\Pi)^{-1}\mu_{i}p 
$$

- 财富的边际效用：
  
$$ 
\mu_{i}=\frac{-W_{i}+p^T\left(\Pi^{-1}b_{i}-e_{i}\right)}{p^T(\Pi^T\Pi)^{-1}p}
$$

- 市场清算：
  
$$ 
\sum c_{i}=\sum e_{i}
$$

表示总消费 $\sum_i c_{i}=c$ 和 $\sum_i \mu_i = \mu$.

市场清算需要

$$ 
\Pi^{-1}\left(\sum_{i}b_{i}\right)-(\Pi^{T}\Pi)^{-1}p\left(\sum_{i}\mu_{i}\right)=\sum_{i}e_{i}
$$

这经过几步计算后得到

$$
p=\mu^{-1}\left(\Pi^{T}b-\Pi^{T}\Pi e\right)
$$

其中

$$ 
\mu = \sum_i\mu_{i}=\frac{0 + p^{T}\left(\Pi^{-1}b-e\right)}{p^{T}(\Pi^{T}\Pi)^{-1}p}.
$$

现在考虑上述指定的代表性消费者经济。

用 $\tilde{\mu}$ 表示代表性消费者的财富边际效用。

需求函数是

$$
c=\Pi^{-1}b-(\Pi^{T}\Pi)^{-1}\tilde{\mu} p
$$

将这个代入预算约束得到

$$
\tilde{\mu}=\frac{p^{T}\left(\Pi^{-1}b-e\right)}{p^{T}(\Pi^{T}\Pi)^{-1}p}
$$

在平衡状态下 $c=e$，所以

$$
p=\tilde{\mu}^{-1}(\Pi^{T}b-\Pi^{T}\Pi e)
$$

因此，我们已经验证了，直到选择一个基准物来表达绝对价格为止，我们代表性消费者经济中的价格向量与具有多个消费者的基本经济中的价格向量相同。