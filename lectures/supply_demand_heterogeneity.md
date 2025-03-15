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
# 具有异质性的市场均衡

## 概览

在{doc}`前一讲座<supply_demand_multiple_goods>`中，我们研究了包含多种商品的经济体系中的竞争均衡。

虽然这种分析很有启发性，但我们做了一个很强的简化假设 - 所有经济主体都是完全相同的。

然而在现实世界中，家庭、企业和其他经济主体在很多方面都存在差异。

本讲座将通过引入消费者偏好和禀赋的差异性，来探讨这种异质性。

我们将分析在这种更现实的设定下，竞争均衡是如何形成的。

我们还会展示如何构建一个"代表性消费者"来简化分析。

让我们先导入需要用到的包：

```{code-cell} ipython3
import numpy as np
from scipy.linalg import inv
```

## 一个简单的例子

让我们研究一个没有生产的**纯交换**经济体的简单例子。

现在有两个消费者，他们的禀赋向量 $e_i$ 和餍足点向量 $b_i$ 是不同的，其中 $i=1,2$。

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

我们可以使 $\mu_1 + \mu_2 =1$ 来对价格进行归一化处理，然后解方程

$$
\mu_i(p,e) = \frac{p^{\top} (\Pi^{-1} b_i - e_i)}{p^{\top} (\Pi^{\top} \Pi)^{-1} p}
$$ (eq:old7)

求得 $\mu_i, 其中 i = 1,2$。

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

## 纯交换经济体

让我们进一步探讨一个有 $n$ 种商品和 $m$ 个人的纯交换经济体。

### 竞争均衡

我们会来计算一个竞争均衡。

想要计算纯交换经济体的竞争均衡，我们会运用以下事实：

- 竞争均衡中的相对价格与特殊单人经济体或代表性消费者经济体中的相对价格相同，其中偏好为 $\Pi$，$b=\sum_i b_i$，并且禀赋为 $e = \sum_i e_{i}$。

我们可以运用以下步骤来计算竞争均衡：

- 首先，我们通过归一化 $\mu = 1$ 来解决单一代表性消费者的经济模型。然后，我们使用第一种消费品作为计价单位，重新对价格向量进行归一化处理。

- 接下来我们使用竞争均衡价格来计算每个消费者的财富边际效用：

$$
\mu_{i}=\frac{-W_{i}+p^{T}\left(\Pi^{-1}b_{i}-e_{i}\right)}{p^{T}(\Pi^{T}\Pi)^{-1}p}
$$

- 最后我们利用需求曲线来计算竞争均衡分配：

$$
c_{i}=\Pi^{-1}b_{i}-(\Pi^{T}\Pi)^{-1}\mu_{i}p
$$

### 编写一些 Python 代码

下面我们会构建一个 Python 类，它具有以下属性：

 * **偏好** 的形式包括：
   
   * 一个 $n \times n$ 的正定矩阵 $\Pi$
   * 一个 $n \times 1$ 的餍足点向量 $b$

 * **禀赋** 的形式包括：
  
   * 一个 $n \times 1$ 的向量 $e$
   * 一个默认值为 $0$ 的标量 "财富" $W$

这个类会检查每个消费者的餍足点是否充分大于其禀赋的转换值(即 $b \gg \Pi e$)。如果不满足这个条件，类会抛出异常。

类的结构如下:

* **个体消费者** 由以下要素刻画:
    * 偏好参数(包括替代矩阵 $\Pi$ 和餍足点 $b$)
    * 初始禀赋 $e$ 和财富 $W$

 * **纯交换经济体** 包括：

    * $m$ 个 **人** 的集合

       * $m=1$ 指的是单主体经济体
       * $m=2$ 指的是我们所描绘的纯交换经济体的示例

    * 一个均衡价格向量 $p$（以某种方式归一化）
    * 一个均衡分配 $c_1, c_2, \ldots, c_m$ -- $m$ 个维度为 $n \times 1$ 的向量的集合

现在让我们开始编程。

```{code-cell} ipython3
class ExchangeEconomy:
    def __init__(self, 
                 Π, 
                 bs, 
                 es, 
                 Ws=None, 
                 thres=1.5):
        """
        为交换经济体设置环境

        参数:
            Π (np.array): 共用替代矩阵
            bs (list): 所有消费者的餍足点
            es (list): 所有消费者的禀赋
            Ws (list): 所有消费者的财富
            thres (float): 设定一个阈值来测试是否违反了 b >> Pi e 
        """
        n, m = Π.shape[0], len(bs)

        # 检查不饱和条件
        for b, e in zip(bs, es):
            if np.min(b / np.max(Π @ e)) <= thres:
                raise Exception('设置更远的餍足点')

        if Ws == None:
            Ws = np.zeros(m)
        else:
            if sum(Ws) != 0:
                raise Exception('无效的财富分配')

        self.Π, self.bs, self.es, self.Ws, self.n, self.m = Π, bs, es, Ws, n, m

    def competitive_equilibrium(self):
        """
        计算竞争均衡价格和竞争均衡分配
        """
        Π, bs, es, Ws = self.Π, self.bs, self.es, self.Ws
        n, m = self.n, self.m
        slope_dc = inv(Π.T @ Π)
        Π_inv = inv(Π)

        # 加总
        b = sum(bs)
        e = sum(es)

        # 用 mu=1 来计算价格向量并重新归一化
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
                raise Exception('负分配：均衡不存在')

        return p, c_s, μ_s
```

## 实践

接下来我们使用上面定义的类 ``ExchangeEconomy`` 来研究

* 一个没有生产的两人经济体,
* 一个动态经济体, 和
* 一个有风险和阿罗证券的经济体。

### 没有生产的两人经济体

在这里我们研究竞争均衡 $p, c_1, c_2$ 如何对不同的 $b_i$ 和 $e_i$, $i \in \{1, 2\}$ 作出反应。

```{code-cell} ipython3
Π = np.array([[1, 0],
              [0, 1]])

bs = [np.array([5, 5]),  # 第一个消费者的餍足点
      np.array([5, 5])]  # 第二个消费者的餍足点

es = [np.array([0, 2]),  # 第一个消费者的禀赋
      np.array([2, 0])]  # 第二个消费者的禀赋

EE = ExchangeEconomy(Π, bs, es)
p, c_s, μ_s = EE.competitive_equilibrium()

print('竞争均衡价格向量:', p)
print('竞争均衡分配:', c_s)
```

如果第一个消费者更喜欢第一种商品，而第二个消费者更喜欢第二种商品会怎样？

```{code-cell} ipython3
EE.bs = [np.array([6, 5]),  # 第一个消费者的餍足点
         np.array([5, 6])]  # 第二个消费者的餍足点

p, c_s, μ_s = EE.competitive_equilibrium()

print('竞争均衡价格向量:', p)
print('竞争均衡分配:', c_s)
```

我们让第一个消费者变得稍微更穷。

```{code-cell} ipython3
EE.es = [np.array([0.5, 0.5]),  # 第一个消费者的禀赋
         np.array([1, 1])]  # 第二个消费者的禀赋

p, c_s, μ_s = EE.competitive_equilibrium()

print('竞争均衡价格向量:', p)
print('竞争均衡分配:', c_s)
```

我们现在构建一个自给自足的（即无贸易的）均衡模型。

```{code-cell} ipython3
EE.bs = [np.array([4, 6]),  # 第一个消费者的餍足点
      np.array([6, 4])]  # 第二个消费者的餍足点

EE.es = [np.array([0, 2]),  # 第一个消费者的禀赋
      np.array([2, 0])]  # 第二个消费者的禀赋

p, c_s, μ_s = EE.competitive_equilibrium()

print('竞争均衡价格向量:', p)
print('竞争均衡分配:', c_s)
```

在进行贸易之前重新分配禀赋。

```{code-cell} ipython3
bs = [np.array([5, 5]),  # 第一个消费者的餍足点
      np.array([5, 5])]  # 第二个消费者的餍足点

es = [np.array([1, 1]),  # 第一个消费者的禀赋
      np.array([1, 1])]  # 第二个消费者的禀赋

Ws = [0.5, -0.5]
EE_new = ExchangeEconomy(Π, bs, es, Ws)
p, c_s, μ_s = EE_new.competitive_equilibrium()

print('竞争均衡价格向量:', p)
print('竞争均衡分配:', c_s)
```

### 动态经济体

现在我们利用上述方法来研究一个动态经济体，其中涉及两个时期。

```{code-cell} ipython3
beta = 0.95

Π = np.array([[1, 0],
              [0, np.sqrt(beta)]])

bs = [np.array([5, np.sqrt(beta) * 5])]

es = [np.array([1, 1])]

EE_DE = ExchangeEconomy(Π, bs, es)
p, c_s, μ_s = EE_DE.competitive_equilibrium()

print('竞争均衡的价格向量:', p)
print('竞争均衡分配:', c_s)
```

### 具有阿罗证券的风险经济体

我们通过上文所述的方法，将 $c_1, c_2$ 解释为“阿罗证券”，这些证券是对状态依赖型商品的索取权。

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

## 推导代表性消费者

在我们研究的这类多消费者经济体中，事实上存在一个单一的**代表性消费者**，其偏好和禀赋可以从众多单独的个体消费者的偏好和禀赋中推导出来。

考虑一个具有初始财富分配 $W_i$ 的多消费者经济体，其中初始财富满足 $\sum_i W_{i}=0$

我们假设初始财富可以重新分配。

我们有以下对象


- 需求曲线：
  
$$ 
c_{i}=\Pi^{-1}b_{i}-(\Pi^T\Pi)^{-1}\mu_{i}p 
$$

- 财富的边际效用：
  
$$ 
\mu_{i}=\frac{-W_{i}+p^T\left(\Pi^{-1}b_{i}-e_{i}\right)}{p^T(\Pi^T\Pi)^{-1}p}
$$

- 市场出清：
  
$$ 
\sum c_{i}=\sum e_{i}
$$

表示总消费 $\sum_i c_{i}=c$ 和 $\sum_i \mu_i = \mu$。

市场出清需要

$$ 
\Pi^{-1}\left(\sum_{i}b_{i}\right)-(\Pi^{T}\Pi)^{-1}p\left(\sum_{i}\mu_{i}\right)=\sum_{i}e_{i}
$$

经过几步计算后得到

$$
p=\mu^{-1}\left(\Pi^{T}b-\Pi^{T}\Pi e\right)
$$

其中

$$ 
\mu = \sum_i\mu_{i}=\frac{0 + p^{T}\left(\Pi^{-1}b-e\right)}{p^{T}(\Pi^{T}\Pi)^{-1}p}.
$$

现在考虑上述的代表性消费者经济体。

用 $\tilde{\mu}$ 表示代表性消费者的财富边际效用。

需求函数是

$$
c=\Pi^{-1}b-(\Pi^{T}\Pi)^{-1}\tilde{\mu} p
$$

将这个代入预算约束得到

$$
\tilde{\mu}=\frac{p^{T}\left(\Pi^{-1}b-e\right)}{p^{T}(\Pi^{T}\Pi)^{-1}p}
$$

在均衡状态下 $c=e$，所以

$$
p=\tilde{\mu}^{-1}(\Pi^{T}b-\Pi^{T}\Pi e)
$$

因此，我们证明了：在选定一个计价单位来表达绝对价格后，代表性消费者经济体中的价格向量与具有多个消费者的基础经济体中的价格向量相同。
