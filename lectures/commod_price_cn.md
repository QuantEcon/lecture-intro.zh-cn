---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(commod_price)=
# 商品价格

## 概述

在全球超过一半的国家中，[商品](https://en.wikipedia.org/wiki/Commodity)占[总出口的主要部分](https://unctad.org/publication/commodities-and-development-report-2019)。

商品的例子包括铜、钻石、铁矿石、锂、棉花和咖啡豆。

在本讲座中，我们将介绍商品价格理论。

与本系列中的其他讲座相比，本讲座相当高级。

我们需要计算一个均衡，而该均衡由一个价格函数描述。

我们将求解一个方程，其中价格函数是未知的。

这比求解一个未知数或向量的方程要难。

本讲座将讨论求解一个未知函数的「函数方程」的一种方法。

对于本讲座，我们需要 `yfinance` 库。

```{code-cell} ipython3
:tags: [hide-output]
!pip install yfinance
```
```

```




# 商品价格

## 概述

在全球超过一半的国家中，[商品](https://en.wikipedia.org/wiki/Commodity)占[总出口的主要部分](https://unctad.org/publication/commodities-and-development-report-2019)。

商品的例子包括铜、钻石、铁矿石、锂、棉花和咖啡豆。

在本讲座中，我们将介绍商品价格理论。

与本系列中的其他讲座相比，本讲座相当高级。

我们需要计算一个均衡，而该均衡由一个价格函数描述。

我们将求解一个方程，其中价格函数是未知的。

这比求解一个未知数或向量的方程要难。

本讲座将讨论求解一个未知函数的「函数方程」的一种方法。

对于本讲座，我们需要 `yfinance` 库。

```{code-cell} ipython3
:tags: [hide-output]
!pip install yfinance
```
```

## 数据

下图显示了自2016年初以来棉花的价格（以美元计）。

```{code-cell} ipython3
:tags: [hide-input, hide-output]

s = yf.download('CT=F', '2016-1-1', '2023-4-1')['Adj Close']
```

```{code-cell} ipython3
:tags: [hide-input]

fig, ax = plt.subplots()

ax.plot(s, marker='o', alpha=0.5, ms=1)
ax.set_ylabel('价格', fontsize=12)
ax.set_xlabel('日期', fontsize=12)

plt.show()
```

## 棉花价格的波动

上图显示了棉花价格的惊人波动。

是什么导致了这些波动？

一般来说，价格取决于以下各方的选择和行为：

1. 供应商，
2. 消费者，以及
3. 投机者。

我们的重点将是这些各方之间的互动。

我们将在一个动态的供需模型中把它们连接起来，这个模型被称为
*竞争性储存模型*。

这个模型是由
{cite}`samuelson1971stochastic`,
{cite}`wright1982economic`, {cite}`scheinkman1983simple`,
{cite}`deaton1992on`, {cite}`deaton1996competitive` 和
{cite}`chambers1996theory` 开发的。


## 竞争性储存模型

在竞争性储存模型中，商品是资产，

1. 可以被投机者交易，并且
1. 对消费者有内在价值。

总需求是消费者需求和投机者需求的总和。

供应是外生的，取决于“收成”。

```{note}
现如今，基本的计算机芯片和集成电路等商品在金融市场上往往被视作商品，
因为它们高度标准化。对于这类商品，"收成"这个词不太适用。

尽管如此，为了简化问题，我们仍使用这个词。
```

均衡价格通过竞争确定。

它是当前状态的函数（该状态决定了当前的收成并预测未来的收成）。

## 模型

考虑一个单一商品的市场，其价格在$t$时为$p_t$。

该商品在$t$时的收成为$Z_t$。

我们假设序列$\{ Z_t \}_{t \geq 1}$是具有共同密度函数$\phi$的独立同分布（IID）。

投机者可以在各期之间储存商品，本期购买的$I_t$单位商品在下期可产出$\alpha I_t$单位。

这里$\alpha \in (0,1)$是该商品的折旧率。

为了简单起见，我们假设无风险利率为零，所以购买$I_t$单位的预期利润是

$$
  \mathbb{E}_t \, p_{t+1} \cdot \alpha I_t - p_t I_t
   = (\alpha \mathbb{E}_t \, p_{t+1} - p_t) I_t
$$

这里$\mathbb{E}_t \, p_{t+1}$是时间$t$时对$p_{t+1}$的预期。

## 均衡

在本节中，我们定义均衡并讨论如何计算它。

### 均衡条件

假设投机者是风险中性的，这意味着他们在预期利润为正时购买商品。

因此，如果预期利润为正，那么市场不在均衡状态。

因此，要达成均衡，价格必须满足“无套利”条件

$$
  \alpha \mathbb{E}_t \, p_{t+1}  - p_t \leq 0
$$ (eq:arbi)


利润最大化给出了额外的条件

$$
  \alpha \mathbb{E}_t \, p_{t+1}  - p_t  < 0 \text{ implies } I_t = 0
$$ (eq:pmco)


我们还要求市场在每个周期都能清算。

我们假设消费者根据价格 $p$ 产生需求量 $D(p)$。

令 $P := D^{-1}$ 为逆需求函数。

关于数量，

* 供应是投机者的结转量和当前收成的总和
```

* 需求是消费者购买量和投机者购买量的总和。

在数学上，

* 供应 $ = X_t = \alpha I_{t-1} + Z_t$，其取值范围为 $S := \mathbb R_+$，
* 需求 $ = D(p_t) + I_t$

因此，市场均衡条件为

$$
  \alpha I_{t-1} + Z_t =  D(p_t) + I_t
$$ (eq:mkeq)


初始条件 $X_0 \in S$ 被视为已知。



### 均衡函数

如何找到一个均衡？

我们的攻击路径是寻找一个仅依赖于当前状态的价格系统。

换句话说，我们在 $S$ 上取一个函数 $p$ 并设置每个 $t$ 的 $p_t = p(X_t)$。

价格和数量满足

$$
  p_t = p(X_t), \quad I_t = X_t - D(p_t), \quad X_{t+1} = \alpha I_t + Z_{t+1}
$$ (eq:eosy)


我们选择 $p$ 使得这些价格和数量满足上述均衡条件。

更确切地说，我们寻找一个 $p$ 使得 [](eq:arbi) 和 [](eq:pmco) 对应的系统 [](eq:eosy) 成立。

为此，假设存在一个函数 $p^*$ 在 $S$ 上满足

$$
  p^*(x) = \max
    \left\{
    \alpha \int_0^\∞ p^*(\alpha I(x) + z) \phi(z)dz, P(x)
    \right\}
    \qquad (x \in S)
$$ (eq:dopf)

其中

$$
  I(x) := x - D(p^*(x))
    \qquad (x \in S)
$$ (eq:einvf)

事实证明，这样一个 $p^*$ 就足够了，在上述意义上 [](eq:arbi) 和 [](eq:pmco) 对应的系统 [](eq:eosy) 成立。

为了观察这一点，首先注意到

$$
  \mathbb{E}_t \, p_{t+1}
   = \mathbb{E}_t \, p^*(X_{t+1})
   = \mathbb{E}_t \, p^*(\alpha I(X_t) + Z_{t+1})
   = \int_0^\∞ p^*(\alpha I(X_t) + z) \phi(z)dz
$$

因此 [](eq:arbi) 要求

$$
   \alpha \int_0^\∞ p^*(\alpha I(X_t) + z) \phi(z)dz \leq p^*(X_t)
$$

这个不等式可以直接从 [](eq:dopf) 推导出来。

其次，关于 [](eq:pmco)，假设

$$
   \alpha \int_0^\∞ p^*(\alpha I(X_t) + z) \phi(z)dz < p^*(X_t)
$$

那么根据 [](eq:dopf) 我们有 $p^*(X_t) = P(X_t)$

此时 $D(p^*(X_t)) = X_t$ 和 $I_t = I(X_t) = 0$。

因此，[](eq:arbi) 和 [](eq:pmco) 都成立。

我们已经找到了一个均衡。


### 计算均衡

我们现在知道可以通过找到一个满足 [](eq:dopf) 的函数 $p^*$ 来获得均衡。

可以证明，在温和条件下，$S$ 上恰好有一个满足 [](eq:dopf) 的函数。

此外，我们可以使用连续逼近法计算这个函数。

这意味着我们从一个函数猜测开始，然后利用 [](eq:dopf) 更新这个函数。

这会生成一个函数序列 $p_1, p_2, \ldots$

我们继续这一过程，直到 $p_k$ 和 $p_{k+1}$ 非常接近。

然后我们将最终计算得到的 $p_k$ 作为 $p^*$ 的近似值。

为了实现我们的更新步骤，将 [](eq:dopf) 和 [](eq:einvf) 结合起来是有帮助的。

这导致了更新规则

$$
  p_{k+1}(x) = \max
    \left\{
    \alpha \int_0^\∞ p_k(\alpha ( x - D(p_{k+1}(x))) + z) \phi(z)dz, P(x)
    \right\}
$$ (eq:dopf2)
```

所以说，我们假定$p_k$为已知，在每个$x$处求解$q$，如

$$
  q = \max
    \left\{
    \alpha \int_0^\infty p_k(\alpha ( x - D(q)) + z) \phi(z)dz, P(x)
    \right\}
$$ (eq:dopf3)

实际上，我们不能在每个$x$处进行计算，所以我们在点$x_1, \ldots, x_n$的网格上进行计算。

然后我们得到对应的值$q_1, \ldots, q_n$。

然后我们计算$p_{k+1}$，即在网格$x_1, \ldots, x_n$上的$q_1, \ldots, q_n$值的线性插值。

然后我们重复这一过程，直到收敛。

## 代码

下面的代码实现了这个迭代过程，从$p_0 = P$开始。

分布 $\phi$ 被设置为一个移位的 Beta 分布（虽然也可以选择其他多种分布）。

在 [](eq:dopf3) 中的积分通过蒙特卡洛方法计算。

```{code-cell} ipython3
α, a, c = 0.8, 1.0, 2.0
beta_a, beta_b = 5, 5
mc_draw_size = 250
gridsize = 150
grid_max = 35
grid = np.linspace(a, grid_max, gridsize)

beta_dist = beta(5, 5)
Z = a + beta_dist.rvs(mc_draw_size) * c    # Shock observations
D = P = lambda x: 1.0 / x
tol = 1e-4

def T(p_array):

    new_p = np.empty_like(p_array)

    # Interpolate to obtain p as a function.
    p = interp1d(grid,
                 p_array,
                 fill_value=(p_array[0], p_array[-1]),
                 bounds_error=False)

    # Update
    for i, x in enumerate(grid):

        h = lambda q: q - max(α * np.mean(p(α * (x - D(q)) + Z)), P(x))
        new_p[i] = brentq(h, 1e-8, 100)

    return new_p


fig, ax = plt.subplots()

price = P(grid)
ax.plot(grid, price, alpha=0.5, lw=1, label="inverse demand curve")
error = tol + 1
while error > tol:
    new_price = T(price)
    error = max(np.abs(new_price - price))
    price = new_price

ax.plot(grid, price, 'k-', alpha=0.5, lw=2, label=r'$p^*$')
ax.legend()
ax.set_xlabel('$x$', fontsize=12)

plt.show()
```

上图显示了逆需求曲线$P$，也是$p_0$，以及我们对$p^*$的近似。

一旦我们有了$p^*$的近似，我们就可以模拟一个价格时间序列。

```{code-cell} ipython3
# Turn the price array into a price function
p_star = interp1d(grid,
                  price,
                  fill_value=(price[0], price[-1]),
                  bounds_error=False)

def carry_over(x):
    return α * (x - D(p_star(x)))

def generate_cp_ts(init=1, n=50):
    X = np.empty(n)
    X[0] = init
    for t in range(n-1):
            Z = a + c * beta_dist.rvs()
            X[t+1] = carry_over(X[t]) + Z
    return p_star(X)

fig, ax = plt.subplots()
ax.plot(generate_cp_ts(), label="price")
ax.set_xlabel("time")
ax.legend()
plt.show()
```