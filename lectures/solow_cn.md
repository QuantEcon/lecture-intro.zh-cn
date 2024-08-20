---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(solow)=
# 索洛-斯旺增长模型

在本讲中，我们回顾一个由[罗伯特·索洛 (1925--2023)](https://en.wikipedia.org/wiki/Robert_Solow) 和 [特雷弗·斯旺 (1918--1989)](https://en.wikipedia.org/wiki/Trevor_Swan)提出的著名模型。

该模型用于研究长期的经济增长。

尽管模型很简单，但它包含了一些有趣的教训。


我们将使用以下导入。

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
```
图像输入功能：已启用

## 模型

在一个索洛-斯旺经济中，代理人会保存他们当前收入的一部分。

储蓄维持或增加资本存量。

资本与劳动结合来生产产出，而产出又会支付给工人和资本所有者。

为了简化，我们忽略人口和生产率增长。

对于每个整数 $t \geq 0$，时期 $t$ 的产出 $Y_t$ 表示为 $Y_t = F(K_t, L_t)$，其中 $K_t$ 是资本，$L_t$ 是劳动，而 $F$ 是一个总生产函数。

假设函数 $F$ 是非负和**线性齐次的**，即

$$
    F(\lambda K, \lambda L) = \lambda F(K, L)
    \quad \text{对所有 } \lambda \geq 0
$$

具有这一特性的生产函数包括

* **柯布-道格拉斯**函数 $F(K, L) = A K^{\alpha} L^{1-\alpha}$，且 $0 \leq \alpha \leq 1$
* **CES** 函数 $F(K, L) = \left\{ a K^\rho + b L^\rho \right\}^{1/\rho}$，其中 $a, b, \rho > 0$

我们假设一个封闭的经济体，因此总国内投资等于总国内储蓄。

储蓄率是常数 $s$，满足 $0 \leq s \leq 1$，因此总投资和储蓄都等于 $s Y_t$。

资本会折旧：如果不通过投资补充，今天的一单位资本会变成明天的 $1-\delta$ 单位。

因此，

$$
    K_{t+1} = s F(K_t, L_t) + (1 - \delta) K_t
$$

没有人口增长，$L_t$ 等于某个常数 $L$。

设定 $k_t := K_t / L$ 并使用一阶齐次性现在产生

$$
    k_{t+1}
    = s \frac{F(K_t, L)}{L} + (1 - \delta) k_t
    = s F(k_t, 1) + (1 - \delta) k_t
$$

对于 $f(k) := F(k, 1)$，资本动态的最终表达式是

```{math}
:label: solow
    k_{t+1} = g(k_t)
    \text{ 其中 } g(k) := s f(k) + (1 - \delta) k
```

我们的目标是了解在给定外生初始资本存量 $k_0$ 的情况下，$k_t$ 随时间的演变。

## 图形视角

为了理解序列 $(k_t)_{t \geq 0}$ 的动态，我们使用一个 45 度图。

为此，我们首先需要指定函数形式 $f$ 并为参数赋值。

我们选择柯布-道格拉斯规格 $f(k) = 上述模型的代码和图 及其实现如下：A k^\alpha$，并设定 $A=2.0$，$\alpha=0.3$，$s=0.3$ 和 $\delta=0.4$。

然后绘制来自 {eq}`solow` 的函数 $g$，以及 45 度线。

让我们定义常数。

```{code-cell} ipython3
A, s, alpha, delta = 2, 0.3, 0.3, 0.4
x0 = 0.25
xmin, xmax = 0, 3
```

现在，我们定义函数 $g$。

```{code-cell} ipython3
def g(A, s, alpha, delta, k):
    return A * s * k**alpha + (1 - delta) * k
```

我们绘制 $g$ 和 45 度线 $h(k) = k$。

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 6))
k = np.linspace(xmin, xmax, 100)
ax.plot(k, g(A, s, alpha, delta, k), label='$g(k)$', color='blue')
ax.plot(k, k, label='$45$度线', color='red', linestyle='--')
ax.legend()
ax.grid(True)
ax.set_xlabel('$k$')
ax.set_ylabel('$g(k)$')
plt.show()
```

让我们绘制 $g$ 的 45 度图。

```{code-cell} ipython3
def plot45(kstar=None):
    xgrid = np.linspace(xmin, xmax, 12000)

    fig, ax = plt.subplots()

    ax.set_xlim(xmin, xmax)

    g_values = g(A, s, alpha, delta, xgrid)

    ymin, ymax = np.min(g_values), np.max(g_values)
    ax.set_ylim(ymin, ymax)

    lb = r'$g(k) = sAk^{\alpha} + (1 - \delta)k$'
    ax.plot(xgrid, g_values,  lw=2, alpha=0.6, label=lb)
    ax.plot(xgrid, xgrid, 'k-', lw=1, alpha=0.7, label='45')

    if kstar:
        fps = (kstar,)

        ax.plot(fps, fps, 'go', ms=10, alpha=0.6)

        ax.annotate(r'$k^* = (sA / \delta)^{(1/(1-\alpha))}$',
                 xy=(kstar, kstar),
                 xycoords='data',
                 xytext=(-40, -60),
                 textcoords='offset points',
                 fontsize=14,
                 arrowprops=dict(arrowstyle="->"))

    ax.legend(loc='upper left', frameon=False, fontsize=12)

    ax.set_xticks((0, 1, 2, 3))
    ax.set_yticks((0, 1, 2, 3))

    ax.set_xlabel('$k_t$', fontsize=12)
    ax.set_ylabel('$k_{t+1}$', fontsize=12)

    plt.show()
```

绘制 $g$ 的 45 度图像。

```{code-cell} ipython3
plot45()
```

假设在某个 $k_t$ 时，$g(k_t)$ 的值严格高于 45 度线。

那么我们会有 $k_{t+1} = g(k_t) > k_t$，且每个工人的资本增加。

如果 $g(k_t) < k_t$，那么每个工人的资本减少。

如果 $g(k_t) = k_t$，那么我们处于**稳态**且 $k_t$ 保持不变。

（该模型的稳态是映射 $g$ 的一个 [不动点](https://en.wikipedia.org/wiki/Fixed_point_(mathematics))。）

从图中函数 $g$ 的形状来看，我们可以看到在 $(0, \infty)$ 中有一个唯一的稳态。

它满足 $k = s Ak^{\alpha} + (1-\delta)k$，因此给定如下

```{math}
:label: kstarss
    k^* := \left( \frac{s A}{\delta} \right)^{1/(1 - \alpha)}
```

如果初始资本低于 $k^*$，那么资本会随时间增加。

如果初始资本高于这个水平，那么情况正好相反。

让我们绘制 45 度图来在图中显示 $k^*$。

```{code-cell} ipython3
kstar = ((s * A) / delta)**(1/(1 - alpha))
plot45(kstar)
```

从我们的图形分析来看，$(k_t)$ 似乎趋向于 $k^*$，无论初始资本 $k_0$ 为何。

这是一种全局稳定性。

下图显示了三条资本时间路径，从三个不同的初始条件出发，在上述参数设置下。

在这个参数设置下，$k^* \approx 1.78$。

让我们定义常数和三个不同的初始条件

```{code-cell} ipython3
A, s, alpha, delta = 2, 0.3, 0.3, 0.4
x0 = np.array([.25, 1.25, 3.25])

ts_length = 20
xmin, xmax = 0, ts_length
ymin, ymax = 0, 3.5
```

现在，我们绘制每条路径。

```{code-cell} ipython3
def simulate_ts(x0_values, ts_length):

    k_star = (s * A / delta)**(1/(1-alpha))
    fig, ax = plt.subplots(figsize=[11, 5])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ts = np.zeros(ts_length)

    # 模拟和绘制时间序列
    for x_init in x0_values:
        ts[0] = x_init
        for t in range(1, ts_length):
            ts[t] = g(A, s, alpha, delta, ts[t-1])
        ax.plot(np.arange(ts_length), ts, '-o', ms=4, alpha=0.6,
                label=r'$k_0=%g$' %x_init)
    ax.plot(np.arange(ts_length), np.full(ts_length,k_star),
            alpha=0.6, color='red', label=r'$k^*$')
    ax.legend(fontsize=10)

    ax.set_xlabel(r'$t$', fontsize=14)
    ax.set_ylabel(r'$k_t$', fontsize=14)

    plt.show()

simulate_ts(x0, ts_length)
```

上面的三条时间路径验证了我们对全局稳定性的预期。

## 总结

综上所述，索洛-斯旺增长模型说明了储蓄对长期经济增长的重要性。 尽管模型的假设较为简化，但它展示了资本积累在决定一个经济体稳定状态下的资本存量和产出水平方面的作用。

As observed, the time paths in the figure all converge to $k^*$.

Next we define the function $g$ for growth in continuous time

```{code-cell} ipython3
def g_con(A, s, alpha, delta, k):
    return A * s * k**alpha - delta * k
```

我们为该模型绘制一个函数图，并标记稳态 $k^*$。

```{code-cell} ipython3
def plot_gcon(kstar=None):

    k_grid = np.linspace(0, 2.8, 10000)  # 生成 k 的离散点

    fig, ax = plt.subplots(figsize=[11, 5])
    ax.plot(k_grid, g_con(A, s, alpha, delta, k_grid), label='$g(k)$')  # 绘制 g(k)
    ax.plot(k_grid, 0 * k_grid, label="$k'=0$")  # 绘制零线

    if kstar:
        fps = (kstar,)
        ax.plot(fps, 0, 'go', ms=10, alpha=0.6)  # 绘制稳态点

        ax.annotate(r'$k^* = (sA / \delta)^{(1/(1-\alpha))}$',  # 标注 k^*
                 xy=(kstar, 0),
                 xycoords='data',
                 xytext=(0, 60),
                 textcoords='offset points',
                 fontsize=12,
                 arrowprops=dict(arrowstyle="->"))

    ax.legend(loc='lower left', fontsize=12)

    ax.set_xlabel("$k$",fontsize=10)
    ax.set_ylabel("$k'$", fontsize=10)

    ax.set_xticks((0, 1, 2, 3))
    ax.set_yticks((-0.3, 0, 0.3))

    plt.show()
```

然后我们绘制该图并包括稳态 $k^*$。

```{code-cell} ipython3
kstar = ((s * A) / delta)**(1/(1 - alpha))
plot_gcon(kstar)
```

前面的45度图和时间路径图显示了对于特定的参数值，$k_t$ 会趋向于 $k^*$ 随时间变化。

这从直观上显示了固定参数化的全局稳定性，但如何为一连串可能的参数正式展示同样的结论呢？

在离散时间情况下，很难得到一个简洁的 $k_t$ 表达式。

在连续时间情况下，过程更简单：我们可以得到一个相对简单的 $k_t$ 表达式，指定整个路径。

第一步是设置 $x_t := k_t^{1-\alpha}$，所以 $x'_t = (1-\alpha) k_t^{-\alpha} k'_t$。

替换到 $k'_t = sAk_t^\alpha - \delta k_t$ 得到如下线性微分方程

```{math}
:label: xsolow
    x'_t = (1-\alpha) (sA - \delta x_t)
```

该方程的精确解为

$$
    x_t
    = \left(
        k_0^{1-\alpha} - \frac{sA}{\delta}
      \right)
      \mathrm{e}^{-\delta (1-\alpha) t} +
    \frac{sA}{\delta}
$$

（你可以通过对其关于 $t$ 微分，确认该函数 $x_t$ 满足 {eq}`xsolow`。）

转换回 $k_t$ 得到

```{math}
:label: ssivs
    k_t
    =
    \left[
        \left(
        k_0^{1-\alpha} - \frac{sA}{\delta}
      \right)
      \mathrm{e}^{-\delta (1-\alpha) t} +
    \frac{sA}{\delta}
    \right]^{1/(1-\alpha)}
```

由于 $\delta > 0$ 且 $\alpha \in (0, 1)$，我们立刻看到 $k_t \to k^*$ 当 $t \to \infty$ 独立于 $k_0$。

因此，全局稳定性成立。

## 练习

```{exercise}
:label: solow_ex1

绘制稳态下的人均消费 $c$ 作为储蓄率 $s$ 的函数，其中 $0 \leq s \leq 1$。

使用柯布-道格拉斯规格 $f(k) = A k^\alpha$。

设定 $A=2.0, \alpha=0.3,$ 和 $\delta=0.5$

另外，找到使 $c^*(s)$ 最大化的 $s$ 的近似值并在图中显示出来。

```

```{solution-start} solow_ex1
:class: dropdown
```

储蓄率 $s$ 下的稳态消费由以下公式给出

$$
    c^*(s) = (1-s)f(k^*) = (1-s)A(k^*)^\alpha
$$

```{code-cell} ipython3
A = 2.0
alpha = 0.3
delta = 0.5
```

```{code-cell} ipython3
# 储蓄率的网格
s_grid = np.linspace(0, 1, 1000)

# 稳态资本和消费
k_star = ((s_grid * A) / delta)**(1/(1 - alpha))
c_star = (1 - s_grid) * A * k_star ** alpha

# 找到最大消费点
max_c_star = np.max(c_star)
max_s = s_grid[np.argmax(c_star)]

# 绘制图形
fig, ax = plt.subplots()
ax.plot(s_grid, c_star, label='$c^*(s)$')
ax.axvline(max_s, linestyle='--', color='red', alpha=0.7, label=f'最大 $s$={max_s:.4f}')
ax.legend()

# 设置标签和标题
ax.set_xlabel('$s$')
ax.set_ylabel('$c^*(s)$')
ax.set_title('稳态人均消费 $c^*(s)$ 随储蓄率 $s$ 的变化')

plt.show()
```

图的峰值大约为

```{code-cell} ipython3
max_s
```

```{solution-end}
```

为了找到使 $c^*(s)$ 最大化的 $s$，我们可以使用 `scipy.optimize.minimize_scalar`。我们将使用 $-c^*(s)$，因为 `minimize_scalar` 找到的是最小值。

```{code-cell} ipython3
from scipy.optimize import minimize_scalar
```

我们定义 $c^*(s)$ 并最小化负值

```{code-cell} ipython3
def c_star(s, A, alpha, delta):
    if isinstance(s, np.ndarray):
        k_star = ((s * A) / delta)**(1/(1 - alpha))
    else:
        k_star = ((s * A) / delta)**(1/(1 - alpha))
    return (1 - s) * A * k_star ** alpha

result = minimize_scalar(lambda s: -c_star(s, A, alpha, delta), bounds=(0, 1), method='bounded')
optimal_s = result.x
optimal_s
```

正如我们所见，最大消费发生在 $s \approx 0.17$ 时。。

这与我们从图中观察到的结果是一致的。

让我们最后用最优值标记图形。

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(s_grid, c_star, label='$c^*(s)$')
ax.axvline(optimal_s, linestyle='--', color='green', alpha=0.7, label=f'最优 $s$={optimal_s:.4f}')
ax.legend()

# 设置标签和标题
ax.set_xlabel('$s$')
ax.set_ylabel('$c^*(s)$')
ax.set_title('稳态人均消费 $c^*(s)$ 随储蓄率 $s$ 的变化')

plt.show()
```

```{code-cell} ipython3
x_s_max = np.array([optimal_s, optimal_s])
y_s_max = np.array([0, c_star(s_grid[np.argmax(c_star)], A, alpha, delta)])

fig, ax = plt.subplots(figsize=[11, 5])

# 高亮最大点
ax.plot((optimal_s, ), (c_star(s_grid[np.argmax(c_star)], A, alpha, delta),), 'go', ms=8, alpha=0.6)

ax.annotate(r'$s^*$',
         xy=(optimal_s, c_star(s_grid[np.argmax(c_star)], A, alpha, delta)),
         xycoords='data',
         xytext=(20, -50),
         textcoords='offset points',
         fontsize=12,
         arrowprops=dict(arrowstyle="->"))

# 绘制c_star和垂直虚线
ax.plot(s_grid, c_star, label=r'$c*(s)$')
ax.plot(x_s_max, y_s_max, alpha=0.5, ls='dotted')
ax.set_xlabel(r'$s$')
ax.set_ylabel(r'$c^*(s)$')
ax.legend()

plt.show()
```

另一个方法是通过微分 $c^*(s)$ 并使用 [sympy](https://www.sympy.org/en/index.html) 求解 $\frac{d}{ds}c^*(s)=0$ 尝试数学解决问题。

```{code-cell} ipython3
from sympy import solve, Symbol
```

```{code-cell} ipython3
s_symbol = Symbol('s', real=True)
k = ((s_symbol * A) / delta)**(1/(1 - alpha))
c = (1 - s_symbol) * A * k ** alpha
```

让我们微分 $c$ 并使用 [sympy.solve](https://docs.sympy.org/latest/modules/solvers/solvers.html#sympy.solvers.solvers.solve) 求解

```{code-cell} ipython3
# 使用 sympy 求解
s_star = solve(c.diff())[0]
print(f"s_star = {s_star}")
```

顺便说一句，最大化人均消费稳态水平的储蓄率被称为[黄金规则储蓄率](https://en.wikipedia.org/wiki/Golden_Rule_savings_rate)。

```{solution-end}
```

```{exercise-start}
:label: solow_ex2
```
**随机生产力**

为了使索洛-斯旺模型更接近数据，我们需要考虑处理宏观经济变量的随机波动。

除其他事项外，这将消除人均产出 $y_t = A k^\alpha_t$ 收敛到常数 $y^* := A (k^*)^\alpha$ 的不现实预测。

我们转向离散时间进行以下讨论。

一种方法是将常数生产率替换为某些随机序列 $(A_t)_{t \geq 1}$。

现在的动态变为

```{math}
:label: solowran
    k_{t+1} = s A_{t+1} f(k_t) + (1 - \delta) k_t
```

我们假设 $f$ 是柯布-道格拉斯且 $(A_t)$ 是独立同分布并且对数正态分布。

现在，由于系统在每个时间点都会受到新的冲击，确定性情况下的长期收敛性消失。

考虑 $A=2.0, s=0.6, \alpha=0.3,$ 和 $\delta=0.5$

生成并绘制 $k_t$ 的时间序列。

```{exercise-end}
```

```{solution-start} solow_ex2
:class: dropdown
```

让我们定义对数正态分布的常数和用于模拟的初始值

```{code-cell} ipython3
# 定义常数
sig = 0.2
mu = np.log(2) - sig**2 / 2
A = 2.0
s = 0.6
alpha = 0.3
delta = 0.5
x0 = [.25, 3.25] # 用于模拟的初始值列表
```

让我们定义函数 *k_next* 以找到 $k$ 的下一个值

```{code-cell} ipython3
def lgnorm():
    return np.exp(mu + sig * np.random.randn())

def k_next(s, alpha, delta, k):
    return lgnorm() * s * k**alpha + (1 - delta) * k
```

```{code-cell} ipython3
def ts_plot(x_values, ts_length):
    fig, ax = plt.subplots(figsize=[11, 5])
    ts = np.zeros(ts_length)

    # 模拟和绘制时间序列
    for x_init in x_values:
        ts[0] = x_init
        for t in range(1, ts_length):
            ts[t] = k_next(s, alpha, delta, ts[t-1])
        ax.plot(np.arange(ts_length), ts, '-o', ms=4,
                alpha=0.6, label=r'$k_0=%g$' %x_init)

    ax.legend(loc='best', fontsize=10)

    ax.set_xlabel(r'$t$', fontsize=12)
    ax.set_ylabel(r'$k_t$', fontsize=12)


    plt.show()
```

```{code-cell} ipython3
ts_plot(x0, 50)
```

```{solution-end}
```