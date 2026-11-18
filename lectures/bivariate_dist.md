---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
translation:
  title: 二元分布
  headings:
    Outline: 概述
    Joint distributions: 联合分布
    Joint distributions::Discrete case: 离散情形
    Joint distributions::Continuous case: 连续情形
    Independence: 独立性
    Covariance and correlation: 协方差与相关系数
    How joint distributions arise: 联合分布是如何产生的
    How joint distributions arise::Independent components: 独立的成分
    How joint distributions arise::A common building block: 共同的构建部分
    Back to the normal distribution: 回到正态分布
    Back to the normal distribution::A word of caution: 一个提醒
    Back to the data: 回到数据
    Fitting a bivariate normal by the method of moments: 用矩方法拟合二元正态分布
    'Preview: regression as a conditional mean': 预告：回归作为条件均值
    Exercises: 练习
---

(bivariate_dist)=
# 二元分布

```{index} single: Bivariate Distributions
```

## 概述

{doc}`prob_dist`、{doc}`observed_distributions` 和 {doc}`fitting_distributions` 这几篇讲座都是一次研究一个变量（例如房价的分布）。

但我们通常对不止一个变量感兴趣。

在这种情况下，我们通常希望了解这些变量之间的相互关系。

例如，更大的房子是否往往卖出更高的价格？

在本讲座中，我们将简要介绍**二元分布**：成对随机变量上的概率分布。

我们将涵盖联合分布与边缘分布、独立性、协方差与相关性、联合分布产生的若干方式，以及二元正态分布。

最后我们会简要预告 {doc}`simple_linear_regression`。

我们使用以下的导入：

```{code-cell} ipython3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats
import seaborn as sns

np.set_printoptions(legacy='1.25')   # print scalars as plain numbers
```

为了引出后面的内容，让我们看一下我们在 {doc}`observed_distributions` 和 {doc}`fitting_distributions` {cite}`decock2011ames` 中研究过的艾姆斯（Ames）房屋的销售价格与建筑面积。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: House price against floor area
    name: fig:bivariate-ames-scatter
tags: [hide-input]
---
url = ('https://github.com/QuantEcon/data-lectures/raw/main/'
       'lectures/ames_house_prices.csv')
houses = pd.read_csv(url)
price = houses['price']
area = houses['living_area_sqft']

fig, ax = plt.subplots()
ax.scatter(area, price, alpha=0.3, s=10)
ax.set_xlabel('living area (square feet)')
ax.set_ylabel('sale price (US$)')
plt.show()
```

每个点代表一栋房子。

在 {doc}`observed_distributions` 中，我们使用箱线图比较了四个建筑面积分组之间的价格，发现建筑面积是价格的一个很好的预测因子。

上面的散点图更直接地展示了同样的事实（且无需先将数据分组）。

这正是二元分布可以描述的这类模式。

## 联合分布

让我们从一些理论和定义开始。

我们先讨论离散情形（求和），再讨论密度情形（积分）。

### 离散情形

让我们从两个离散随机变量 $X$ 和 $Y$ 开始，它们分别取有限集合 $S_X$ 和 $S_Y$ 中的值。

$X$ 和 $Y$ 的**联合概率质量函数**是定义在 $S_X \times S_Y$ 上的函数 $p$，满足

$$
p(x,y) = \mathbb P\{X = x, Y = y\}
$$

与单变量情形一样，$p$ 的取值是非负的，并且在两个变量上求和为一：

$$
\sum_{x \in S_X} \sum_{y \in S_Y} p(x,y) = 1
$$

让我们根据房价数据构建一个例子。

设如果一栋房子的建筑面积高于样本均值则 $X = 1$，否则 $X=0$，并以同样的方式为价格定义 $Y$。

```{code-cell} ipython3
X = (area > area.mean()).astype(int)
Y = (price > price.mean()).astype(int)
joint = pd.crosstab(X, Y, normalize=True)
joint.index.name = 'x (area above mean)'
joint.columns.name = 'y (price above mean)'
joint
```

这里的每个单元格表示具有该特定 $X$ 和 $Y$ 组合的房子所占的比例。

这个表中的四个数字可以视为 $X$ 和 $Y$ 的联合概率质量函数：它们是非负的，且总和为一。

让我们将其可视化为热力图，这样比原始数字更容易比较这四个单元格的相对大小。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Heatmap of the joint distribution
    name: fig:bivariate-joint-heatmap
---
fig, ax = plt.subplots()
sns.heatmap(joint, annot=True, fmt='.2f', cmap='viridis', cbar=False, vmin=0, vmax=0.5, ax=ax)
ax.invert_yaxis()  # so x increases upward, matching a standard scatter plot
plt.show()
```

（低于均值，低于均值）这一单元格明显是最大的：大约 48% 的房子同时具有低于均值的建筑面积和低于均值的价格。

给定一个联合分布，我们总是可以通过对 $Y$ 求和（或积分）来单独恢复出 $X$ 的分布。

在离散情形下，

$$
p_X(x) = \sum_{y \in S_Y} p(x,y)
$$

对 $p_Y$ 也是对称地成立。

我们称 $p_X$ 和 $p_Y$ 为 $X$ 和 $Y$ 的**边缘分布**。

在上面的表格中，边缘分布就是行和与列和：

```{code-cell} ipython3
p_X, p_Y = joint.sum(axis=1), joint.sum(axis=0)
p_X, p_Y
```

这两个边缘分布都不接近 0.5：只有约 45% 的房子建筑面积高于均值，而只有约 38% 的房子售价高于均值。

这正是我们在 {doc}`observed_distributions` 中遇到的右偏现象再次出现：大量昂贵的大房子形成的长右尾将均值拉高到分布中部之上，因此“低于均值”覆盖了超过一半的样本。

不过要注意，仅凭边缘分布并不能告诉我们全部情况。

分别知道建筑面积高于均值的房子的比例，以及价格高于均值的房子的比例，并不能说明*同一批*房子是否倾向于同时满足这两者——要回答这个问题，我们需要联合分布，而不仅仅是两个边缘分布。

让我们将两个边缘分布并排绘制出来。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The two marginal distributions
    name: fig:bivariate-discrete-marginals
---
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].bar(['below mean', 'above mean'], p_X)
axes[0].set_title('marginal of x (area)')
axes[0].set_ylabel('probability')
axes[1].bar(['below mean', 'above mean'], p_Y)
axes[1].set_title('marginal of y (price)')
axes[1].set_ylabel('probability')
plt.show()
```

### 连续情形

有些变量是连续的而非离散的（例如面积、价格）。

联合概率质量函数的连续类比是**联合概率密度函数** $p(x,y)$，它是 $\mathbb R^2$ 上的一个非负函数，满足

$$
\int_{\mathbb R} \int_{\mathbb R} p(x,y) \, dx \, dy = 1
$$

我们说 $(X,Y)$ 具有联合密度 $p$，如果对于每个区域 $A \subset \mathbb R^2$，

$$
\mathbb P\{(X,Y) \in A\} = \iint_A p(x,y) \, dx \, dy
$$

这意味着 $\mathbb P\{(X,Y) \in A\}$ 等于 $p(x,y)$ 与 $0$ 之间、位于二维区域 $A$ 之上的三维空间的体积。

让我们立刻认识一个具体且非常有用的例子：二元正态密度。

正如正态分布是单变量分布中的主力分布一样，**二元正态分布**是联合分布中的主力分布。

它有五个参数：两个均值 $\mu_X, \mu_Y$，两个标准差 $\sigma_X, \sigma_Y$，以及相关系数 $\rho \in (-1,1)$。

它的密度为

$$
p(x,y) = \frac{1}{2\pi \sigma_X \sigma_Y \sqrt{1-\rho^2}}
\exp\left(
-\frac{1}{2(1-\rho^2)}
\left[
\frac{(x-\mu_X)^2}{\sigma_X^2}
- \frac{2\rho (x-\mu_X)(y-\mu_Y)}{\sigma_X \sigma_Y}
+ \frac{(y-\mu_Y)^2}{\sigma_Y^2}
\right]
\right)
$$

可以证明，对于这个分布，$\rho$ 恰好就是 $X$ 和 $Y$ 之间的相关系数。

SciPy 通过 `scipy.stats.multivariate_normal` 提供了这个分布，它接受一个均值和一个协方差矩阵（这里由 $\sigma_X, \sigma_Y, \rho$ 构造而成）。

```{code-cell} ipython3
def bivariate_normal(μ_x, μ_y, σ_x, σ_y, ρ):
    cov = [[σ_x**2, ρ * σ_x * σ_y],
           [ρ * σ_x * σ_y, σ_y**2]]
    return scipy.stats.multivariate_normal([μ_x, μ_y], cov)
```

在俯视它之前，让我们先看看这个密度真实的样子：一个位于 $(x,y)$ 平面之上的曲面。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The bivariate normal density surface
    name: fig:bivariate-normal-surface
---
x_grid = np.linspace(-3, 3, 100)
y_grid = np.linspace(-3, 3, 100)
X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
pos = np.dstack((X_mesh, Y_mesh))

u = bivariate_normal(0, 0, 1, 1, 0.6)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X_mesh, Y_mesh, u.pdf(pos), cmap='viridis', linewidth=0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('density')
plt.show()
```

这和单变量正态密度一样，都是一个钟形的山丘，只不过这里是建立在一个平面之上而非一条直线之上，并且由相关系数 $\rho$ 带来了倾斜。

在实践中，直接俯视这个山丘并绘制其等高线会方便得多，这就像地形图在不绘制三维图形的情况下展示山丘的形状一样。

让我们对几个 $\rho$ 的取值这样做。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Bivariate normal contours by correlation
    name: fig:bivariate-normal-contours
---
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
for ax, ρ in zip(axes, (-0.8, 0.0, 0.8)):
    u = bivariate_normal(0, 0, 1, 1, ρ)
    ax.contour(X_mesh, Y_mesh, u.pdf(pos), levels=6, cmap='viridis')
    ax.set_title(rf'$\rho={ρ}$')
    ax.set_xlabel('x')
    ax.set_aspect('equal')
axes[0].set_ylabel('y')
plt.show()
```

当 $\rho = 0$ 时，等高线是圆形。

当 $\rho \neq 0$ 时，它们变成倾斜的椭圆，当 $\rho > 0$ 时沿直线 $y=x$ 方向倾斜，当 $\rho < 0$ 时沿 $y=-x$ 方向倾斜。

现在让我们用与离散情形相同的方法求出 $X$ 和 $Y$ 的边缘分布：对另一个变量积分联合密度。

$$
p_X(x) = \int_{-\infty}^\infty p(x,y) \, dy
$$

对 $p_Y$ 也是对称地成立。

可以证明，对于二元正态分布，$X$ 的边缘分布是 $N(\mu_X, \sigma_X^2)$，$Y$ 的边缘分布是 $N(\mu_Y, \sigma_Y^2)$——换句话说，每个变量单独来看都只是一个普通的单变量正态分布。

注意，两个边缘分布都完全不依赖于 $\rho$：相关系数描述的是 $X$ 和 $Y$ 如何*共同*变动，而这正是我们单独观察任一变量时所丢失的信息——这恰恰印证了我们在上面离散例子中得到的教训。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Marginal densities of the bivariate normal
    name: fig:bivariate-normal-marginals
---
μ_x, σ_x = 0, 1
μ_y, σ_y = 2, 0.6

x_grid = np.linspace(μ_x - 4*σ_x, μ_x + 4*σ_x, 200)
y_grid = np.linspace(μ_y - 4*σ_y, μ_y + 4*σ_y, 200)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].plot(x_grid, scipy.stats.norm(μ_x, σ_x).pdf(x_grid))
axes[0].set_title('marginal of x')
axes[0].set_xlabel('x')
axes[0].set_ylabel('density')
axes[1].plot(y_grid, scipy.stats.norm(μ_y, σ_y).pdf(y_grid))
axes[1].set_title('marginal of y')
axes[1].set_xlabel('y')
axes[1].set_ylabel('density')
plt.show()
```

## 独立性

如果联合分布可以分解为边缘分布的乘积，则称 $X$ 和 $Y$ 是**独立的**：

$$
p(x,y) = p_X(x) \, p_Y(y) \qquad \text{for all } x, y
$$

独立意味着知道 $X$ 的取值不会告诉我们任何关于 $Y$ 的信息，反之亦然。

让我们通过比较联合分布表与假设 $X$ 和 $Y$ *确实*独立时所得到的表（即边缘分布的乘积），来检验我们这个离散例子是否接近独立。

```{code-cell} ipython3
independent_table = pd.DataFrame(np.outer(p_X, p_Y),
                                  index=joint.index, columns=joint.columns)
independent_table
```

在独立性假设下，每个单元格都只是对应边缘分布的乘积——例如，（低于均值，低于均值）这个单元格将是 $0.55 \times 0.62 \approx 0.34$。

让我们把两个表作为热力图并排放置，这比比较原始数字更容易看出差异。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Actual joint versus independent joint
    name: fig:bivariate-independence-heatmap
---
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
sns.heatmap(joint, annot=True, fmt='.2f', cmap='viridis', cbar=False,
            vmin=0, vmax=0.45, ax=axes[0])
axes[0].set_title('actual joint')
axes[0].invert_yaxis()
sns.heatmap(independent_table, annot=True, fmt='.2f', cmap='viridis', cbar=False,
            vmin=0, vmax=0.45, ax=axes[1])
axes[1].set_title('if independent')
axes[1].invert_yaxis()
plt.show()
```

独立情形下的热力图并不是平坦的，但它是平滑的：它仅仅反映了边缘分布，由于两个边缘分布都偏向“低于均值”，因此（低于均值，低于均值）单元格最大。

实际的热力图看起来却相当不同：仅（低于均值，低于均值）这一个单元格就占了约 48% 的房子——远高于独立性所预测的 34%——而非对角线的单元格相应地比独立性所暗示的要空得多。

所以 $X$ 和 $Y$ 远非独立——这正是我们所预期的，因为更大的房子往往更昂贵。

## 协方差与相关系数

独立性是一个“非此即彼”的属性。

为了衡量依赖关系的*强度*和*方向*，我们使用**协方差**

$$
\mathrm{Cov}(X,Y) = \mathbb E \left[ (X - \mu_X)(Y - \mu_Y) \right]
$$

其中 $\mu_X = \mathbb E[X]$，$\mu_Y = \mathbb E[Y]$。

当 $X$ 和 $Y$ 倾向于同时高于各自的均值（或同时低于各自的均值）时，协方差为正；当其中一个倾向于高于均值而另一个倾向于低于均值时，协方差为负。

如果 $X$ 和 $Y$ 独立，则 $\mathrm{Cov}(X,Y) = 0$。

```{note}
反过来并不成立：协方差为零并不意味着独立。

它只是排除了*线性*依赖关系。

我们将在下文关于二元正态分布的讨论中看到这样的一个例子。
```

协方差的单位是 $X$ 的单位乘以 $Y$ 的单位，这使得单独看它很难解释。

因此我们通常将其标准化为**相关系数**

$$
\rho = \mathrm{Corr}(X,Y) = \frac{\mathrm{Cov}(X,Y)}{\sigma_X \sigma_Y}
$$

其中 $\sigma_X$ 和 $\sigma_Y$ 分别是 $X$ 和 $Y$ 的标准差。

相关系数没有单位，并且总是落在 $[-1, 1]$ 之间，只有当 $Y$ 是 $X$ 的精确线性函数时才能取到极值。

让我们使用 `np.corrcoef` 为我们的离散房屋例子计算它。

给定两个数组，`np.corrcoef` 会返回它们完整的 $2 \times 2$ 相关矩阵：对角线上是 1（每个变量与自身完全相关），非对角线的两个元素都是 $\mathrm{Corr}(X,Y)$。

我们只需要那一个数字，所以用 `[0, 1]` 索引来取出 $X$ 和 $Y$ 之间的相关系数。

```{code-cell} ipython3
np.corrcoef(X, Y)[0, 1]
```

高于均值/低于均值指示变量之间约 0.57 的相关系数证实了表格已经向我们展示的内容：建筑面积与价格是同向变动的。

## 联合分布是如何产生的

值得停下来思考一下两个变量*为什么*最终会出现相关性。

下面是两种简单且常见的机制。

### 独立的成分

最简单的情形是完全没有关系：独立地抽取 $X$ 和 $Y$。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Two independent normal variables
    name: fig:bivariate-independent
---
rng = np.random.default_rng(1234)
N = 500
x_indep = rng.standard_normal(N)
y_indep = rng.standard_normal(N)

fig, ax = plt.subplots()
ax.scatter(x_indep, y_indep, alpha=0.5, s=10)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
plt.show()
```

这里没有可见的模式：知道 $x$ 并不能告诉我们 $y$ 将落在哪里。

### 共同的构建部分

第二种，也是非常常见的机制是：$Y$ 的一部分*来自* $X$ 的构造。

例如，假设

$$
Y = a X + b + U
$$

其中 $U$ 是独立于 $X$ 的噪声，均值为零，标准差为 $\sigma_U$。

（可以把 $X$ 想象成建筑面积，把 $Y$ 想象成价格：更大的房子在机械意义上建造成本更高，再加上来自地段、装修质量等因素的一些噪声。）

由于 $U$ 独立于 $X$，所以 $\mathrm{Cov}(X, U) = 0$，因此

$$
\mathrm{Cov}(X,Y) = \mathrm{Cov}(X, aX + b + U) = a \, \mathrm{Cov}(X,X) = a \sigma_X^2
$$

类似地，$\mathbb V[Y] = a^2 \sigma_X^2 + \sigma_U^2$，所以

$$
\mathrm{Corr}(X,Y) = \frac{a \sigma_X}{\sqrt{a^2 \sigma_X^2 + \sigma_U^2}}
$$

相关系数完全由信号（$a X$）与噪声（$U$）的*相对*大小所决定。

让我们通过图像来观察这一点，固定 $a=1$ 和 $\sigma_X = 1$，并逐渐增大 $\sigma_U$。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Correlation strength as noise grows
    name: fig:bivariate-signal-noise
---
a = 1.0
sigma_U_vals = [0.2, 1.0, 3.0]

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
for ax, sigma_U in zip(axes, sigma_U_vals):
    x = rng.standard_normal(N)
    u = rng.normal(scale=sigma_U, size=N)
    y = a * x + u
    rho = a / np.sqrt(a**2 + sigma_U**2)
    ax.scatter(x, y, alpha=0.5, s=10)
    ax.set_title(rf'$\sigma_U={sigma_U}$, $\rho={rho:.2f}$')
    ax.set_xlabel('x')
axes[0].set_ylabel('y')
plt.show()
```

随着噪声的增大，点云变得更加分散，相关系数也随之下降，尽管底层的机制——$Y$ 由 $X$ 加上独立噪声构成——从未改变。

每当你看到两个相关的变量时，这是一个值得牢记在心的思维模型：通常存在某种驱动两者的共同成分，再加上其上独立的噪声。

## 回到正态分布

我们现在已经在一般情形下讨论了独立性、协方差与相关系数。

让我们回到二元正态密度，并将其与我们刚刚学到的内容联系起来。

回想一下上面的等高线图：当 $\rho = 0$ 时，等高线是圆形；当 $\rho \neq 0$ 时，它们是倾斜的椭圆。

在这里，圆形等高线正是独立性的体现：此时密度恰好分解为我们上面求出的两个边缘密度的乘积，即 $p(x,y) = p_X(x) \, p_Y(y)$。

所以，对于二元正态分布——*且仅*对于二元正态分布——零相关等价于独立。

```{note}
还有两个二元正态分布特有且值得了解的性质：

* 任何线性组合 $a X + b Y$ 都服从正态分布；
* 给定 $X=x$ 时 $Y$ 的条件分布本身也是正态的，其均值是 $x$ 的*线性*函数。

我们将在下一小节中用到第一个性质，并在后面预告线性回归时用到第二个性质。
```

值得看看从这个分布中抽取的一个*样本*是什么样的，因为真实数据永远不会以一个干净的密度形式出现——只会以一系列点的形式出现。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Simulated draws from a bivariate normal
    name: fig:bivariate-normal-sample
---
u = bivariate_normal(0, 0, 1, 1, 0.7)
sample = u.rvs(500, random_state=1234)

fig, ax = plt.subplots()
ax.scatter(sample[:, 0], sample[:, 1], alpha=0.4, s=10)
ax.contour(X_mesh, Y_mesh, u.pdf(pos), levels=6, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
```

这些点正好分散在等高线周围，形成了我们现在应该已经预料到的椭圆形点云——越靠近中心越密集，在尾部越稀疏。

请记住这幅图：它正是我们在下文转向真实数据时所要寻找的形状。

### 一个提醒

人们很容易以为，如果 $X$ 和 $Y$ 各自都是正态分布，那么这对 $(X,Y)$ 必定是二元正态分布。

这是错误的。

这里有一个简单的反例。

设 $X$ 为标准正态分布，并通过以下方式构造 $Y$：

$$
Y = \begin{cases} X & \text{if } |X| < 1 \\ -X & \text{if } |X| \geq 1 \end{cases}
$$

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Normal marginals, but not jointly normal
    name: fig:bivariate-not-normal
---
N = 2000
x = rng.standard_normal(N)
y = np.where(np.abs(x) < 1, x, -x)

g = sns.jointplot(x=x, y=y, height=5, alpha=0.4, s=8)
g.set_axis_labels('x', 'y')
plt.show()
```

边缘位置上的两个直方图看起来已经很像我们熟悉的正态密度钟形曲线了。

由于标准正态密度关于零点对称，当 $|X| \geq 1$ 时翻转 $X$ 的符号并不会改变它的分布，所以 $Y$ 也是标准正态分布。

```{code-cell} ipython3
scipy.stats.skew(y), scipy.stats.kurtosis(y)
```

两者都接近于零，这正符合正态边缘分布应有的特征。

然而，$(X,Y)$ 的联合分布看起来一点也不像我们上面看到的椭圆形点云——它集中在两条交叉的直线上，在上方中间面板中清晰可见。

我们在上面提到过，对于真正的二元正态分布而言，*任何*线性组合 $aX+bY$ 都是正态的。

这给了我们一个比逐一检查边缘分布更严格的检验：让我们来看看 $X+Y$。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Sum of the two variables
    name: fig:bivariate-sum-not-normal
---
s = x + y

fig, ax = plt.subplots()
ax.hist(s, bins=60, density=True)
ax.set_xlabel('x + y')
ax.set_ylabel('density')
plt.show()
```

大约三分之一的质量集中在零点处的一个尖峰上——恰好对应那些满足 $|X| \geq 1$ 的抽样，此时 $Y=-X$，因此 $X+Y=0$——其余部分则分布在两侧。

这与正态密度的差距之大，几乎是一个分布所能达到的极限，尽管 $X+Y$ 的偏度和峰度都接近于零，这两个指标所讲述的故事要平淡得多。

教训是：仅检查每个变量*单独*看起来是否正态是不足以证明可以采用二元正态模型的，甚至标准的数值诊断指标也可能漏掉一个图像能够立刻发现的失败情形。

我们应该使用下面的工具直接检验联合分布。

## 回到数据

让我们回到房价和建筑面积数据，更仔细地观察它们的联合分布。

我们在讲座开头已经看到过原始的散点图。

一种类似直方图的替代方案，是我们在 {doc}`observed_distributions` 中所用直方图的二维类比：**六边形分箱图（hexbin plot）**，它统计落入每个小六边形箱子中的点数，并据此为该箱子着色。

让我们用 `seaborn` 绘制一个这样的图，并在边缘附加每个变量的边缘直方图——这样一张图里就直接展示了联合分布以及我们上面所定义的两个边缘分布。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Joint and marginal distributions
    name: fig:bivariate-jointplot
tags: [hide-input]
---
g = sns.jointplot(x=area, y=price, kind='hex', height=5)
g.set_axis_labels('living area (square feet)', 'sale price (US$)')
plt.show()
```

中间的面板是六边形分箱形式的联合分布，与我们之前在散点图中看到的形状一致。

顶部和右侧的面板是两个边缘分布，恰好是我们在上面离散示例中用来构建边缘分布的行和与列和，只是现在应用到了原始的连续数据上。

样本相关系数从数值上证实了这种正向关系。

```{code-cell} ipython3
np.corrcoef(area, price)[0, 1]
```

回想一下 {doc}`fitting_distributions` 中提到，取对数使价格数据看起来更接近正态分布。

建筑面积也是如此，而两个取对数变量之间的相关系数与之前几乎相同。

```{code-cell} ipython3
log_price = np.log(price)
log_area = np.log(area)
np.corrcoef(log_area, log_price)[0, 1]
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Log price against log area
    name: fig:bivariate-log-scatter
tags: [hide-input]
---
fig, ax = plt.subplots()
ax.scatter(log_area, log_price, alpha=0.3, s=10)
ax.set_xlabel('log(living area)')
ax.set_ylabel('log(sale price)')
plt.show()
```

这张对数-对数散点图看起来是进行二元正态拟合的良好候选：一个沿主轴大致对称的椭圆形点云。

## 用矩方法拟合二元正态分布

回想一下 {doc}`fitting_distributions` 中提到的矩方法，它通过将样本矩与总体矩相匹配来选择参数。

二元正态分布有五个参数，因此我们需要匹配五个样本矩：两个均值、两个标准差，以及相关系数。

```{code-cell} ipython3
def fit_bivariate_normal(x, y):
    μ_x, μ_y = x.mean(), y.mean()
    σ_x, σ_y = x.std(), y.std()
    ρ = np.corrcoef(x, y)[0, 1]
    return bivariate_normal(μ_x, μ_y, σ_x, σ_y, ρ), (μ_x, μ_y, σ_x, σ_y, ρ)
```

```{code-cell} ipython3
fitted, (μ_x, μ_y, σ_x, σ_y, ρ) = fit_bivariate_normal(log_area, log_price)
μ_x, μ_y, σ_x, σ_y, ρ
```

让我们把拟合密度的等高线叠加在数据的散点图上。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Fitted bivariate normal density
    name: fig:bivariate-fit
---
x_grid = np.linspace(log_area.min(), log_area.max(), 100)
y_grid = np.linspace(log_price.min(), log_price.max(), 100)
X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
pos = np.dstack((X_mesh, Y_mesh))

fig, ax = plt.subplots()
ax.scatter(log_area, log_price, alpha=0.2, s=8)
ax.contour(X_mesh, Y_mesh, fitted.pdf(pos), levels=6, cmap='viridis')
ax.set_xlabel('log(living area)')
ax.set_ylabel('log(sale price)')
plt.show()
```

椭圆形等高线与点云的形状吻合得很好。

这种拟合并不完美——真实数据很少是完美的——但它较好地捕捉了数据的中心、离散程度和倾斜方向。

## 预告：回归作为条件均值

我们在上面提到，对于二元正态分布，给定 $X=x$ 时 $Y$ 的条件分布是正态的，其均值是 $x$ 的线性函数。

该条件均值的公式为

$$
\mathbb E[Y \mid X=x] = \mu_Y + \rho \frac{\sigma_Y}{\sigma_X} (x - \mu_X)
$$

这是一个非常有用的事实：它告诉我们在观测到 $X=x$ 的情况下，对 $Y$ 的最佳猜测（在均方误差意义下）是什么，而且它是 $x$ 的一条直线。

让我们把它绘制在我们拟合出的等高线之上。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The conditional mean line
    name: fig:bivariate-conditional-mean
---
slope = ρ * σ_y / σ_x
intercept = μ_y - slope * μ_x

fig, ax = plt.subplots()
ax.scatter(log_area, log_price, alpha=0.2, s=8)
ax.plot(x_grid, intercept + slope * x_grid, 'k--', lw=2,
        label=r'$\mathbb{E}[Y \mid X=x]$')
ax.set_xlabel('log(living area)')
ax.set_ylabel('log(sale price)')
ax.legend()
plt.show()
```

这条直线恰好穿过点云的中央，方向正是最能概括 $y$ 如何随 $x$ 变化的方向。

事实上，如果我们计算这些数据的普通最小二乘直线——即 {doc}`simple_linear_regression` 中所使用的方法——我们会得到完全相同的斜率和截距。

```{code-cell} ipython3
β, α = np.polyfit(log_area, log_price, 1)
(α, β), (intercept, slope)
```

这并非巧合。

对于这类数据而言，用普通最小二乘法拟合一条直线，与计算拟合出的二元正态分布的条件均值，其实是同一个计算从两个不同角度来看而已。

{doc}`simple_linear_regression` 将完整地展开普通最小二乘法，而无需依赖联合正态性的假设。

## 练习

```{exercise}
:label: bivariate_ex1

在 {doc}`observed_distributions` 中，我们使用小提琴图分别比较了亚马逊（Amazon）和好市多（Costco）股票的月度收益率，这是把每个序列单独处理的。

使用 `yfinance`，下载 `'AMZN'` 和 `'COST'` 在 2000-1-1 到 2024-1-1 之间的月度收盘价，并按照 {doc}`observed_distributions` 中的方法，为每一个计算月度收益率（百分比变化）。

1. 绘制这两个收益率序列的散点图，并计算它们的样本相关系数。
2. 用矩方法拟合一个二元正态分布，并将其等高线叠加在散点图上。
3. 回想一下 {doc}`fitting_distributions` 中提到，月度股票收益率比正态分布具有更重的尾部。

   考虑到这一点，你认为这里的二元正态拟合会比我们之前为房价找到的拟合更好还是更差？

   对照图像检验你的答案。
```

```{solution-start} bivariate_ex1
:class: dropdown
```

```{code-cell} ipython3
:tags: [hide-output]

!pip install yfinance
```

```{code-cell} ipython3
import yfinance as yf

df_amzn = yf.download('AMZN', '2000-1-1', '2024-1-1', interval='1mo')
x_amazon = df_amzn['Close'].pct_change()[1:]['AMZN'] * 100

df_cost = yf.download('COST', '2000-1-1', '2024-1-1', interval='1mo')
x_costco = df_cost['Close'].pct_change()[1:]['COST'] * 100

np.corrcoef(x_amazon, x_costco)[0, 1]
```

相关系数为适度的正值：这两只股票倾向于同向变动，但远非完美相关。

```{code-cell} ipython3
fitted, (μ_x, μ_y, σ_x, σ_y, ρ) = fit_bivariate_normal(x_amazon.values, x_costco.values)

x_grid = np.linspace(x_amazon.min(), x_amazon.max(), 100)
y_grid = np.linspace(x_costco.min(), x_costco.max(), 100)
X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
pos = np.dstack((X_mesh, Y_mesh))

fig, ax = plt.subplots()
ax.scatter(x_amazon, x_costco, alpha=0.4, s=10)
ax.contour(X_mesh, Y_mesh, fitted.pdf(pos), levels=6, cmap='viridis')
ax.set_xlabel('Amazon monthly return (%)')
ax.set_ylabel('Costco monthly return (%)')
plt.show()
```

这里的拟合效果明显比房价的情形要差。

散点中有好几个点远远落在最外层等高线之外，尤其是在 2000--2001 年和 2008--2009 年期间出现的极端收益率。

这与我们在 {doc}`fitting_distributions` 中已经知道的情况相符：月度股票收益率的尾部比正态分布所允许的更重，而这个缺陷也直接延续到了二元情形中。

```{solution-end}
```