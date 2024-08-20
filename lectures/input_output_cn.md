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

# 输入-输出模型

## 概述

在继续之前，这节课需要以下的导入和安装。

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon_book_networks
!pip install quantecon
!pip install pandas-datareader
```
图像输入功能：已启用
```

```{code-cell} ipython3
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import quantecon_book_networks
import quantecon_book_networks.input_output as qbn_io
import quantecon_book_networks.plotting as qbn_plt
import quantecon_book_networks.data as qbn_data
import matplotlib as mpl
from matplotlib.patches import Polygon

quantecon_book_networks.config("matplotlib")
mpl.rcParams.update(mpl.rcParamsDefault)
```

以下图形说明了从美国经济分析局2021年投入产出账户数据中获得的15个部门之间的联系网络。


```{code-cell} ipython3
:tags: [hide-cell]

def build_coefficient_matrices(Z, X):
    """
    Build coefficient matrices A and F from Z and X via

        A[i, j] = Z[i, j] / X[j]
        F[i, j] = Z[i, j] / X[i]

    """
    A, F = np.empty_like(Z), np.empty_like(Z)
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            A[i, j] = Z[i, j] / X[j]
            F[i, j] = Z[i, j] / X[i]

    return A, F

ch2_data = qbn_data.production()
codes = ch2_data["us_sectors_15"]["codes"]
Z = ch2_data["us_sectors_15"]["adjacency_matrix"]
X = ch2_data["us_sectors_15"]["total_industry_sales"]
A, F = build_coefficient_matrices(Z, X)
```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 美国15个部门生产网络
    name: us_15sectors
tags: [hide-input]
---
centrality = qbn_io.eigenvector_centrality(A)

# 移除自环
for i in range(A.shape[0]):
    A[i][i] = 0

fig, ax = plt.subplots(figsize=(8, 10))
plt.axis("off")
color_list = qbn_io.colorise_weights(centrality,beta=False)

qbn_plt.plot_graph(A, X, ax, codes,
              layout_type='spring',
              layout_seed=5432167,
              tol=0.0,
              node_color_list=color_list)

plt.show()
```


|标签|       部门      |标签|        部门       |标签|              部门           |
|:---:|:-------------:|:---:|:--------------:|:---:|:-------------------------:|
| ag  |  农业          | wh  |   批发业         | pr  |     专业服务                 |
| mi  |    采矿业       | re  |     零售业       | ed  |     教育和健康               |
| ut  |   公共事业      | tr  | 交通运输        | ar  |    艺术和娱乐               |
| co  |  建筑业        | in  |   信息业         | ot  |   其他服务（不含政府）       |
| ma  | 制造业        | fi  |    金融业         | go  |      政府                  |


从$i$到$j$的箭头意味着某些部门$i$的产出作为部门$j$生产的投入。

经济体的特征在于许多这样的联系。

一个基本的分析框架是[列昂节夫](https://en.wikipedia.org/wiki/Wassily_Leontief)的投入产出模型。


在介绍输入-输出模型后，我们描述它与{doc}`线性规划讲座 <lp_intro>`的一些联系。


## 投入产出分析

设

 * $x_0$ 是生产的单个外生投入量，比如劳动力
 * $x_j, j = 1,\ldots n$ 是最终商品$j$的总产出
 * $d_j, j = 1,\ldots n$ 是可供最终消费的最终商品$j$的净产出
 * $z_{ij} $ 是分配给生产商品$j$的输入商品$i$的数量，对于$i=1, \ldots n$, $j = 1, \ldots n$
 * $z_{0j}$ 是分配给生产商品$j$的劳动数量。
 * $a_{ij}$ 是生产一单位商品$j$所需的商品$i$的数量， $i=0, \ldots, n, j= 1, \ldots n$。
 * $w >0$ 是以每单位劳动计的外生工资，以美元为单位
 * $p$ 是一个 $n \times 1$ 的生产商品 $i = 1, \ldots , n$ 的价格向量。

对于 $j \in \{1, \ldots , n\}$ 生产商品的技术由 **列昂节夫** 函数描述

$$
    x_j = \min_{i \in \{0, \ldots , n \}} \left( \frac{z_{ij}}{a_{ij}}\right)
$$

### 两种商品

为说明问题，我们从设置 $n =2$ 并制定以下网络开始。

```{code-cell} ipython3
:tags: [hide-input]

G = nx.DiGraph()

nodes= (1, 2, 'c')
edges = ((1, 1), (1, 2), (2, 1), (2, 2), (1, 'c'), (2, 'c'))
edges1 = ((1, 1), (1, 2), (2, 1), (2, 2), (1, 'c'))
edges2 = [(2,'c')]
G.add_nodes_from(nodes)
G.add_edges_from(edges)

pos_list = ([0, 0], [2, 0], [1, -1])
pos = dict(zip(G.nodes(), pos_list))

fig, ax = plt.subplots()
plt.axis("off")

nx.draw_networkx_nodes(G, pos=pos, node_size=800,
                       node_color='white', edgecolors='black')
nx.draw_networkx_labels(G, pos=pos)
nx.draw_networkx_edges(G,pos=pos, edgelist=edges1,
                       node_size=300, connectionstyle='arc3,rad=0.2',
                       arrowsize=10, min_target_margin=15)
nx.draw_networkx_edges(G, pos=pos, edgelist=edges2,
                       node_size=300, connectionstyle='arc3,rad=-0.2',
                       arrowsize=10, min_target_margin=15)

plt.text(0.055, 0.125, r'$z_{11}$')
plt.text(1.825, 0.125, r'$z_{22}$')
plt.text(0.955, 0.1, r'$z_{21}$')
plt.text(0.955, -0.125, r'$z_{12}$')
plt.text(0.325, -0.5, r'$d_{1}$')
plt.text(1.6, -0.5, r'$d_{2}$')

plt.show()
```

对于$j = 1, 2$，将商品 $j$ 的生产设成

$$
x_j = \min \left\{ \frac{z_{1j}}{a_{1j}}, \frac{z_{2j}}{a_{2j}}  \right\}
$$

商品$j$的相应最终需求是

$$
d_j = x_j - z_{1j} - z_{2j}
$$


假设左图中的联系网络，给可以更改成：

$$
x_1 = \min \left\{ \frac{z_{11}}{a_{11}}, \frac{z_{21}}{a_{21}} \right\}
$$

$$
x_2 = \min \left\{ \frac{z_{12}}{a_{12}}, \frac{z_{22}}{a_{22}} \right\}
$$

这说明 $x_1$ 和 $x_2$ 的生产分别由 $a_{ij}$ 的投入密度和 $z_{ij}$ 的两单位输出期决定。

满足约束

$$
z_{ij} \leq a_{ij} x_{j}
$$

将满足

$$
d_j = x_j - z_{1j} - z_{2j} = x_j - x_j = 0
$$

所以最终需求需要产品和请求了没有额外市场的生产目的。如下图所示。

```{code-cell} ipython3
G = nx.DiGraph()

nodes= (1, 2, 'c')
edges = ((1, 1), (1, 2), (2, 1), (2, 2), (1, 'c'), (2, 'c'))
edges1 = ((1, 1), (1, 'c'))
edges2 = ((2, 'c'))
color_map = []
for node in G:
    if node == 1:
        color_map.append('blue')
    elif node == 2:
        color_map.append('red')
    else:
        color_map.append('green') 

fig, ax = plt.subplots()
plt.axis("off")

nx.draw_networkx_nodes(G, pos=pos, node_size=800,
                       node_color=color_map, edgecolors='black')
nx.draw_networkx_labels(G, pos=pos)
nx.draw_networkx_edges(G,pos=pos, edgelist=edges1,
                       node_size=300, connectionstyle='arc3,rad=0.2',
                       arrowsize=10, min_target_margin=15)
nx.draw_networkx_edges(G, pos=pos, edgelist=edges2,
                       node_size=300, connectionstyle='arc3,rad=-0.2',
                       arrowsize=10, min_target_margin=15)

plt.show()

---

与更多一般情况，生产约束为

$$
\begin{aligned}
(I - A) x &  \geq d \cr
a_0^\top x & \leq x_0
\end{aligned}
$$ (eq:inout_1)

其中 $A$ 是元素为 $a_{ij}$ 的 $n \times n$ 矩阵， $a_0^\top = \begin{bmatrix} a_{01} & \cdots & a_{0n} \end{bmatrix}$。



如果我们通过{eq}`eq:inout_1`的第一个方程组为总产出$x$求解，我们得到

$$
x = (I -A)^{-1} d \equiv L d
$$ (eq:inout_2)

其中矩阵 $L = (I-A)^{-1}$ 有时被称为 **列昂节夫逆矩阵**。



为了保证{eq}`eq:inout_2`的解$x$为正向量，以下**霍金斯-西蒙条件**是充分的：

$$
\begin{aligned}
\det (I - A) > 0 \text{ 并且} \;\;\; \\
(I-A)_{ij} > 0 \text{ 对所有 } i=j
\end{aligned}
$$



例如由以下描述的两商品经济

$$
A =
\begin{bmatrix}
    0.1 & 40 \\
    0.01 & 0
\end{bmatrix}
\text{ 和 }
d =
\begin{bmatrix}
    50 \\
    2
\end{bmatrix}
$$ (eq:inout_ex)

```{code-cell} ipython3
A = np.array([[0.1, 40],
             [0.01, 0]])
d = np.array([50, 2]).reshape((2, 1))
```


Hawkins-Simon 条件满足（ $|I -A|=1 > 0$ ），列昂节夫逆矩阵是

```{code-cell} ipython3
I = np.identity(2)
L = np.linalg.inv(I - A)
L
```

标准需求分析表明，产品1和2的必要总生产为两者均为正，分别为 $x = (10, 4001/4)$。我们可以通过求解用网络x2的公式，并且应用回归矩阵计算。

```{code-cell} ipython3
x = L @ d
x
```
### 总成本最小化

我们可以通过设定适当价格系统的方式来分析经济状态，测度不同生产的总生产成本被考虑为目标函数的最大化。

这里，我们假定，在{总成本最小化}的概念下，商品价格 $p$ 满足以下条件

(i) 利润最大化（价格 $w$ 和商品 $i$ 的生产公式）：

$$
p_j < \frac{\max p_i}{a_{ij}} \;\;\; \forall \; j = 1, \ldots n \\
\frac{1}{a_{0j}} \leq p_j < \min \max p_i
$$

所有设定满足

$$
\min p \cdot d + w x \\
min_{w: (I - A) x ) > 0
\frac{1}{a_{0j}} \leq p_j < \min \max p_i \cdot g(x)
$$

我们可以通过{(总成本最小化)}测度总生产。

---


基于此前一个与简明的观察，可用于描述经济网络及其总产出。

```{code-cell} ipython3
G = nx.DiGraph()

nodes= (1, 2, 'c')
edges = ((1, 1), (1, 2), (2, 1), (2, 2), (1, 'c'), (2, 'c'))
edges1 = ((1, 'c'))
edges2 = [(2, 'c')]

fig, ax = plt.subplots()
plt.axis("off")

nx.draw_networkx_nodes(G, pos=pos, node_size=800,
                       node_color=color_map, edgecolors='black')
nx.draw_networkx_labels(G, pos=pos)
nx.draw_networkx_edges(G,pos=pos, edgelist=edges1,
                       node_size=300, connectionstyle='arc3,rad=0.2',
                       arrowsize=10, min_target_margin=15)
nx.draw_networkx_edges(G, pos=pos, edgelist=edges2,
                       node_size=300, connectionstyle='arc3,rad=-0.2',
                       arrowsize=10, min_target_margin=15)

plt.show()
```


$(I - A)x$ 在 Hawkin-Simon 条件下有限。满足条件的常见网络给我们与之结构一样的图像。

列昂节夫逆矩阵可以计算这些。

记住每单位劳动分配 $p_j \geq L d_j$。劳动标准定价可以给我们逐单位成本。这些费用 (y=f(x)) 支持最小化的捍卫功能 {p}。

在此特性下的网络仿真图，如下所示。

```{code-cell} ipython3
det = np.linalg.det(I - A)  > 0 # 检查霍金斯-西蒙条件
det


```

现在，我们来计算**列昂节夫逆矩阵**

```{code-cell} ipython3
L = np.linalg.inv(B) # obtaining Leontief inverse matrix
L
```

完成所有步骤，包括网络图像和矩阵生成，可解释性和劳动力生产规范为

资源是外部，但如果我们给子索引写法，同样通用:

劳动力占用和生产需求，在前面的示例中，商品输出情况会得到实时综合性表述。

总结：

商品1 $多生产劳动力$的关联关系，意味着对于同样生产块的交互

商品2 与产品连接，商品1与部分产品需求。最终可以得出作为标准系统的 $\theta_{ij} = 1$ 和 $=\hat(w)$


网络和经济的连通代表性
## 是高度重要形成最终解决的过程!
```



## 生产可能性边界

{eq}`eq:inout_1`的第二个方程可以写成

$$
a_0^\top x = x_0
$$

或者

$$
A_0^\top d = x_0
$$ (eq:inout_frontier)

其中

$$
A_0^\top = a_0^\top (I - A)^{-1}
$$

对于 $i \in \{1, \ldots , n\}$，$A_0$ 的第 $i$ 个分量是生产一种最终产品 $i$ 所需的劳动量。

方程 {eq}`eq:inout_frontier` 描绘了一条可以通过外生劳动输入 $x_0$ 生产的最终消费包 $d$ 的 **生产可能性边界**。

考虑例子 {eq}`eq:inout_ex`。

假设我们现在给定

$$
a_0^\top = \begin{bmatrix}
4 & 100
\end{bmatrix}
$$

然后我们可以通过以下方式找到 $A_0^\top$

```{code-cell} ipython3
a0 = np.array([4, 100])
A0 = a0 @ L
A0

因此，这个经济体的生产可能性边界是

$$
10d_1 + 500d_2 = x_0
$$

---

## 价格

{cite}`DoSSo` 认为， $n$ 种生产的商品的相对价格必须满足

$$
\begin{aligned}
p_1 = a_{11}p_1 + a_{21}p_2 + a_{01}w \\
p_2 = a_{12}p_1 + a_{22}p_2 + a_{02}w
\end{aligned}
$$

更一般地，

$$
p = A^\top p + a_0 w
$$

这表明每个最终商品的价格等于总生产成本，它包括了中间投入的成本 $A^\top p$ 加上劳动成本 $a_0 w$。

这个方程可以写成

$$
(I - A^\top) p = a_0 w
$$ (eq:inout_price)

这意味着

$$
p = (I - A^\top)^{-1} a_0 w
$$

请注意 {eq}`eq:inout_price` 与 {eq}`eq:inout_1` 通过出现它们的转置运算符形成了一个 **共轭对**。

这种联系在经典的线性规划及其对偶问题中再次出现。

## 线性规划

一个 **原问题** 是

$$
\min_{x} w a_0^\top x
$$

约束条件为

$$
(I - A) x \geq d
$$

相关的 **对偶问题** 是

$$
\max_{p} p^\top d
$$

约束条件为

$$
(I -A)^\top p \leq a_0 w
$$

原问题选择一个可行的生产计划，以最小化交付预先分配的最终商品消费向量 $d$ 的成本。

对偶问题选择价格以最大化预先分配的最终商品向量 $d$ 的价值，前提是价格覆盖了生产成本。

根据[强对偶性定理](https://en.wikipedia.org/wiki/Dual_linear_program#Strong_duality)，原问题和对偶问题的最优值一致：

$$
w a_0^\top x^* = p^* d
$$

其中 $^*$ 表示原问题和对偶问题的最优选择。

对偶问题可以如下图形表示。

```{code-cell} ipython3
:tags: [hide-input]

fig, ax = plt.subplots()
ax.grid()

# 画约束线
ax.hlines(0, -1, 50)
ax.vlines(0, -1, 250)

ax.plot(np.linspace(4.75, 49, 100), (4-0.9*np.linspace(4.75, 49, 100))/(-0.16), color="r")
ax.plot(np.linspace(0, 50, 100), (33+1.46*np.linspace(0, 50, 100))/0.83, color="r")

ax.text(15, 175, "$(1-a_{11})p_1 - a_{21}p_2 \leq a_{01}w$", size=10)
ax.text(30, 85, "$-a_{12}p_1 + (1-a_{22})p_2 \leq a_{02}w$", size=10)

# 画可行区域
feasible_set = Polygon(np.array([[17, 69],
                                 [4, 0],
                                 [0,0],
                                 [0, 40]]),
                       color="cyan")
ax.add_patch(feasible_set)

# 画最优解
ax.plot(17, 69, "*", color="black")
ax.text(18, 60, "对偶解", size=10)

plt.show()

这个图展示了对偶问题的两个约束。 对偶问题的可行区域是多边形，当价格达到最优解的地方时，它会在其中一个边界点碰到。


一个更高的衡量值表示作为供应商的重要性更高。

因此，大多数行业的需求冲击会显著影响具有高特征向量中心性的行业的活动。

上图表明制造业是美国经济中最主要的行业。

### 产出乘数

另一种在投入产出网络中对行业进行排名的方法是通过产出乘数。

行业 $j$ 的**产出乘数** $\mu_j$ 通常定义为单位需求变化在整个行业范围内的总影响。

前面在讨论需求冲击时，我们得出结论，对于 $L = (l_{ij})$，元素 $l_{ij}$ 表示在行业 $j$ 的单位需求变化对行业 $i$ 的影响。

因此，

$$
\mu_j = \sum_{j=1}^n l_{ij}
$$

这可以写成 $\mu^\top = \mathbb{1}^\top L$ 或


$$
\mu^\top = \mathbb{1}^\top (I-A)^{-1}
$$

请注意，这里我们使用 $\mathbb{1}$ 表示一个全1向量。

在这一衡量标准下排名靠前的行业是中间商品的重要买家。

这些行业的需求冲击将对整个生产网络产生重大影响。

下图展示了 {numref}`us_15sectors` 中代表的行业的产出乘数

```{code-cell} ipython3
:tags: [hide-input]

A, F = build_coefficient_matrices(Z, X)
omult = qbn_io.katz_centrality(A, authority=True)

fig, ax = plt.subplots()
omult_color_list = qbn_io.colorise_weights(omult,beta=False)
ax.bar(codes, omult, color=omult_color_list, alpha=0.6)
ax.set_ylabel("Output multipliers", fontsize=12)
plt.show()
```

与特征向量中心性度量一样，制造和农业是排名最高的行业。

## 练习

```{exercise-start}
:label: io_ex1
```

{cite}`DoSSo`第9章讨论了以下参数设置的一个例子：

$$
A = \begin{bmatrix}
     0.1 & 1.46 \\
     0.16 & 0.17
    \end{bmatrix}
\text{ 和 }
a_0 = \begin{bmatrix} .04 & .33 \end{bmatrix}
$$

$$
x = \begin{bmatrix} 250 \\ 120 \end{bmatrix}
\text{ 和 }
x_0 = 50
$$

$$
d = \begin{bmatrix} 50 \\ 60 \end{bmatrix}
$$

描述它们如何从农产品和制造业的以下假设“数据”中推断出 $A$ 和 $a_0$ 的投入产出系数：

$$
z = \begin{bmatrix} 25 & 175 \\
         40 &   20 \end{bmatrix}
\text{ 和 }
z_0 = \begin{bmatrix} 10 & 40 \end{bmatrix}
$$

其中 $z_0$ 是在每个行业中使用的劳动力服务的向量。

```{exercise-end}
```

```{solution-start} io_ex1
:class: dropdown
```

对于每个 $i = 0,1,2$ 和 $j = 1,2$

$$
a_{ij} = \frac{z_{ij}}{x_j}
$$

```{solution-end}
```

```{exercise-start}
:label: io_ex2
```

推导前一个练习中描述的经济的生产可能性边界。

```{exercise-end}
```

```{solution-start} io_ex2
:class: dropdown
```

```{code-cell} ipython3
A = np.array([[0.1, 1.46],
              [0.16, 0.17]])
a_0 = np.array([0.04, 0.33])
```

```{code-cell} ipython3
I = np.identity(2)
B = I - A
L = np.linalg.inv(B)
```

```{code-cell} ipython3
A_0 = a_0 @ L
A_0
```

因此，生产可能性边界是

$$
0.17 d_1 + 0.69 d_2 = 50
$$

```{solution-end}
```