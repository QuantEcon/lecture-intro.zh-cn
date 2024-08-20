---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# 网络

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon-book-networks pandas-datareader
```
图像输入功能：已启用

## 大纲

近年来，一个叫做[网络科学](https://en.wikipedia.org/wiki/Network_science)的领域发展迅速。

网络科学研究对象之间的关系。

一个重要的例子是[万维网](https://en.wikipedia.org/wiki/World_Wide_Web#Linking)，其中网页通过超链接连接。

另一个例子是[人脑](https://en.wikipedia.org/wiki/Neural_circuit)：大脑功能的研究强调神经细胞（神经元）之间的网络连接。

[人工神经网络](https://en.wikipedia.org/wiki/Artificial_neural_network)基于这个思想，使用数据在简单的处理单元之间建立复杂的连接。

研究像COVID-19这样的疾病传播的流行病学家分析人类宿主组之间的相互作用。

在运筹学中，网络分析用于研究基本问题，如最小成本流、旅行商问题、[最短路径](https://en.wikipedia.org/wiki/Shortest_path_problem)和分配问题。

本讲座介绍了经济和金融网络。

本讲座的某些部分取自文本https://networks.quantecon.org/，但本讲座的水平更为入门。

我们需要以下导入。

```{code-cell} ipython3
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import quantecon as qe

import matplotlib.cm as cm
import quantecon_book_networks.input_output as qbn_io
import quantecon_book_networks.data as qbn_data

import matplotlib.patches as mpatches
```
图像输入功能：已启用

## 经济和金融网络

在经济学中，重要的网络示例包括

* 金融网络
* 生产网络
* 贸易网络
* 运输网络和
* 社交网络

社交网络影响市场情绪和消费者决策的趋势。

金融网络的结构有助于确定金融系统的相对脆弱性。

生产网络的结构影响贸易、创新和局部冲击的传播。

为了更好地理解这些网络 ，让我们更深入地看一些示例。

### 示例: 飞机出口

下图显示了基于国际贸易数据SITC修订版2, 在2019年大型商用飞机的国际贸易。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "商用飞机网络 \n"
    name: aircraft_network
tags: [hide-input]
---
ch1_data = qbn_data.introduction()
export_figures = False

DG = ch1_data['aircraft_network']
pos = ch1_data['aircraft_network_pos']

centrality = nx.eigenvector_centrality(DG)
node_total_exports = qbn_io.node_total_exports(DG)
edge_weights = qbn_io.edge_weights(DG)

node_pos_dict = pos

node_sizes = qbn_io.normalise_weights(node_total_exports,10000)
edge_widths = qbn_io.normalise_weights(edge_weights,10)

node_colors = qbn_io.colorise_weights(list(centrality.values()),color_palette=cm.viridis)
node_to_color = dict(zip(DG.nodes,node_colors))
edge_colors = []
for src,_ in DG.edges:
    edge_colors.append(node_to_color[src])

fig, ax = plt.subplots(figsize=(10, 10))
ax.axis('off')

nx.draw_networkx_nodes(DG,
                       node_pos_dict,
                       node_color=node_colors,
                       node_size=node_sizes,
                       linewidths=2,
                       alpha=0.6,
                       ax=ax)

nx.draw_networkx_labels(DG,
                        node_pos_dict,
                        ax=ax)

nx.draw_networkx_edges(DG,
                       node_pos_dict,
                       edge_color=edge_colors,
                       width=edge_widths,
                       arrows=True,
                       arrowsize=20,
                       ax=ax,
                       arrowstyle='->',
                       node_size=node_sizes,
                       connectionstyle='arc3,rad=0.15')

plt.show()
```

图中的圆圈被称为**节点**或**顶点**——在这种情况下，它们代表国家。

图中的箭头被称为**边**或**链接**。

节点大小与总出口成正比，边的宽度与对目标国家的出口成正比。

（数据来源于CID数据库，涉及重量至少为15,000公斤的商用飞机的贸易。）

图中显示美国、法国和德国是主要的出口枢纽。

在下面的讨论中，我们将学习量化这些想法。

### 示例：马尔可夫链

回顾，在我们的{ref}`马尔可夫链 <mc_eg2>`讲座中，我们研究了一个商业周期的动态模型，其中的状态是

* "ng" = "正常增长"
* "mr" = "轻度衰退"
* "sr" = "严重衰退"

让我们看看下面这幅图

```{image} /_static/lecture_specific/networks/mc.png
:name: mc_networks
:align: center
```

这是一个网络示例，其中节点集 $V$ 等于状态：

$$
    V = \{ \text{"ng", "mr", "sr"} \}
$$

节点之间的边显示了一个月的转换概率。

## 图论简介

现在我们已经看过了一些示例，让我们开始讨论理论。

这个理论将帮助我们更好地组织我们的思维。

网络科学的理论部分是通过一个主要的数学分支——[图论](https://en.wikipedia.org/wiki/Graph_theory) 构建的。

图论可以很复杂，我们只会覆盖基础知识。

然而，这些概念已经足以让我们讨论经济和金融网络中的有趣和重要的想法。

我们关注“有向”图，其中的连接通常是不对称的（箭头通常只指向一个方向，而不是两个方向）。

例如，

* 银行 $A$ 向银行 $B$ 贷款
* 企业 $A$ 向企业 $B$ 提供商品
* 个人 $A$ 在某个社交网络上“关注”个人 $B$

（“无向”图中，连接是对称的，它们是有向图的特例——我们只需要坚持每个从 $A$ 到 $B$ 的箭头都配对一个从 $B$ 到 $A$ 的箭头。）

### 关键定义

一个**有向图**由两部分组成：

1. 一个有限集 $V$
2. 一组对 $(u, v)$，其中 $u$ 和 $v$ 是 $V$ 的元素。

$V$ 的元素被称为图的**顶点**或**节点**。

$(u, v)$ 对被称为图的**边**，所有边的集合通常用 $E$ 表示。

直观和视觉上，边 $(u, v)$ 被理解为从节点 $u$ 到节点 $v$ 的箭头。

（表示箭头的一个简洁方式是记录箭头的尾部和头部的位置，这正是边所做的。）

在 {numref}`aircraft_network` 所示的飞机出口示例中

* $V$ 是数据集中包含的所有国家。
* $E$ 是图中的所有箭头，每个箭头表示从一个国家到另一个国家的某种正的飞机出口量。

让我们看看更多的示例。


## 贫困陷阱

下面显示了两张图，每张图有三个节点。

```{figure} /_static/lecture_specific/networks/poverty_trap_1.png
:name: poverty_trap_1

贫困陷阱
```

+++

现在我们构建一个具有相同节点但不同边的图。

```{figure} /_static/lecture_specific/networks/poverty_trap_2.png
:name: poverty_trap_2

贫困陷阱
```

+++

对于这些图，箭头（边）可以被认为代表在给定时间单位内的正迁移概率。

一般而言，如果存在一条边 $(u, v)$，则节点 $u$ 被称为 $v$ 的
**直接前驱**，$v$ 被称为 $u$ 的**直接后继**。

此外，对于 $v \in V$，

* **入度** 是 $i_d(v) = $ $v$ 的直接前驱数量
* **出度** 是 $o_d(v) = $ $v$ 的直接后继数量


### Networkx 中的有向图

Python 包 [Networkx](https://networkx.org/) 提供了一个方便的数据结构来表示有向图，并实现了许多常用的分析例程。

例如，让我们用 Networkx 重新创建 {numref}`poverty_trap_2`。

首先，我们创建一个空的 `DiGraph` 对象:

```{code-cell} ipython3
G_p = nx.DiGraph()
```

接下来我们用节点和边填充它。

为此，我们写下所有边的列表，用 *poor* 表示 *p* 等等：

```{code-cell} ipython3
edge_list = [('p', 'p'),
             ('m', 'p'), ('m', 'm'), ('m', 'r'),
             ('r', 'p'), ('r', 'm'), ('r', 'r')]
```

现在我们添加它们：

```{code-cell} ipython3
G_p.add_edges_from(edge_list)
```

最后，我们将边添加到我们的 `DiGraph` 对象中：

```{code-cell} ipython3
for e in edge_list:
    u, v = e
    G_p.add_edge(u, v)
```

或者，我们可以使用方法 `add_edges_from`.

```{code-cell} ipython3
G_p.add_edges_from(edge_list)
```

添加边会自动添加节点，因此 `G_p` 现在是我们图的正确表示。

我们可以通过以下代码通过 Networkx 绘制图来验证这一点：

```{code-cell} ipython3
fig, ax = plt.subplots()
nx.draw_spring(G_p, ax=ax, node_size=500, with_labels=True,
               font_weight='bold', arrows=True, alpha=0.8,
               connectionstyle='arc3,rad=0.25', arrowsize=20)
plt.show()
```

上面获得的图与 {numref}`poverty_trap_2` 中的原始有向图一致。

`DiGraph` 对象有方法可以计算节点的入度和出度。

例如，

```{code-cell} ipython3
G_p.in_degree('p')
```
图像输入功能：已启用

这告诉我们节点 `p` 有两个直接前驱。

### 强连通性

接下来，我们研究通信和连通性，这些对经济网络有重要影响。

当节点 $v$ 可以从节点 $u$ 访问时，可以称节点 $v$ 是**可到达**的，如果 $u=v$ 或者存在一个从 $u$ 到 $v$ 的边的序列。

* 在这种情况下，我们写作 $u \to v$

（在视觉上，有一系列从 $u$ 到 $v$ 的箭头。）

例如，假设我们有一个表示生产网络的有向图，其中

* $V$ 的元素是工业部门，并且
* 存在一个边 $(i, j)$ 表示 $i$ 向 $j$ 提供产品或服务。

那么 $m \to \ell$ 意味着部门 $m$ 是部门 $\ell$ 的上游供应商。

当 $u \to v$ 和 $v \to u$ 都成立时，两个节点 $u$ 和 $v$ 被称为**通信**。

当所有节点都可以通信时，一个图被称为**强连通**。

例如，{numref}`poverty_trap_1` 是强连通的，但在 {numref}`poverty_trap_2` 中，rich 无法从 poor 访问，因此它不是强连通的。

我们可以通过首先使用 Networkx 构建图，然后使用 `nx.is_strongly_connected` 方法验证这一点。

```{code-cell} ipython3
fig, ax = plt.subplots()
G1 = nx.DiGraph()

G1.add_edges_from([('p', 'p'),('p','m'),('p','r'),
             ('m', 'p'), ('m', 'm'), ('m', 'r'),
             ('r', 'p'), ('r', 'm'), ('r', 'r')])

nx.draw_networkx(G1, with_labels = True)
```

```{code-cell} ipython3
nx.is_strongly_connected(G1)    #检查上述图是否是强连通的
```

```{code-cell} ipython3
fig, ax = plt.subplots()
G2 = nx.DiGraph()

G2.add_edges_from([('p', 'p'),
             ('m', 'p'), ('m', 'm'), ('m', 'r'),
             ('r', 'p'), ('r', 'm'), ('r', 'r')])

nx.draw_networkx(G2, with_labels = True)
```

```{code-cell} ipython3
nx.is_strongly_connected(G2)    #检查上述图是否是强连通的
```

## 网络图

无向图是有向图的特例，其中所有边都是双向的。

中心设施和消费者的位置，以及路径依赖，由大宗商品的特点决定。

因为无向图是有向图的特例，所有的定义都是一致的。

## 加权图

我们现在介绍加权图，每条边都附有权重（数字）。

### 按国家划分的国际私人信贷流动

为了引出这个概念，考虑下图，它显示了按原籍国分类的私人银行之间的资金流动（即贷款）。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "国际信贷网络 \n"
    name: financial_network
tags: [hide-input]
---
Z = ch1_data["adjacency_matrix"]["Z"]
Z_visual= ch1_data["adjacency_matrix"]["Z_visual"]
countries = ch1_data["adjacency_matrix"]["countries"]

G = qbn_io.adjacency_matrix_to_graph(Z_visual, countries, tol=0.03)

centrality = qbn_io.eigenvector_centrality(Z_visual, authority=False)
node_total_exports = qbn_io.node_total_exports(G)
edge_weights = qbn_io.edge_weights(G)

node_pos_dict = nx.circular_layout(G)

node_sizes = qbn_io.normalise_weights(node_total_exports,3000)
edge_widths = qbn_io.normalise_weights(edge_weights,10)


node_colors = qbn_io.colorise_weights(centrality)
node_to_color = dict(zip(G.nodes,node_colors))
edge_colors = []
for src,_ in G.edges:
    edge_colors.append(node_to_color[src])

fig, ax = plt.subplots(figsize=(10, 10))
ax.axis('off')

nx.draw_networkx_nodes(G,
                       node_pos_dict,
                       node_color=node_colors,
                       node_size=node_sizes,
                       edgecolors='grey',
                       linewidths=2,
                       alpha=0.4,
                       ax=ax)

nx.draw_networkx_labels(G,
                        node_pos_dict,
                        font_size=12,
                        ax=ax)

nx.draw_networkx_edges(G,
                       node_pos_dict,
                       edge_color=edge_colors,
                       width=edge_widths,
                       arrows=True,
                       arrowsize=20,
                       alpha=0.8,
                       ax=ax,
                       arrowstyle='->',
                       node_size=node_sizes,
                       connectionstyle='arc3,rad=0.15')

plt.show()
```

国家代码在下表中给出

|代码|    国家    |代码| 国家 |代码|   国家   |代码|     国家    |
|:--:|:--------------|:--:|:-------:|:--:|:-----------:|:--:|:--------------:|
| AU |   澳大利亚   | DE | 德国 | CL |    智利    | ES |     西班牙      |
| PT |   葡萄牙   | FR |  法国 | TR |   土耳其    | GB | 英国 |
| US | 美国 | IE | 爱尔兰 | AT |   奥地利   | IT |     意大利      |
| BE |   比利时     | JP |  日本  | SW | 瑞士 | SE | 瑞典      |

从日本到美国的箭头表示日本银行对所有美国注册银行的总债权，由国际清算银行（BIS）收集。

图中每个节点的大小随着其他所有节点在该节点上的总外债增加而增大。

箭头的宽度与其代表的外债成比例。

注意，在这个网络中，几乎每对 $u$ 和 $v$ （即网络中的几乎每个国家）都存在一条边 $(u, v)$。

（实际上，还有更多的小箭头，我们为了清晰起见删除了。）

因此，从一个节点到另一个节点的边的存在并不是特别有信息量。

为了理解网络，我们需要记录不仅是信用流动的存在或缺失，还需要记录流动的大小。

记录这种信息的正确数据结构是“加权有向图”。

+++

### 定义

**加权有向图**是我们添加了一个**权重函数** $w$，该函数为每条边分配一个正数的有向图。

上图显示了一个加权有向图，其中权重是资金流动的大小。

下图显示了一个加权有向图，箭头表示诱导有向图的边。

```{figure} /_static/lecture_specific/networks/weighted.png
:name: poverty_trap_weighted

加权贫困陷阱
```

边上的数字是权重。

在这种情况下，您可以将箭头上的数字视为例如一年内家庭的迁移概率。

我们看到富裕家庭在一年内有10\% 的可能性变成贫困家庭。

## 邻接矩阵

我们可以用邻接矩阵来表示权重，这种方式对于数值工作非常方便。

节点 $\{v_1, \ldots, v_n\}$ 有边 $E$ 和权重函数 $w$ 的加权有向图的**邻接矩阵** 是

$$
A = (a_{ij})_{1 \leq i,j \leq n}
\quad \text{with} \quad
a_{ij} =
%
\begin{cases}
    w(v_i, v_j) & \text{ if } (v_i, v_j) \in E
    \\
    0           & \text{ otherwise}.
\end{cases}
%
$$

一旦枚举了 $V$ 中的节点，权重函数和邻接矩阵就提供了本质上相同的信息。

例如，假设 $\{$poor, middle, rich$\}$ 映射到 $\{1, 2, 3\}$，对应于加权有向图的邻接矩阵在 {numref}`poverty_trap_weighted` 中是

$$
\begin{pmatrix}
    0.9 & 0.1 & 0 \\
```

    0.4 & 0.4 & 0.2 \\
    0.1 & 0.1 & 0.8
\end{pmatrix}.
$$

在 QuantEcon 的 `DiGraph` 实现中，通过 `weighted` 关键字记录权重：

```{code-cell} ipython3
A = ((0.9, 0.1, 0.0),
     (0.4, 0.4, 0.2),
     (0.1, 0.1, 0.8))
A = np.array(A)
G = qe.DiGraph(A, weighted=True)    # 存储权重
```

### 转置的作用

我们选择邻接矩阵 $A$ 作为权重函数的矩阵表示

$$
(i, j) \to w(v_i, v_j).
$$

一种观察网络的方法是关注该网络的**转置**。

指出，在图 $G = (V, E)$ 对应的邻接矩阵 $A$ 中，$a_{ij}$ 是从节点 $v_i$ 到节点 $v_j$ 的边的权重。

图 $G$ 的**转置**是图 $\bar G = (V, \bar E)$，其中 $\bar E$ 是通过反转 $E$ 中所有边的方向获得的集合。

一般来说，对于有向图 $G$，我们定义**转置** $G^T$ 作为图，其中所有边的方向都被反转。

如果 $G$ 的邻接矩阵是 $A$，那么 $G^T$ 的邻接矩阵是 $A^T$，即 $A$ 的转置。

记住，转置运算不会改变边的权重。

+++

### 幂次

邻接矩阵 $A$ 的 $n$ 次幂是 $A$ 乘 $n$ 次自身。

也就是说 $A^{n}$ 是

$$
A \times \cdots \times A
$$

乘有向邻接矩阵的次数 $n$ 等于从 $i$ 到 $j$ 的路径总数，其中边的数量正好是 $n$。

对于图 $G$ 和邻接矩阵 $A$，请注意，下列关系成立，

$$
(A^n)_{ij}=\sum_{i_0,i_1,\cdots,i_{n-2},i_{n-1}}a_{i_0i_1}a_{i_1i_2}\cdots a_{i_{n-2}i_{n-1}}
$$

其中 $a_{ij}$ 是邻接矩阵 $A$ 的权重。

## 矩阵幂的应用

幂次运算有时是通过一个简单的例子展示其功效的最佳方式。

我们用二维向量 $\begin{pmatrix}1\\0\end{pmatrix}$ 和 $A^2$ 乘法进行运算，

$$
\begin{pmatrix}1\\0\end{pmatrix}
$$

然后将向量 $v$ 表示为两个随时间演化的行。

最重要的是，一旦我们有了邻接矩阵，我们可以非常高效地使用线性代数工具来计算有关网络的重要信息

接下来，我们在网络图上的简单示例。

```{code-cell} ipython3
fig, ax = plt.subplots()
G4 = nx.DiGraph()

G4.add_edges_from([('1','2'),
                   ('2','1'),('2','3'),
                   ('3','4'),
                   ('4','2'),('4','5'),
                   ('5','1'),('5','3'),('5','4')])
pos = nx.circular_layout(G4)

edge_labels={('1','2'): '100',
             ('2','1'): '50', ('2','3'): '200',
             ('3','4'): '100',
             ('4','2'): '500', ('4','5'): '50',
             ('5','1'): '150',('5','3'): '250', ('5','4'): '300'}

nx.draw_networkx(G4, pos, node_color = 'none', node_size = 500)
nx.draw_networkx_edge_labels(G4, pos, edge_labels=edge_labels)
nx.draw_networkx_nodes(G4, pos, linewidths= 0.5, edgecolors = 'black', node_color = 'none', node_size = 500)

plt.show()
```

我们看到银行 2 向银行 3 提供了一笔 200 的贷款。

相应的邻接矩阵为

$$
A =
\begin{pmatrix}
    0 & 100 & 0 & 0 & 0 \\
    50 & 0 & 200 & 0 & 0 \\
    0 & 0 & 0 & 100 & 0 \\
    0 & 500 & 0 & 0 & 50 \\
    150 & 0 & 250 & 300 & 0
\end{pmatrix}.
$$

转置矩阵为

$$
A^\top =
\begin{pmatrix}
    0   & 50  & 0   & 0   & 150 \\
    100 & 0   & 0   & 500 & 0 \\
    0   & 200 & 0   & 0   & 250 \\
    0   & 0   & 100 & 0   & 300 \\
    0   & 0   & 0   & 50  & 0
\end{pmatrix}.
$$

相应的网络在下图中可视化，显示贷款授予后的负债网络。

这两个网络（原始网络和转置网络）对分析金融市场都有用。

```{code-cell} ipython3
G5 = nx.DiGraph()

G5.add_edges_from([('1','2'),('1','5'),
                   ('2','1'),('2','4'),
                   ('3','2'),('3','5'),
                   ('4','3'),('4','5'),
                   ('5','4')])

edge_labels={('1','2'): '50', ('1','5'): '150',
             ('2','1'): '100', ('2','4'): '500',
             ('3','2'): '200', ('3','5'): '250',
             ('4','3'): '100', ('4','5'): '300',
             ('5','4'): '50'}

nx.draw_networkx(G5, pos, node_color = 'none',node_size = 500)
nx.draw_networkx_edge_labels(G5, pos, edge_labels=edge_labels)
nx.draw_networkx_nodes(G5, pos, linewidths= 0.5, edgecolors = 'black',
                       node_color = 'none',node_size = 500)

plt.show()
```

一般而言，每个非负 $n \times n$ 矩阵 $A = (a_{ij})$ 可以被视为加权有向图的邻接矩阵。

构建图时，我们设置 $V = 1, \ldots, n$ 并且选取边集 $E$ 为所有满足 $a_{ij} > 0$ 的 $(i,j)$。

对于权重函数，我们设置所有边 $(i,j)$ 的权重 $w(i, j) = a_{ij}$。

我们称这个图为 $A$ 所诱导的加权有向图。

## 性质

考虑一个具有邻接矩阵 $A$ 的加权有向图。

令 $a^k_{ij}$ 为 $A^k$ 的第 $k$ 次幂中的元素 $i, j$。

以下结果在许多应用中非常有用：

````{prf:theorem}
:label: graph_theory_property1

对于在 $V$ 中的不同节点 $i, j$ 和任意整数 $k$，我们有

$$
a^k_{i j} > 0
\quad \text{当且仅当} \quad
\text{ $j$ 可以从 $i$ 访问到}.
$$

````

+++

上面的结果在 $k=1$ 时是显然的，一般情况的证明可以在 {cite}`sargent2022economic` 中找到。

现在从特征值讲座中回忆，一个非负矩阵 $A$ 被称为 {ref}`不可约<irreducible>` 矩阵 ，如果对于每个 $(i,j)$ 存在一个 $\geq 0$ 的整数 $k$ 使得 $a^{k}_{ij} > 0$。

根据前面的定理，不难（详见 {cite}`sargent2022economic`）得出接下来的结果。

````{prf:theorem}
:label: graph_theory_property2

对于一个加权有向图，以下陈述是等价的：

1. 这个有向图是强连通的。
2. 这个图的邻接矩阵是不可约的。

````

+++

我们用一个简单的例子来说明上述定理。

考虑下图所示的加权有向图。

```{image} /_static/lecture_specific/networks/properties.png
:name: properties_graph

```

+++

我们首先将上面这个网络创建为一个 Networkx 的 `DiGraph` 对象。

```{code-cell} ipython3
G6 = nx.DiGraph()

G6.add_edges_from([('1','2'),('1','3'),
                   ('2','1'),
                   ('3','1'),('3','2')])
```
图像输入功能：已启用

接着我们计算其节点的数量、边的数量和输入度。

```{code-cell} ipython3
print('节点数量:', G6.number_of_nodes())
print('边的数量:', G6.number_of_edges())
print('节点1的输入度:', G6.in_degree('1'))
print('节点2的输入度:', G6.in_degree('2'))
print('节点3的输入度:', G6.in_degree('3'))
```

节点和边的数量，如预期一样，数量分别为 $|1,2,3|=3$ 和 $|E|=5$。

现在我们输出节点的邻接矩阵。

```{code-cell} ipython3
A = nx.adjacency_matrix(G6)
A_array = A.toarray()
print(A_array)
```

我们将图转置，然后再次检查节点的数量，边的数量和输入度。

```{code-cell} ipython3
G7 = G6.reverse()
print('转置后的节点数量:', G7.number_of_nodes())
print('转置后的边的数量:', G7.number_of_edges())
print('转置后节点1的输入度:', G7.in_degree('1'))
print('转置后节点2的输入度:', G7.in_degree('2'))
print('转置后节点3的输入度:', G7.in_degree('3'))
```

最后，我们通过检查图的连通性的方法来确认 `DiGraph` 对象的构建。

```{code-cell} ipython3
nx.is_strongly_connected(G6)      # 检查图的连通性
```

## 网络中心性

在研究各种类型的网络时，经常涉及的一个主题是不同节点的相对“中心性”或“重要性”。

例子包括

* 搜索引擎对网页的排名
* 确定在金融网络中最重要的银行（在金融危机中中央银行应该救助哪一个）
* 确定经济中最重要的工业部门。

在接下来的内容中，一个 **中心性度量** 与每个带权有向图相关联一个向量 $m$，其中 $m_i$ 被解释为节点 $v_i$ 的中心性（或排名）。

### 度中心性

在一个给定的有向图中，“重要性”的两个基本度量是其入度和出度。

这两者都提供了一个中心性度量。

入度中心性是一个向量，包含图中每个节点的入度。

考虑以下简单的例子。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 示例图
    name: sample_gph_1
---
G7 = nx.DiGraph()

G7.add_nodes_from(['1','2','3','4','5','6','7'])

G7.add_edges_from([('1','2'),('1','6'),
                   ('2','1'),('2','4'),
                   ('3','2'),
                   ('4','2'),
                   ('5','3'),('5','4'),
                   ('6','1'),
                   ('7','4'),('7','6')])
pos = nx.planar_layout(G7)

nx.draw_networkx(G7, pos, node_color='none', node_size=500)
nx.draw_networkx_nodes(G7, pos, linewidths=0.5, edgecolors='black',
                       node_color='none',node_size=500)

plt.show()
```

我们可以使用 Networkx 来计算图中所有节点的入度中心性

```{code-cell} ipython3
iG7 = [G7.in_degree(v) for v in G7.nodes()]   # 计算入度中心性

for i, d in enumerate(iG7):
    print(i+1, d)
```

以下是 Graph 的出度中心性度量：

```{code-cell} ipython3
dict(G7.out_degree()) # 使用 Networkx 的内置函数计算出度
```

### 特征向量中心性

总的来说，出度和入度中心性既简单又有用。

然而，它们并不总是捕捉我们需要的中心性概念。

考虑例如银行网络：有两个银行 $A$ 和 $B$，它们的入度分别是 $i_A = i_B$ 都=2。

如果银行 $A$ 的债务人是花旗和德意志银行，而银行 $B$ 的债务人是地方性合作银行，这两个银行的系统重要性应该相同吗？

答案显然是否定的。

在上面的例子中，重要的银行 $A$ 有一个重要的债务人 ${\rm Citibank}$，使它的重要性更大。

为了捕捉到这种差异性，网络科学家经常求助于更复杂的中心性度量。

其中一种是**特征向量中心性**。

特征向量中心性是一个向量 $m = (m_1, \ldots, m_n)$，其值被解释为 $G$ 的节点 $v_i$ 的中心性。

考虑加权邻接矩阵 $A$ 和由 $c = A v$ 给出的线性方程组。

当有向图 $G$ 的邻接矩阵 $A$ 本质上不可约时，特征向量中心性是

$$
m = A m
$$

的特征向量。

如下解释，Networkx 的实现依赖于线性代数方法来求解特征向量。

```{code-cell} ipython3
eigenvector_centralities = nx.eigenvector_centrality(G7)
print(eigenvector_centralities)  # 使用 Networkx 的内置函数计算特征向量中心性
```

```{code-cell} ipython3
fig, ax = plt.subplots()

df = centrality_plot_data(countries, indegree)

ax.bar('code', 'centrality', data=df, color=df["color"], alpha=0.6)

patch = mpatches.Patch(color=None, label='in degree', visible=False)
ax.legend(handles=[patch], fontsize=12, loc="upper left", handlelength=0, frameon=False)

ax.set_ylim((0,20))

plt.show()
```

遗憾的是，尽管入度和出度中心性的计算很简单，但并不总是具备信息量。

在 {numref}`financial_network` 中，几乎每个节点之间都有一条边，所以基于入度或出度的中心性排名无法有效区分国家。

在上述图中也可以看到这一点。

另一个例子是搜索引擎的任务，每当用户输入搜索时，搜索引擎都会按相关性对页面进行排名。

假设网页 A 的入链数量是页面 B 的两倍。

入度中心性告诉我们，页面 A 值得更高的排名。

但实际上，页面 A 可能比页面 B 重要性更低。

为了明白这一点，假设链接到 A 的页面几乎没有流量，而链接到 B 的页面流量很大。

在这种情况下，页面 B 可能会获得更多访问量，这反过来表明页面 B 包含更多有价值（或有趣）的内容。

考虑到这一点，可能暗示重要性可能是*递归的*。

这意味着，给定节点的重要性取决于链接到它的其他节点的重要性。

另一个例子是假设有一个生产网络，其中给定部门的重要性取决于其供应的部门的重要性。

这颠倒了前一个例子的顺序：现在给定节点的重要性取决于*它链接到*的其他节点的重要性。

下一个中心性度量将具有这些递归特性。


### 特征向量中心性

假设我们有一个带有邻接矩阵 $A$ 的加权有向图。

为简单起见，我们假设图的节点 $V$ 只是整数 $1, \ldots, n$。

令 $r(A)$ 表示 $A$ 的{ref}`谱半径<neumann_series_lemma>`。

**特征向量中心性** 被定义为求解以下 $n$-向量 $e$

$$ 
\begin{aligned}
    e = \frac{1}{r(A)} A e.
\end{aligned}
$$ (ev_central)

换句话说，$e$ 是 $A$ 的主特征向量（最大特征值的特征向量——详见特征值讲座中的{ref}`Perron-Frobenius 定理<perron-frobe>`讨论）。

为了更好地理解 {eq}`ev_central`，我们写出一些元素 $e_i$

$$
\begin{aligned}
    e_i = \frac{1}{r(A)} \sum_{1 \leq j \leq n} a_{ij} e_j
\end{aligned}
$$ (eq_eicen)

注意定义的递归性质：通过节点 $i$ 获得的中心性与所有节点的中心性之和成比例，其中这些节点是 $i$ 通过*流量比例*进入这些节点的加权。

如果
1. 离开 $i$ 的边很多，
2. 这些边有很大的权重，并且
3. 这些边指向其他高度排名的节点。

那么节点 $i$ 的排名就会很高。

稍后，当我们研究生产网络中的需求冲击时，对特征向量中心性将有更具体的解释。

我们将看到，在生产网络中，具有高特征向量中心性的部门是重要的*供应商*。

要绘制特定的节点

```{code-cell} ipython3
highlight_nodes = ["JPN", "CHN", "AUS"]
highlight_colors = ["darkred", "darkorange", "green"]
highlight_edge_colors = ["black", "gray", "brown"]

pos = nx.circular_layout(G)
node_sizes = [400 + np.max(node_total_exports) * 500 * centrality[g] for g in G.nodes()]

fig, ax = plt.subplots(figsize=(10,10))

nx.draw_networkx(G, pos=pos, node_size=node_sizes, alpha=0.5,
                 node_color="none", edge_color='grey', arrows=True, ax=ax)

nx.draw_networkx_nodes(G, pos=pos, nodelist=highlight_nodes,
                       node_size=node_sizes,
                       alpha=0.9, node_color=highlight_colors,
                       linewidths=3, ax=ax)

nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)

highlight_color_edge_tree = []

for idx, color in enumerate(highlight_colors):
    highlight_color_edge_tree.append(
        mpatches.Patch(color=color, label=highlight_nodes[idx])
    )

plt.legend(handles=highlight_color_edge_tree, loc='lower left')

plt.show()
```

## 生产网络

让 $(A, e)$ 表示生产网络，其中 $A$ 是邻接矩阵，$e$ 是特征向量中心性。

如果我们已经找到了 $A$ 的谱半径 $r$， 指定节点 $i$ 的中心性是按

$$e_i = \frac{1}{r(A)} \sum_{j} a_{ij} e_j$$

的递归关系定义的，例如 {eq}`eq_eicen` 的等式。

**主特征向量** 的另一个解释是让 $x = (x_1, .., x_n)$，令关联的输入输出矩阵为 $A = (A_{ij})$ 其中 $X = (x_1, .., x_n)$ 为实际输出向量。

假设 $Q = (q_{ij})$ 是表示所有行业间交易情况的交易矩阵，其中 $q_{ij}$ 是从行业 $i$ 向行业 $j$ 的交易量。特征向量 $e = (e_i,..,e_n)$ 表示行业 $i$ 的加权影响。

我们接下来学习如何用 Neighborhood 中心性来度量每个节点的重要性，涵义是在二阶段生产网络。

### 邻域中心性

我们可以看到特征向量中心性解释出供应商的重要性。如果一个节点被同级别更大的节点供应商链接，它就在市场中更健全、更有活力。

在生产网络 $V$ 中，`邻域中心性` 是一个向量 $\tilde{e}$，其值被解释为节点的中心度量。

当 $A$ 是 $V$ 的邻接矩阵时,

令 $\bar A$ 为 $A^{T}$，即反向图中的邻接矩阵。

可以看出 $\tilde{e}$ 是 $\bar A$ 的主特征向量。

如果 $\lambda_2$ 表示 $\bar{A}$ 的谱半径（最大特征值），则

$$ \tilde{e} = \frac{1}{\lambda_2} \bar {A} \tilde e. $$

这是供给链中邻域中心性的衡量方法。

假设 $A$ 是邻接矩阵，则 $\lambda _2$ 存在唯一的非负特征向量 $\tilde{e}$。

这也可以用 eigenvector_centrality 方法来计算。表明了同级别边最小半径的强大中心位置。

### 邻接矩阵与中心性度量的应用：COVID-19 疫情网络

接下来我们通过COVID-19疫情进行特征分析

```{code-cell} ipython3
import pandas as pd
covid_path = qbn_data.covid()
cov19_df = pd.read_csv(covid_path)

cov19_df['country'] = cov19_df['country'].replace(
            {
                'United States of America': 'US',
                'Russian Federation': 'Russia',
                'Central African Republic': 'CAR'
            }
        )
cov19_df = cov19_df.rename(columns={"cases_new": "new_cases", "country": "Country"})

country_to_idx = {country: idx for idx, country in enumerate(cov19_df["Country"].unique())}

dates = pd.to_datetime(cov19_df["date"])
unique_countries = pd.Series(cov19_df["Country"].unique())

for date in sorted(set(dates)):
    idx = pd.DateOffset(months=1)
    date_first = (dates >= (date - idx)).sum()
    date_second = date_first

print(unique_countries.size, dates.min(), dates.max())
```

### Matplotlib 中的可视化

最后绘制网络中节点的关闭中心性度量
```{code-cell} ipython3
values = [prox_cent.get(node, 0) for node in graph.nodes()]
nx.draw_spring(graph, cmap = plt.get_cmap('Blues'), node_color = values, node_size=1800, with_labels=True)
plt.show()
```

我们计算闭合中心性度量, 例如：

```{code-cell} ipython3
nx.closeness_centrality(graph)
```

到此为止，我们能够正式揭示网络的关键节点，并观察到网络中不同中心性度量的影响。

本讲座提供了经济和金融网络的初步、广泛的和实用的概述。在经济和金融网络分析来理解结构上重要信息时，采用不同工具和度量方法非常有助。

我们还有推荐的必读参考书籍：

1. *M. O. Jackson* (2008). **Social and Economic Networks**. Princeton Univ. Press.
2. Practical Economic Networks:
  https://networks.quantecon.org

本讲座引用了 Networks Lectures 及其他书籍中相关内容。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Eigenvector centrality
    name: eigenvctr_centrality
---
fig, ax = plt.subplots()

df = centrality_plot_data(countries, eig_central)

ax.bar('code', 'centrality', data=df, color=df["color"], alpha=0.6)

patch = mpatches.Patch(color=None, visible=False)
ax.legend(handles=[patch], fontsize=12, loc="upper left", handlelength=0, frameon=False)

plt.show()
```

```{code-cell} ipython3

eig_central 

[centrality_plot.find("code") for c in countries for x in eig_central]

import matplotlib.patches as mpatches
import matplotlib.cm as cm

df = centrality_plot_data(countries, eig_central)

def centrality_plot_data(countries, centrality):
    return pd.DataFrame({
        "code": countries,
        "centrality": [centrality.get(c, 0) for c in countries],
        "color": [
            if c in ["USA", "CHN"]
            else "blue"
            for c in countries
        ]
    })



plt.show()
```

请注意，特征向量中心性是一种递归度量，即一个节点的中心性被定义为指向它的节点的中心性之和。

（现在如果一个节点被许多高中心性的节点指向，它将有高的中心性。）

例如，加权有向图的基于特征向量的中心性，其邻接矩阵 $A$ 是向量 $e$ 解决了

$$
e = \frac{1}{r(A)} A^\top e.
$$

唯一的区别是我们替换 $A$ 为其转置矩阵。

（转置操作并不影响矩阵的谱半径，所以我们写了 $r(A)$ 而不是 $r(A^\top)$。）

对于每个元素而言，这个方程是

$$
e_j = \frac{1}{r(A)} \sum_{1 \leq i \leq n} a_{ij} e_i
$$

我们可以看到，当许多具有高权限排名的节点链接到 $j$ 时，$e_j$ 会很高。

下图显示了国际信贷网络用基于权限的特征向量中心性的排序情况，如 {numref}`financial_network` 所示。

```{code-cell} ipython3
ecentral_authority = eigenvector_centrality(Z, authority=True)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Eigenvector authority
    name: eigenvector_centrality
---
fig, ax = plt.subplots()

df = centrality_plot_data(countries, ecentral_authority)

ax.bar('code', 'centrality', data=df, color=df["color"], alpha=0.6)

patch = mpatches.Patch(color=None, visible=False)
ax.legend(handles=[patch], fontsize=12, loc="upper left", handlelength=0, frameon=False)

plt.show()
```

高度排名的国家与先前我们用 eigen_centrality 的结果惊人地吻合。

*日本仍然排名最高，接下来是美国和中国*。

这是我们从图形的性质知道的标准结果，
百分数排名仅次于美国。

其他指标也可以用在这种经济模型中，例如，例如追踪向量路径的 模型，.

## 习题

### 习题: 图 & 中心性

#### 编写函数

* 编写函数 `find_dominant_eigen` 使用 power 方法来近似求解特征值和特征向量.
* 编写函数 `rank_countries` 返回按重要性排序的前 `n` 个国家，使用 `eigenvector_centralities` 方法计算.

```{code-cell} ipython3
def find_dominant_eigen(matrix, maxit=1000, tol=1e-10):
    vec = np.ones(matrix.shape[0])
    
    for _ in range(maxit):
        vec_new = np.dot(matrix, vec)
        vec_new = vec_new / np.linalg.norm(vec_new)
        if np.linalg.norm(vec - vec_new) < tol:
            break
        vec = vec_new
    
    eigenvalue = np.dot(vec, np.dot(matrix, vec)) / np.dot(vec, vec)
    return eigenvalue, vec


def rank_countries(countries, eigenvector_centrality, n=10):
    centrality_pairs = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)
    return centrality_pairs[:n]

eigenvalue, eigenvector = find_dominant_eigen(Z)
ranked_countries = rank_countries(countries, ecentral_authority)

# 查看前 10 个重要国家
print(ranked_countries)
```

### 习题

考虑具有邻接矩阵 $A$ 的图 $G$。

1. 证明 $G$ 是强连通当且仅当对于所有 $u, v \in V$ ，由 $\text{MxPower} \left(A \right)$ 中的元素 $ > 0 $ ， 可以找到从 $u$ 到 $v$ 的路径。
2. 对于任意两个节点 $i$ 和 $j$ ， $i \neq j$ ，检验 $i$ 和 $j$ 是否可达，假设对于邻接矩阵 $A$ 满足 $\text{MxPower}(A_{i,j}) > 0$.
3. 思考 {numref}`networks_ex3` 中点的权重 $S_matrix$.

### 解答：
1. 证明：见最小生成树定理分析矩阵的 Rx 行列式。
2. 如下代码所示验证

```{code-cell} ipython3
matrix = np.array([[0, 1, 0, 1],
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [1, 0, 1, 0]])
```

```{code-cell} ipython3
A = nx.to_numpy_array(G)      #找到与G相关的邻接矩阵

A
```

```{code-cell} ipython3
oG = [G.out_degree(v) for v in G.nodes()]   # 计算入度中心性

for i, d in enumerate(oG):
    print(i, d)
```

```{code-cell} ipython3
e = eigenvector_centrality(A)   # 计算特征向量中心性
n = len(e)

for i in range(n):
    print(i+1, e[i])
```

```{solution-end}
```

```{exercise-start}
:label: networks_ex3
```

考虑一个有 $n$ 节点和 $n \times n$ 邻接矩阵 $A$ 的图 $G$。

设 $S = \sum_{k=0}^{n-1} A^k$

我们可以说对于任意两个节点 $i$ 和 $j$，当且仅当 $S_{ij} > 0$ 时，$j$ 对于 $i$ 是可达的。

设计一个函数 `is_accessible` 来检查给定图的任意两个节点是否是可达的。

考虑 {ref}`networks_ex2` 中的图，并使用该函数检查

1. $2$ 对 $1$ 是否可达
2. $6$ 对 $3$ 是否可达

```{exercise-end}
```

```{solution-start} networks_ex3
:class: dropdown
```

```{code-cell} ipython3
def is_accessible(G,i,j):
    A = nx.to_numpy_array(G)
    n = len(A)
    result = np.zeros((n, n))
    for i in range(n):
        result += np.linalg.matrix_power(A, i)
    if result[i,j]>0:
        return True
    else:
        return False
```

```{code-cell} ipython3
G = nx.DiGraph()

G.add_nodes_from(np.arange(8))  # 添加节点

G.add_edges_from([(0,1),(0,3),  # 添加边
                  (1,0),
                  (2,4),
                  (3,2),(3,4),(3,7),
                  (4,3),
                  (5,4),(5,6),
                  (6,3),(6,5),
                  (7,0)])
```

```{code-cell} ipython3
is_accessible(G, 2, 1)
```

```{code-cell} ipython3
is_accessible(G, 3, 6)
```

```{solution-end}
```