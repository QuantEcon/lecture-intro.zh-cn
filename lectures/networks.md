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

## 概述

近年来，一个被称为[网络科学](https://en.wikipedia.org/wiki/Network_science)的领域迅速发展。
网络科学研究对象群体之间的关系。
一个重要的例子是[万维网](https://en.wikipedia.org/wiki/World_Wide_Web#Linking)，其中网页通过超链接相互连接。
另一个例子是[人脑](https://en.wikipedia.org/wiki/Neural_circuit)：对大脑功能的研究强调神经细胞（神经元）之间的连接网络。
[人工神经网络](https://en.wikipedia.org/wiki/Artificial_neural_network)基于这一理念，利用数据在简单处理单元之间建立复杂的连接。
研究COVID-19等[疾病传播](https://en.wikipedia.org/wiki/Network_medicine#Network_epidemics)的流行病学家分析人类宿主群体之间的相互作用。
在运筹学中，网络分析用于研究基本问题，如最小成本流、旅行商问题、[最短路径](https://en.wikipedia.org/wiki/Shortest_path_problem)和分配问题。
本讲座介绍了经济和金融网络。
本讲座的部分内容来自教材书[《经济网络》](https://networks.quantecon.org)，但本讲座的水平更为入门。
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

## 经济和金融网络

在经济学中，网络的重要例子包括：
* 金融网络
* 生产网络
* 贸易网络
* 运输网络
* 社交网络

社交网络影响市场情绪和消费者决策的趋势。
金融网络的结构有助于确定金融系统的相对脆弱性。
生产网络的结构影响贸易、创新和局部冲击的传播。
为了更好地理解这些网络，让我们深入看一些例子。

### 例子：飞机出口

下图显示了基于国际贸易数据SITC第2修订版的2019年大型商用飞机国际贸易情况。

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

图中的圆圈被称为**节点**或**顶点** -- 在这个例子中，它们代表国家。
图中的箭头被称为**边**或**链接**。
节点大小与总出口成正比，边的宽度与对目标国家的出口成正比。
（数据针对重量至少15,000公斤的商用飞机贸易，数据来源于CID Dataverse。）
图表显示美国、法国和德国是主要的出口枢纽。
在下面的讨论中，我们将学习如何量化这些概念。

### 例子：马尔可夫链

回想一下，在我们关于{ref}`马尔可夫链 <mc_eg2>`的讲座中，我们研究了一个商业周期的动态模型，其中的状态包括：
* "ng" = "正常增长"
* "mr" = "轻度衰退"
* "sr" = "严重衰退"

让我们来看看下面这个图表

```{image} /_static/lecture_specific/networks/mc.png
:name: mc_networks
:align: center
```

这是一个网络的例子，其中节点集 $V$ 等于状态集：
$$
    V = \{ \text{"ng", "mr", "sr"} \}
$$
节点之间的边表示一个月的转移概率。

## 图论简介

现在我们已经看过一些例子，让我们转向理论部分。

这个理论将帮助我们更好地组织我们的思路。

网络科学的理论部分是使用数学的一个主要分支——[图论](https://en.wikipedia.org/wiki/Graph_theory)构建的。

图论可能很复杂，我们只会涉及基础部分。

但是，这些概念已经足以让我们讨论有关经济和金融网络的有趣且重要的想法。

我们关注"有向"图，其中连接通常是不对称的（箭头通常只指向一个方向，而不是双向）。

例如，
* 银行 $A$ 向银行 $B$ 贷款
* 公司 $A$ 向公司 $B$ 供应商品
* 个人 $A$ 在特定社交网络上"关注"个人 $B$

（"无向"图，即连接是对称的，是有向图的一种特殊情况——我们只需要坚持每个从 $A$ 指向 $B$ 的箭头都配对一个从 $B$ 指向 $A$ 的箭头。）

### 关键定义

一个**有向图**由两个部分组成：
1. 一个有限集 $V$ 和
2. 一组元素为 $V$ 中的元素的对 $(u, v)$。

$V$ 的元素被称为图的**顶点**或**节点**。

对 $(u,v)$ 被称为图的**边**，所有边的集合通常用 $E$ 表示。

直观和视觉上，边 $(u,v)$ 被理解为从节点 $u$ 到节点 $v$ 的一个箭头。

（一个用来表示箭头的简洁方法就是记录箭头的尾部和头部的位置，这正是边所做的。）

在 {numref}`aircraft_network` 所示的飞机出口例子中

* $V$ 是数据集中包含的所有国家。

* $E$ 是图中的所有箭头，每个箭头表示从一个国家到另一个国家的某个正数量的飞机出口。

让我们看更多的例子。

下面显示了两个图，每个图都有三个节点。

```{figure} /_static/lecture_specific/networks/poverty_trap_1.png
:name: poverty_trap_1

贫困陷阱
```

+++

现在，我们构造一个具有相同节点但具有不同边的图。

```{figure} /_static/lecture_specific/networks/poverty_trap_2.png
:name: poverty_trap_2

贫困陷阱
```

+++

对于这些图，箭头（边）可以被视为表示给定时间单位内的正向转移概率。

通常，如果存在边 $(u, v)$，则称节点 $u$ 为 $v$ 的**直接前驱**，$v$ 为 $u$ 的**直接后继**。

此外，对于 $v \in V$，

* **入度**是 $i_d(v) = $ $v$ 的直接前驱数，以及

* **出度**是 $o_d(v) = $ $v$ 的直接后继数。

### Networkx中的有向图

Python包 [Networkx](https://networkx.org/) 为表示有向图提供了一个便捷的数据结构，并实现了许多用于分析它们的常见例程。

作为示例，让我们使用Networkx重新创建 {numref}`poverty_trap_2`。

为此，我们首先创建一个空的 `DiGraph` 对象：

```{code-cell} ipython3
G_p = nx.DiGraph()
```

接下来，我们用节点和边来填充它。

为此，我们列出所有边的列表，其中*贫穷*用*p*表示，以此类推：

```{code-cell} ipython3
edge_list = [('p', 'p'),
             ('m', 'p'), ('m', 'm'), ('m', 'r'),
             ('r', 'p'), ('r', 'm'), ('r', 'r')]
```

最后，我们将边添加到我们的 `DiGraph` 对象中：

```{code-cell} ipython3
for e in edge_list:
    u, v = e
    G_p.add_edge(u, v)
```

或者，我们可以使用 `add_edges_from` 方法。

```{code-cell} ipython3
G_p.add_edges_from(edge_list)
```

添加边会自动添加节点，所以 `G_p` 现在是我们图的正确表示。

我们可以用以下代码通过Networkx绘制图来验证这一点：

```{code-cell} ipython3
fig, ax = plt.subplots()
nx.draw_spring(G_p, ax=ax, node_size=500, with_labels=True,
               font_weight='bold', arrows=True, alpha=0.8,
               connectionstyle='arc3,rad=0.25', arrowsize=20)
plt.show()
```

上面得到的图形与{numref}`poverty_trap_2`中的原始有向图相匹配。

`DiGraph`对象有计算节点入度和出度的方法。

例如，

```{code-cell} ipython3
G_p.in_degree('p')
```
(strongly_connected)=
### 通信

接下来，我们研究通信和连通性，这对经济网络有重要影响。

如果$u=v$或存在一系列从$u$到$v$的边，则称节点$v$从节点$u$**可达**。

* 在这种情况下，我们写作$u \to v$
（从视觉上看，有一系列箭头从$u$指向$v$。）

例如，假设我们有一个表示生产网络的有向图，其中
* $V$的元素是工业部门
* 边$(i, j)$的存在意味着$i$向$j$供应产品或服务。

那么$m \to \ell$意味着部门$m$是部门$\ell$的上游供应商。

如果$u \to v$且$v \to u$，则称两个节点$u$和$v$**相通**。

如果所有节点都相通，则称图是**强连通的**。

例如，{numref}`poverty_trap_1`是强连通的，
然而在{numref}`poverty_trap_2`中，富人节点从穷人节点不可达，因此它不是强连通的。

我们可以通过首先使用Networkx构建图，然后使用`nx.is_strongly_connected`来验证这一点。

```{code-cell} ipython3
fig, ax = plt.subplots()
G1 = nx.DiGraph()

G1.add_edges_from([('p', 'p'),('p','m'),('p','r'),
             ('m', 'p'), ('m', 'm'), ('m', 'r'),
             ('r', 'p'), ('r', 'm'), ('r', 'r')])

nx.draw_networkx(G1, with_labels = True)
```

```{code-cell} ipython3
nx.is_strongly_connected(G1)    #检查上面的图是否强关联
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
nx.is_strongly_connected(G2)    #检查上面的图是否强关联
```

## 加权图

我们现在介绍加权图，其中每条边都附有权重（数字）。

### 按国家划分的国际私人信贷流动

为了说明这个概念，请看下图，它展示了私人银行之间的资金流动（即贷款），按原始国家分组。

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
|:--:|:-----------|:--:|:----:|:--:|:--------:|:--:|:-----------:|
| AU |   澳大利亚  | DE | 德国 | CL |   智利   | ES |    西班牙   |
| PT |   葡萄牙    | FR | 法国 | TR |   土耳其 | GB |    英国     |
| US |   美国      | IE | 爱尔兰| AT |   奥地利 | IT |    意大利   |
| BE |   比利时    | JP | 日本 | SW |   瑞士   | SE |    瑞典     |

从日本到美国的箭头表示日本银行对所有在美国注册的银行的总体债权，这些数据由国际清算银行（BIS）收集。

图中每个节点的大小与所有其他节点对该节点的外国债权总额成正比。

箭头的宽度与它们所代表的外国债权成正比。

请注意，在这个网络中，几乎对于每一对节点$u$和$v$都存在一条边$(u, v)$（即网络中几乎每个国家之间都有联系）。

（事实上，还有更多的小箭头，为了清晰起见我们省略了。）

因此，仅仅一个节点到另一个节点边的存在与否并不特别有信息量。

为了理解这个网络，我们需要记录的不仅是信贷流动的存在或缺失，还要记录流动的规模。

记录这些信息的正确数据结构是"加权有向图"。

+++

### 定义

**加权有向图**是一种有向图，我们为其添加了一个**权重函数**$w$，该函数为每条边分配一个正数。

上图显示了一个加权有向图，其中权重是资金流动的大小。

下图显示了一个加权有向图，箭头表示诱导有向图的边。

```{figure} /_static/lecture_specific/networks/weighted.png
:name: poverty_trap_weighted

加权贫困陷阱
```


边旁边的数字是权重。

在这个例子中，你可以把箭头上的数字看作是一个家庭在一年内的转移概率。

我们可以看到，一个富裕家庭在一年内有10%的机会变成贫困。

## 邻接矩阵

另一种表示权重的方法是通过矩阵，这种方法在数值计算中非常方便。

对于一个有节点集 $\{v_1, \ldots, v_n\}$、边集 $E$ 和权重函数 $w$ 的加权有向图，其**邻接矩阵**是一个矩阵

$$
A = (a_{ij})_{1 \leq i,j \leq n}
\quad \text{其中} \quad
a_{ij} =
%
\begin{cases}
    w(v_i, v_j) & \text{如果} (v_i, v_j) \in E
    \\
    0           & \text{其他情况}
\end{cases}
%
$$

一旦 $V$ 中的节点被列举出来，权重函数和邻接矩阵本质上提供相同的信息。

例如，将 $\{$贫困, 中产, 富裕$\}$ 分别映射到 $\{1, 2, 3\}$，
{numref}`poverty_trap_weighted` 中加权有向图对应的邻接矩阵是

$$
\begin{pmatrix}
    0.9 & 0.1 & 0 \\
    0.4 & 0.4 & 0.2 \\
    0.1 & 0.1 & 0.8
\end{pmatrix}.
$$

在 QuantEcon 的 `DiGraph` 实现中，权重通过关键字 `weighted` 记录：

```{code-cell} ipython3
A = ((0.9, 0.1, 0.0),
     (0.4, 0.4, 0.2),
     (0.1, 0.1, 0.8))
A = np.array(A)
G = qe.DiGraph(A, weighted=True)    # 储存权重
```

关于邻接矩阵的一个关键点是，对其进行转置操作会*反转相关有向图中的所有箭头*。

例如，以下有向图可以被解释为一个金融网络的简化版本，其中节点代表银行，
边表示资金流动。

```{code-cell} ipython3
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

nx.draw_networkx(G4, pos, node_color = 'none',node_size = 500)
nx.draw_networkx_edge_labels(G4, pos, edge_labels=edge_labels)
nx.draw_networkx_nodes(G4, pos, linewidths= 0.5, edgecolors = 'black',
                       node_color = 'none',node_size = 500)

plt.show()
```

我们看到银行2向银行3发放了200的贷款。

对应的邻接矩阵是

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

其转置是

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

相应的网络在下图中可视化，显示了贷款发放后的负债网络。

这两个网络（原始和转置）对于分析金融市场都很有用。

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

一般来说，每个非负的 $n \times n$ 矩阵 $A = (a_{ij})$ 都可以被视为加权有向图的邻接矩阵。

要构建图，我们设 $V = 1, \ldots, n$，并将边集 $E$ 取为所有满足 $a_{ij} > 0$ 的 $(i,j)$。

对于权重函数，我们对所有边 $(i,j)$ 设置 $w(i, j) = a_{ij}$。

我们称这个图为 $A$ 诱导的加权有向图。

## 性质

考虑一个加权有向图，其邻接矩阵为 $A$。

令 $a^k_{ij}$ 为 $A^k$（$A$ 的 $k$ 次方）中的第 $i,j$ 个元素。

以下结果在许多应用中很有用：

````{prf:theorem}
:label: graph_theory_property1

对于 $V$ 中的不同节点 $i, j$ 和任意整数 $k$，我们有
$$
a^k_{i j} > 0
\quad \text{当且仅当} \quad
\text{$j$ 可从 $i$ 到达}。
$$

````

+++

当 $k=1$ 时，上述结果是显而易见的，对于一般情况的证明可以在 {cite}`sargent2022economic` 中找到。

现在回想特征值讲座中提到，一个非负矩阵 $A$ 被称为{ref}`不可约的<irreducible>`，如果对于每个 $(i,j)$，存在一个整数 $k \geq 0$，使得 $a^{k}_{ij} > 0$。

根据前面的定理，不难得出下一个结果（详见 {cite}`sargent2022economic`）。

````{prf:theorem}
:label: graph_theory_property2

对于一个加权有向图，以下陈述是等价的：
1. 该有向图是强连通的。
2. 该图的邻接矩阵是不可约的。

````

+++

我们用一个简单的例子来说明上述定理。
考虑以下加权有向图。


```{image} /_static/lecture_specific/networks/properties.png
:name: properties_graph

```

+++

我们首先将上述网络创建为 Networkx `DiGraph` 对象。

```{code-cell} ipython3
G6 = nx.DiGraph()

G6.add_edges_from([('1','2'),('1','3'),
                   ('2','1'),
                   ('3','1'),('3','2')])
```

然后我们构建相关的邻接矩阵A。

```{code-cell} ipython3
A = np.array([[0,0.7,0.3],    # 邻接矩阵A。
              [1,0,0],
              [0.4,0.6,0]])
```

```{code-cell} ipython3
:tags: [hide-input]

def is_irreducible(P):
    n = len(P)
    result = np.zeros((n, n))
    for i in range(n):
        result += np.linalg.matrix_power(P, i)
    return np.all(result > 0)
```

```{code-cell} ipython3
is_irreducible(A)      #检查A的不可约性
```

```{code-cell} ipython3
nx.is_strongly_connected(G6)      # 检查图的连接性
```

## 网络中心性

在研究各种网络时，一个反复出现的话题是不同节点的相对"中心性"或"重要性"。

例子包括：
* 搜索引擎对网页的排名
* 确定金融网络中最重要的银行（在金融危机时中央银行应该救助哪一家）
* 确定经济中最重要的工业部门

在接下来的内容中，**中心性度量**将每个加权有向图与一个向量$m$关联起来，其中$m_i$被解释为节点$v_i$的中心性（或排名）。

### 度中心性

在给定的有向图中，衡量节点"重要性"的两个基本指标是其入度和出度。

这两者都提供了一种中心性度量。

入度中心性是一个包含图中每个节点入度的向量。

考虑以下简单例子。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 样本图
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

以下代码显示了所有节点的入度中心性。

```{code-cell} ipython3
iG7 = [G7.in_degree(v) for v in G7.nodes()]   #计算入度中心性
for i, d in enumerate(iG7):
    print(i+1, d)
```

考虑{numref}`financial_network`中显示的国际信贷网络。

以下图表显示了每个国家的入度中心性。

```{code-cell} ipython3
D = qbn_io.build_unweighted_matrix(Z)
indegree = D.sum(axis=0)
```

```{code-cell} ipython3
def centrality_plot_data(countries, centrality_measures):
    df = pd.DataFrame({'code': countries,
                       'centrality':centrality_measures,
                       'color': qbn_io.colorise_weights(centrality_measures).tolist()
                       })
    return df.sort_values('centrality')
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

虽然入度和出度中心性计算简单，但它们并不总是有用。

在{numref}`financial_network`中，几乎每个节点之间
都存在边，所以基于入度或出度的中心性排名无法有效区分国家。

这在上图中也可以看到。

另一个例子是网络搜索引擎的任务，它在用户输入搜索时按相关性对页面进行排名。

假设网页A的入站链接是网页B的两倍。

入度中心性告诉我们页面A应该获得更高的排名。

但实际上，页面A可能不如页面B重要。

要理解这一点，可以设想指向A页面的链接来自几乎无人访问的页面，而指向B页面的链接则来自访问量极高的页面。


在这种情况下，页面B可能会收到更多访问者，这反过来表明
页面B包含更有价值（或更有趣）的内容。

这一点告诉我们重要性可能是*递归的*。
这意味着给定节点的重要性取决于
链接到它的其他节点的重要性。

作为另一个例子，我们可以想象一个生产网络，其中一个
给定部门的重要性取决于它供应的部门的重要性。

这与前面的例子相反：现在给定节点的重要性
取决于它*链接到的*其他节点的重要性。

下面的中心性度量将具有这些递归特征。

## 特征向量中心性

假设我们有一个带有邻接矩阵$A$的加权有向图。

为简单起见，我们假设图的节点$V$就是
整数$1, \ldots, n$。

让$r(A)$表示$A$的{ref}`谱半径<neumann_series_lemma>`。

图的**特征向量中心性**被定义为解决以下方程的$n$维向量$e$

$$ 
\begin{aligned}
    e = \frac{1}{r(A)} A e.
\end{aligned}
$$ (ev_central)

换句话说，$e$是$A$的主特征向量（最大特征值的特征向量 --- 
请看特征值lecture中关于{ref}`Perron-Frobenius定理<perron-frobe>`的讨论。

为了更好地理解{eq}`ev_central`，我们写出某个元素$e_i$的完整表达式

$$
\begin{aligned}
    e_i = \frac{1}{r(A)} \sum_{1 \leq j \leq n} a_{ij} e_j
\end{aligned}
$$ (eq_eicen)

注意定义的递归性质：节点$i$获得的中心性
与所有节点中心性的加权和成正比，权重为
从$i$到这些节点的*流量率*。

如果节点$i$满足以下条件，则它的排名很高：
1. 从$i$出发的边很多，
2. 这些边的权重很大，以及
3. 这些边指向其他排名很高的节点。

稍后，当我们研究生产网络中的需求冲击时，特征向量中心性将有更具体的解释。

我们将看到，在生产网络中，具有高特征向量中心性的部门是重要的*供应商*。

特别是，一旦订单通过网络向后流动，它们就会被各种需求冲击激活。

要计算特征向量中心性，我们可以使用以下函数。

```{code-cell} ipython3
def eigenvector_centrality(A, k=40, authority=False):
    """
   计算矩阵A的主特征向量。假设A是本原矩阵，并使用幂法。

    """
    A_temp = A.T if authority else A
    n = len(A_temp)
    r = np.max(np.abs(np.linalg.eigvals(A_temp)))
    e = r**(-k) * (np.linalg.matrix_power(A_temp, k) @ np.ones(n))
    return e / np.sum(e)
```

让我们为{numref}`sample_gph_1`中生成的图计算特征向量中心性。

```{code-cell} ipython3
A = nx.to_numpy_array(G7)         #计算图的邻接矩阵
```

```{code-cell} ipython3
e = eigenvector_centrality(A)
n = len(e)

for i in range(n):
    print(i+1,e[i])
```

虽然节点 $2$ 和 $4$ 具有最高的入度中心性，但我们可以看到节点 $1$ 和 $2$ 具有最高的特征向量中心性。

让我们重新审视{numref}`financial_network`中的国际信贷网络。

```{code-cell} ipython3
eig_central = eigenvector_centrality(Z)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 特征向量中心性
    name: eigenvctr_centrality
---
fig, ax = plt.subplots()

df = centrality_plot_data(countries, eig_central)

ax.bar('code', 'centrality', data=df, color=df["color"], alpha=0.6)

patch = mpatches.Patch(color=None, visible=False)
ax.legend(handles=[patch], fontsize=12, loc="upper left", handlelength=0, frameon=False)

plt.show()
```

根据这个排名评分较高的国家往往在信贷供应方面扮演重要角色。

日本在这项指标中排名最高，尽管像英国和法国这样拥有大型金融部门的国家也紧随其后。

特征向量中心性的优势在于它在衡量节点重要性的同时考虑了其邻居的重要性。

谷歌的PageRank算法的核心就是特征向量中心性的一个变体，用于对网页进行排名。

其主要原理是来自重要节点（以度中心性衡量）的链接比来自不重要节点的链接更有价值。

### 卡茨中心性

特征向量中心性的一个问题是$r(A)$可能为零，这种情况下$1/r(A)$是未定义的。

出于这个和其他原因，一些研究者更倾向于使用另一种称为卡茨中心性的网络中心性度量。

固定$\beta$在$(0, 1/r(A))$区间内，带权有向图的**卡茨中心性**定义为解决以下方程的向量$\kappa$：

$$
\kappa_i =  \beta \sum_{1 \leq j 1} a_{ij} \kappa_j + 1
\qquad  \text{对所有 } i \in \{0, \ldots, n-1\}。
$$ (katz_central)

这里$\beta$是我们可以选择的参数。

用向量形式我们可以写成：

$$
\kappa = \mathbf 1 + \beta A \kappa
$$ (katz_central_vec)

其中$\mathbf 1$是一个全为1的列向量。

这种中心性度量背后的直觉与特征向量中心性提供的类似：当节点$i$被本身具有高中心性的节点链接时，它就会获得高中心性。

只要$0 < \beta < 1/r(A)$，卡茨中心性总是有限且定义明确的，因为这时$r(\beta A) < 1$。

这意味着方程{eq}`katz_central_vec`有唯一解：

$$
\kappa = (I - \beta A)^{-1} \mathbf{1}
$$

这是由{ref}`诺伊曼级数定理<neumann_series_lemma>`得出的。

参数$\beta$用于确保$\kappa$是有限的。

当$r(A)<1$时，我们使用$\beta=1$作为卡茨中心性计算的默认值。


### 权威与枢纽
搜索引擎设计者认识到网页可以通过两种不同的方式变得重要。

一些页面具有高**枢纽中心性**，意味着它们链接到有价值的信息来源（例如，新闻聚合网站）。

其他页面具有高**权威中心性**，意味着它们包含有价值的信息，这一点通过指向它们的链接的数量和重要性来体现（例如，受人尊敬的新闻机构的网站）。

类似的概念也已经应用于经济网络（通常使用不同的术语）。

我们之前讨论的特征向量中心性和Katz中心性衡量的是枢纽中心性。
（如果节点指向其他具有高中心性的节点，则它们具有高中心性。）

如果我们更关心权威中心性，我们可以使用相同的定义，只是取邻接矩阵的转置。

这之所以有效，是因为取转置会反转箭头的方向。

（现在，如果节点接收来自其他具有高中心性的节点的链接，它们将具有高中心性。）

例如，具有邻接矩阵$A$的加权有向图的**基于权威的特征向量中心性**是解决以下方程的向量$e$：

$$
e = \frac{1}{r(A)} A^\top e.
$$ (eicena0)

与原始定义的唯一区别是$A$被其转置替代。

（转置不影响矩阵的谱半径，所以我们写成$r(A)$而不是$r(A^\top)$。）

按元素逐个表示，这可以写成：

$$
e_j = \frac{1}{r(A)} \sum_{1 \leq i \leq n} a_{ij} e_i
$$ (eicena)

我们可以看到，如果许多具有高权威排名的节点链接到$j$，则$e_j$将会很高。

下图显示了{numref}`financial_network`中所示的国际信贷网络的基于权威的特征向量中心性排名。

```{code-cell} ipython3
ecentral_authority = eigenvector_centrality(Z, authority=True)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 特征向量权威度
    name: eigenvector_centrality
---
fig, ax = plt.subplots()

df = centrality_plot_data(countries, ecentral_authority)

ax.bar('code', 'centrality', data=df, color=df["color"], alpha=0.6)

patch = mpatches.Patch(color=None, visible=False)
ax.legend(handles=[patch], fontsize=12, loc="upper left", handlelength=0, frameon=False)

plt.show()
```

排名靠前的国家是那些吸引大量信贷流入或从其他主要参与者获得信贷流入的国家。
在这种情况下，美国作为银行间信贷的目标显然主导了排名。

## 进一步阅读

我们将本讲座中讨论的观点应用于：

关于经济和社会网络的教科书包括 {cite}`jackson2010social`、
{cite}`easley2010networks`、{cite}`borgatti2018analyzing`、
{cite}`sargent2022economic` 和 {cite}`goyal2023networks`。

在网络科学领域，{cite}`newman2018networks`、{cite}`menczer2020first` 和
{cite}`coscia2021atlas` 的著作都非常出色。

## 练习

```{exercise-start}
:label: networks_ex1
```

这是一个适合喜欢证明的人的数学练习。

设 $(V, E)$ 是一个有向图，若 $u$ 和 $v$ 互通，则记作 $u \sim v$。

证明 $\sim$ 是 $V$ 上的[等价关系](https://en.wikipedia.org/wiki/Equivalence_relation)。

```{exercise-end}
```

```{solution-start} networks_ex1
:class: dropdown
```

**自反性：**
显然，$u = v \Rightarrow u \rightarrow v$。
因此，$u \sim u$。

**对称性：**
假设 $u \sim v$
$\Rightarrow u \rightarrow v$ 且 $v \rightarrow u$。
根据定义，这意味着 $v \sim u$。

**传递性：**
假设 $u \sim v$ 且 $v \sim w$
这意味着，$u \rightarrow v$ 且 $v \rightarrow u$，同时 $v \rightarrow w$ 且 $w \rightarrow v$。
因此，我们可以得出 $u \rightarrow v \rightarrow w$ 且 $w \rightarrow v \rightarrow u$。
这意味着 $u \sim w$。

```{solution-end}
```

```{exercise-start}
:label: networks_ex2
```

考虑一个有向图 $G$，其节点集为
$$
V = \{0,1,2,3,4,5,6,7\}
$$
边集为
$$
E = \{(0, 1), (0, 3), (1, 0), (2, 4), (3, 2), (3, 4), (3, 7), (4, 3), (5, 4), (5, 6), (6, 3), (6, 5), (7, 0)\}
$$

1. 使用 `Networkx` 绘制图 $G$。
2. 找出 $G$ 的相关邻接矩阵 $A$。
3. 使用上面定义的函数计算 $G$ 的入度中心性、出度中心性和特征向量中心性。

```{exercise-end}
```

```{solution-start} networks_ex2
:class: dropdown
```

```{code-cell} ipython3
# 首先，让我们绘制给定的图

G = nx.DiGraph()

G.add_nodes_from(np.arange(8))  # 添加节点

G.add_edges_from([(0,1),(0,3),       # 添加边
                  (1,0),
                  (2,4),
                  (3,2),(3,4),(3,7),
                  (4,3),
                  (5,4),(5,6),
                  (6,3),(6,5),
                  (7,0)])

nx.draw_networkx(G, pos=nx.circular_layout(G), node_color='gray', node_size=500, with_labels=True)

plt.show()
```

```{code-cell} ipython3
A = nx.to_numpy_array(G)      #求G的邻接矩阵

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

考虑一个有 $n$ 个节点和 $n \times n$ 邻接矩阵 $A$ 的图 $G$。

令 $S = \sum_{k=0}^{n-1} A^k$
我们可以说对于任意两个节点 $i$ 和 $j$，当且仅当 $S_{ij} > 0$ 时，$j$ 可从 $i$ 到达。

设计一个函数 `is_accessible`，用于检查给定图中的任意两个节点是否可达。

考虑 {ref}`networks_ex2` 中的图，并使用此函数检查

1. 从 $2$ 是否可以到达 $1$
2. 从 $3$ 是否可以到达 $6$

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

G.add_edges_from([(0,1),(0,3),       # 添加边
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
