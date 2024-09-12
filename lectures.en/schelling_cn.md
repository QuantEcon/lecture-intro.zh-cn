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

(schelling)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 种族隔离

```{index} single: Schelling隔离模型
```

```{index} single: 模型; Schelling的隔离模型
```

```{admonition} 迁移课程
:class: warning

这节课程已经从我们的[中级定量经济学与Python](https://python.quantecon.org/intro.html)系列讲座迁移，现在是[定量经济学入门课程](https://intro.quantecon.org/intro.html)的一部分。
```

## 概述

1969年，托马斯·C·谢林开发了一个简单但引人注目的种族隔离模型{cite}`Schelling1969`。

他的模型研究了种族混合社区的动态。

像谢林的大多数工作一样，该模型展示了局部互动如何导致令人惊讶的总体结果。

它研究了一个设定，即代理（可以理解为家庭）对同种族邻居有相对温和的偏好。

例如，这些代理人可能会对种族混合的社区感到舒适，但当他们感到被不同种族的人包围时会感到不舒服。

谢林展示了以下令人惊讶的结果：在这种设置下，种族混合的社区可能是不稳定的，随着时间的推移趋向于崩溃。

实际上，该模型预测了强烈分隔的社区，具有高水平的隔离。

换句话说，即使人们的偏好并不是特别极端，也会出现极端的隔离结果。

这些极端结果的发生是由于模型中代理（例如城市中的家庭）之间的*互动*，这些互动在模型中驱动了自我增强的动态。

随着讲座的展开，这些想法会变得更加清晰。

为了表彰他在隔离和其他研究方面的工作，谢林获得了2005年的诺贝尔经济学奖（与罗伯特·奥曼共同获奖）。


让我们从一些导入开始：

```{code-cell} ipython3
%matplotlib inline
import matplotlib.pyplot as plt
from random import uniform, seed
from math import sqrt
import numpy as np
```
图像输入功能：已启用

## 模型

在本节中，我们将构建一个谢林模型的版本。

### 设置

我们将介绍一个与谢林模型原版不同的变体，但也容易编程且同时捕捉到其主要思想。

假设我们有两种类型的人：橙色人和绿色人。

假设每种类型的人数为 $n$。

这些代理人都生活在一个单位正方形中。

因此，代理人的位置（例如地址）只是一个点 $(x, y)$， 其中 $0 < x, y < 1$。

* 满足 $0 < x, y < 1$ 的所有点 $(x,y)$ 的集合称为**单位正方形**
* 下面我们用 $S$ 表示单位正方形

+++

### 偏好

我们会说，如果一个代理人的10个最近邻居中有5个或更多与她是同一类型，她是*开心*的。

不开心的代理人称为*不开心*。

例如，

* 如果一个代理人是橙色的，她的10个最近邻居中有5个是橙色的，则她是开心的。
* 如果一个代理人是绿色的，她的10个最近邻居中有8个是橙色的，则她是不开心的。

“最近”是指[欧几里得距离](https://en.wikipedia.org/wiki/Euclidean_distance)。

一个重要的点是，代理人并不厌恶混合区域的生活。

如果一半的邻居是不同颜色的，他们也会非常开心。

+++

### 行为

最初，代理人是混合在一起的（综合的）。

特别是，我们假设每个代理人的初始位置是单位正方形 $S$ 上双变量均匀分布的独立抽样。

* 首先，他们的 $x$ 坐标从 $(0,1)$ 的均匀分布中抽取
* 然后，独立地，他们的 $y$ 坐标从同一分布中抽取。

现在，循环通过所有代理人集合，每个代理人现在有机会留下或搬家。

如果代理人是开心的，他们会留下；如果不开心，他们会搬家。

搬家的算法如下：

```{prf:algorithm} 跳跃链算法
:label: move_algo

1. 在 $S$ 中抽取一个随机位置
1. 如果在新位置开心，则搬到那里
1. 否则，返回第1步

```

我们不断循环通过代理人，每次都允许一个不开心的代理人搬家。

我们继续循环直到没有人愿意搬家。

+++

## 结果

现在让我们实现并运行这个模拟。

在下面的内容中，代理人被建模为[对象](https://python-programming.quantecon.org/python_oop.html)。

以下是它们结构的指示：

```{code-block} none
* 数据:

    * 类型（绿色或橙色）
    * 位置

* 方法:

    * 根据其他代理人的位置确定是否开心
    * 如果不开心，就搬家
        * 找到一个让自己开心的新位置
```

让我们构建它们。

```{code-cell} ipython3
class Agent:

    def __init__(self, type):
        self.type = type
        self.draw_location()

    def draw_location(self):
        self.location = uniform(0, 1), uniform(0, 1)

    def get_distance(self, other):
        "计算自己与其他代理人之间的欧几里得距离。"
        a = (self.location[0] - other.location[0])**2
        b = (self.location[1] - other.location[1])**2
        return sqrt(a + b)

    def happy(self,
                agents,                # 其他代理人的列表
                num_neighbors=10,      # 视为邻居的代理人数
                require_same_type=5):  # 多少个邻居必须是同一类型
        """
            如果足够数量的最近邻居是同一类型，则返回True。
        """

        distances = []

        # distances 是一个对 (d, agent) 的列表，其中 d 是 agent 到自己的距离
        for agent in agents:
            if self != agent:
                distance = self.get_distance(agent)
                distances.append((distance, agent))

        # 按距离从小到大排序
        distances.sort()

        # 提取相邻的代理人
        neighbors = [agent for d, agent in distances[:num_neighbors]]

        # 统计有多少邻居与自己是同一类型
        num_same_type = sum(self.type == agent.type for agent in neighbors)
        return num_same_type >= require_same_type

    def update(self, agents):
        "如果不开心，则随机选择新位置直到开心。"
        while not self.happy(agents):
            self.draw_location()
```

只有一个核心函数还需要完成。

这是一个函数，给定一组代理人，将运行代理人搬家的循环直到所有代理人都开心。

我们将其定义如下

```{code-cell} ipython3
def simulateScheling(n=500):
    """
    运行谢林模型的模拟。

    参数:
        n (int): 每种类型的代理数量。
    """
    
    agents = [Agent(type=0) for i in range(n)] + [Agent(type=1) for i in range(n)]

    cycle_num = 1
    unhappy_found = True

    while unhappy_found:
        unhappy_found = False
        for agent in agents:
            if not agent.happy(agents):
                agent.update(agents)
                unhappy_found = True
        cycle_num += 1

    return agents, cycle_num
```

## 可视化

我们剩下的任务是查看模拟的结果。

我们将每种代理类型的位置画在单位正方形上

橙色代理用橙色点表示，绿色代理用绿色点表示。

```{code-cell} ipython3
def plot_distribution(agents, cycle_num):
    "在cycle_num轮循环后绘制代理人的分布。"
    x_values_0, y_values_0 = [], []
    x_values_1, y_values_1 = [], []
    # == 获取每种类型的位置 == #
    for agent in agents:
        x, y = agent.location
        if agent.type == 0:
            x_values_0.append(x)
            y_values_0.append(y)
        else:
            x_values_1.append(x)
            y_values_1.append(y)
    fig, ax = plt.subplots()
    plot_args = {'markersize': 8, 'alpha': 0.8}
    ax.set_facecolor('azure')
    ax.plot(x_values_0, y_values_0,
        'o', markerfacecolor='orange', **plot_args)
    ax.plot(x_values_1, y_values_1,
        'o', markerfacecolor='green', **plot_args)
    ax.set_title(f'Cycle {cycle_num-1}')
    plt.show()
```

例如，我们来看一下 $n=500$ 时的情况

```{code-cell} ipython3
agents, cycle_num = simulateScheling(n=500)
plot_distribution(agents, cycle_num)
```

生成的图像显示了代理人的最终位置分布。

## 总结

我们刚刚实现并展示了一个谢林种族隔离模型的版本。

结果表明，根据非常温和的偏好，种族混合的社区可能是不稳定的，趋向于极端的隔离。

### 主要循环的伪代码以及实现

以下是主要循环的伪代码，其中我们循环遍历所有代理，直到没有人愿意移动。

伪代码如下：

```{code-block} none
plot the distribution
while agents are still moving
    for agent in agents
        give agent the opportunity to move
plot the distribution
```

实际代码如下：

```{code-cell} ipython3
def run_simulation(num_of_type_0=600,
                   num_of_type_1=600,
                   max_iter=100_000,       # 最大迭代次数
                   set_seed=1234):

    # 设置随机种子以保证可重复性
    seed(set_seed)

    # 创建类型0的代理人列表
    agents = [Agent(0) for i in range(num_of_type_0)]
    # 追加类型1的代理人列表
    agents.extend(Agent(1) for i in range(num_of_type_1))

    # 初始化计数器
    count = 1

    # 绘制初始分布
    plot_distribution(agents, count)

    # 循环直到没有代理人愿意移动
    while count < max_iter:
        print('进入循环 ', count)
        count += 1
        no_one_moved = True
        for agent in agents:
            old_location = agent.location
            agent.update(agents)
            if agent.location != old_location:
                no_one_moved = False
        if no_one_moved:
            break

    # 绘制最终分布
    plot_distribution(agents, count)

    if count < max_iter:
        print(f'在 {count} 次迭代后收敛。')
    else:
        print('到达迭代上限并终止。')
```

这个模拟展示了谢林模型的基本思想，即即使代理人的偏好和行为在个体层面上是温和的，最终结果可能会显示出极端的种族隔离模式。

### 运行模拟

让我们执行定义的函数 `run_simulation` 并检查结果：

```{code-cell} ipython3
run_simulation()
```
图像输入功能：已启用

如上所述，代理人最初是随机混合在一起的。

但在经过若干个周期后，它们被隔离到了不同的区域。

在这个实例中，程序在经过少量周期后终止，表明所有代理人都达到了一个满意的状态。

图中值得注意的是种族整合的破裂速度。

尽管模型中的人实际上并不介意与其他类型的人混合生活。

即使有这些偏好，结果也是高度隔离的。

## 练习

```{exercise-start}
:label: schelling_ex1
```

我们上面使用的面向对象风格编码整洁但较难比过程代码（即围绕函数而不是对象和方法的代码）优化。

尝试编写一个新版本的模型，该模型存储以下数据：

* 所有代理人位置的二维 NumPy 浮点数组。
* 所有代理人的类型的平面 NumPy 整数数组。

编写作用于这些数据的函数，使用与上述类似的逻辑更新模型。

但实施以下两个更改：

1. 代理人被随机提供一个移动机会（即随机选择并给予移动机会）。
2. 在代理人移动后，以0.01的概率翻转他们的类型。

第二个更改在模型中引入了额外的随机性。

（我们可以设想偶尔一个代理人搬到一个不同的城市，并且有小概率被其他类型的代理人取代。）

```{exercise-end}
```

```{solution-start} schelling_ex1
:class: dropdown
```
solution here

```{code-cell} ipython3
from numpy.random import uniform, randint

n = 1000                # 代理人数 (代理 = 0, ..., n-1)
k = 10                  # 被视为邻居的代理人数
require_same_type = 5   # 需要有>=require_same_type相同类型的邻居

def initialize_state():
    locations = uniform(size=(n, 2))
    types = randint(0, high=2, size=n)   # 标签为0或1
    return locations, types


def compute_distances_from_loc(loc, locations):
    """ 计算从位置 loc 到所有其他点的距离。 """
    return np.linalg.norm(loc - locations, axis=1)

def get_neighbors(loc, locations):
    " 获取给定位置的所有邻居。 "
    all_distances = compute_distances_from_loc(loc, locations)
    indices = np.argsort(all_distances)   # 按与loc的距离对代理进行排序
    neighbors = indices[:k]               # 保留最近的k个
    return neighbors

def is_happy(i, locations, types):
    happy = True
    agent_loc = locations[i, :]
    agent_type = types[i]
    neighbors = get_neighbors(agent_loc, locations)
    neighbor_types = types[neighbors]
    if sum(neighbor_types == agent_type) < require_same_type:
        happy = False
    return happy

def count_happy(locations, types):
    " 统计开心代理的数量。 "
    happy_sum = 0
    for i in range(n):
        happy_sum += is_happy(i, locations, types)
    return happy_sum

def update_agent(i, locations, types):
    " 如果代理人不开心，则移动。 "
    moved = False
    while not is_happy(i, locations, types):
        moved = True
        locations[i, :] = uniform(), uniform()
    return moved

def plot_distribution(locations, types, title, savepdf=False):
    " 在循环轮数为 cycle_num 后绘制代理人的分布。"
    fig, ax = plt.subplots()
    colors = 'orange', 'green'
    for agent_type, color in zip((0, 1), colors):
        idx = (types == agent_type)
        ax.plot(locations[idx, 0],
                locations[idx, 1],
                'o',
                markersize=8,
                markerfacecolor=color,
                alpha=0.8)
    ax.set_title(title)
    plt.show()

def sim_random_select(max_iter=100_000, flip_prob=0.01, test_freq=10_000):
    """
    通过在每次更新时随机选择一个家庭进行模拟。

    以 `flip_prob` 的概率翻转家庭的颜色。
    """
    locations, types = initialize_state()
    current_iter = 0

    while current_iter <= max_iter:
        # 选择一个随机代理并更新他们
        i = randint(0, n)
        moved = update_agent(i, locations, types)

        if flip_prob > 0:
            # 以概率 epsilon 翻转代理i的类型
            U = uniform()
            if U < flip_prob:
                current_type = types[i]
                types[i] = 0 if current_type == 1 else 1

        # 每隔一定数量的更新，绘制并测试收敛性
        if current_iter % test_freq == 0:
            cycle = current_iter / n
            plot_distribution(locations, types, f'iteration {current_iter}')
            if count_happy(locations, types) == n:
                print(f"在迭代 {current_iter} 时收敛")
                break

        current_iter += 1

    if current_iter > max_iter:
        print(f"在迭代 {current_iter} 时终止")
```

```{solution-end}
```

当我们运行这个程序时，我们再次发现混合社区瓦解，隔离情况出现。

这是一个示例运行。

```{code-cell} ipython3
sim_random_select(max_iter=50_000, flip_prob=0.01, test_freq=10_000)
```

```{code-cell} ipython3

```

图像输入功能：已启用