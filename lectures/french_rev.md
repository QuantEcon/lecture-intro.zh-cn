---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# 法国大革命期间的通货膨胀


## 概览

本讲座描述了 {cite}`sargent_velde1995` 中所写的法国大革命（1789-1799）期间的一些货币和财政特征。

为了筹资公共开支和偿还债务，法国政府实验一系列政策。

这些政策的设计者对政府货币和财政政策如何影响经济结果有一定的理论认识。

他们所依据的一些理论至今仍然具有重要意义:

* **税收平滑**模型，例如罗伯特·巴罗提出的 {cite}`Barro1979`

* 这种规范性（即规定性）模式建议政府主要通过发行国债来为战时临时激增的支出提供资金，并增加税收来偿还战争期间发行的额外债务；然后，在战争结束后，将政府在战争期间积累的债务展期；并在战争结束后永久性地增加税收，增加的税收正好足够支付战后政府债务的利息。

* **不愉快的货币主义方程**类似于本讲中描述的方程 {doc}`unpleasant`
   
* 在 1789 年之前的几十年里，涉及复利的数学支配着法国政府的动态债务；据历史学家称，这种计算方式为法国大革命奠定了基础。

* 关于政府公开市场操作的影响的*真实票据*理论，其中政府用持有的有价值房地产或金融资产来*支持*新发行的纸币，纸币持有者可以用他们的钱从政府购买这些资产。
    
    * 革命者们从亚当·斯密于1776年出版的《国富论》{cite}`smith2010wealth` 和其他资料中了解到这一理论
    * 它塑造了革命者在1789年到1791年之间针对一种名为 [**指券（assignats）**](https://baike.baidu.com/item/%E6%8C%87%E5%88%B8/4594692) 的纸币的发行方式

* 经典的 **金本位** 或 **银本位**
  
    * 拿破仑·波拿巴于1799年成为法国政府首脑。他使用这一理论来指导他的货币和财政政策

* 经典的 **通货膨胀税** 理论，其中{doc}`cagan_ree`中研究的菲利普·凯根的货币需求({cite}`Cagan`) 是一个核心部分

   * 这一理论有助于解释1794年至1797年的法国价格水平和货币供应数据

* 实际余额需求的 **法律限制** 或 **金融抑制** 理论
 
    * 公共安全委员会的十二个成员，他们在恐怖时期即1793年6月至1794年7月当权，使用了这一理论来塑造他们的货币政策

我们使用 `matplotlib` 复现 {cite}`sargent_velde1995` 中用来描述这些实验结果的几个图表

## 数据来源

本讲使用了 {cite}`sargent_velde1995` 中汇编的三个表格中的数据：
  * [datasets/fig_3.xlsx](https://github.com/QuantEcon/lecture-python-intro/blob/main/lectures/datasets/fig_3.xlsx)
  * [datasets/dette.xlsx](https://github.com/QuantEcon/lecture-python-intro/blob/main/lectures/datasets/dette.xlsx)
  * [datasets/assignat.xlsx](https://github.com/QuantEcon/lecture-python-intro/blob/main/lectures/datasets/assignat.xlsx)

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']

base_url = 'https://github.com/QuantEcon/lecture-python-intro/raw/'\
           + 'main/lectures/datasets/'

fig_3_url = f'{base_url}fig_3.xlsx'
dette_url = f'{base_url}dette.xlsx'
assignat_url = f'{base_url}assignat.xlsx'
```

## 政府开支与税收

我们将使用 `matplotlib` 构建一些展示重要历史背景的图表。

我们将复现 {cite}`sargent_velde1995` 中的一些关键图表。这些图表揭示了十八世纪期间一些有趣的现象：

* 在四次大规模战争期间，法国和英国的政府开支都出现了大幅增长，且增长幅度相近
* 英国在和平时期的税收基本能够满足政府开支需求，但战时税收远低于开支水平
* 而法国的情况更为严峻 - 即便在和平时期，税收收入也远远无法覆盖政府开支

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "法国和英国的财政支出"
    name: fr_fig4
---
# 从Excel文件读取数据
data2 = pd.read_excel(dette_url, 
        sheet_name='Militspe', usecols='M:X', 
        skiprows=7, nrows=102, header=None)

# 法国军事开支，1685-1789年，以1726年的里弗尔计
data4 = pd.read_excel(dette_url, 
        sheet_name='Militspe', usecols='D', 
        skiprows=3, nrows=105, header=None).squeeze()
        
years = range(1685, 1790)

plt.figure()
plt.plot(years, data4, '*-', linewidth=0.8)

plt.plot(range(1689, 1791), data2.iloc[:, 4], linewidth=0.8)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().tick_params(labelsize=12)
plt.xlim([1689, 1790])
plt.xlabel('*：法国')
plt.ylabel('百万里弗')
plt.ylim([0, 475])

plt.tight_layout()
plt.show()
```

18 世纪，英国和法国进行了四次大规模战争。

英国赢得了前三场战争，输掉了第四场战争。

每次战争都导致两国政府支出激增，国家必须以某种方式为这些支出提供资金。

图{numref}`fr_fig4`显示了法国（蓝色）和英国在这四场战争中军费开支的激增。

图{numref}`fr_fig4`的一个显著特点是，尽管英国的人口不到法国的一半，但其军费开支却与法国差不多。

这证明英国已经建立了能够维持高税收、政府支出和政府借贷的国家机构。参见{cite}`north1989`。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "英国的政府支出和税收收入"
    name: fr_fig2
---
# 从Excel文件读取数据
data2 = pd.read_excel(dette_url, sheet_name='Militspe', usecols='M:X', 
                      skiprows=7, nrows=102, header=None)

# 绘制数据
plt.figure()
plt.plot(range(1689, 1791), data2.iloc[:, 5], linewidth=0.8)
plt.plot(range(1689, 1791), data2.iloc[:, 11], linewidth=0.8, color='red')
plt.plot(range(1689, 1791), data2.iloc[:, 9], linewidth=0.8, color='orange')
plt.plot(range(1689, 1791), data2.iloc[:, 8], 'o-', 
         markerfacecolor='none', linewidth=0.8, color='purple')

# 自定义图表
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().tick_params(labelsize=12)
plt.xlim([1689, 1790])
plt.ylabel('百万磅', fontsize=12)

# 添加文本注释
plt.text(1765, 1.5, '民用', fontsize=10)
plt.text(1760, 4.2, '民用加偿债', fontsize=10)
plt.text(1708, 15.5, '总政府开支', fontsize=10)
plt.text(1759, 7.3, '税收', fontsize=10)

plt.tight_layout()
plt.show()
```

图{numref}`fr_fig2`和{numref}`fr_fig3`总结了1789年法国大革命开始前一个世纪英国和法国政府的财政政策。

1789年之前，法国的进步力量非常欣赏英国为政府支出提供资金的方式，并希望重新设计法国的财政安排，使其更像英国。

图{numref}`fr_fig2`显示了政府支出及其在以下各项支出中的分配情况 

   * 民事（非军事）活动
   * 偿债，例如支付利息 
   * 军事支出（黄线减去红线） 

图{numref}`fr_fig2`还显示了政府从税收中获得的总收入（紫色圆圈线）

请注意，在这四场战争中，政府总支出的激增与军事支出的激增相关联

   * 18 世纪初反对法国国王路易十四的战争
   * 17 世纪 40 年代的奥地利王位继承战争
   * 17 世纪 50 年代和 60 年代的法印战争
   * 1775 年至 1783 年的美国独立战争

图{numref}`fr_fig2`显示

   * 在和平时期，政府收支基本平衡，债务负担保持稳定
   * 战争期间，政府支出大幅超过税收
      * 为了弥补这一赤字，政府不得不举债
   * 等到战争结束，政府的税收收入会略高于日常开支
      * 这些额外的税收收入刚好足够支付战时债务的利息
      * 政府并不会大幅增税来偿还本金
      * 而是选择继续滚动债务，只要能支付利息就行

因此，图{numref}`fr_fig2`中描绘的18世纪英国财政政策非常像罗伯特·巴罗 {cite}`Barro1979`等列举的有关*税收平滑*模型的例子。 

该图的一个显著特点被我们称为税收与政府支出之间的 “重力法则”。

   * 政府支出水平与税收水平相互吸引
   * 虽然它们会暂时出现差异（战争期间就是如此），但当恢复和平时，它们又会变为相似水平。

接下来，我们会将 18 世纪英国和法国的偿债成本占政府收入比例的数据绘制成图。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "英国和法国的偿债与税收比率"
    name: fr_fig1
---
# 从Excel文件读取数据
data1 = pd.read_excel(dette_url, sheet_name='Debt', 
            usecols='R:S', skiprows=5, nrows=99, header=None)
data1a = pd.read_excel(dette_url, sheet_name='Debt', 
            usecols='P', skiprows=89, nrows=15, header=None)

# 绘制数据
plt.figure()
plt.plot(range(1690, 1789), 100 * data1.iloc[:, 1], linewidth=0.8)

date = np.arange(1690, 1789)
index = (date < 1774) & (data1.iloc[:, 0] > 0)
plt.plot(date[index], 100 * data1[index].iloc[:, 0], 
         '*:', color='r', linewidth=0.8)

# 绘制附加数据
plt.plot(range(1774, 1789), 100 * data1a, '*:', color='orange')

# 标记数据
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_facecolor('white')
plt.gca().set_xlim([1688, 1788])
plt.ylabel('税收的百分比')

plt.tight_layout()
plt.show()
```

图{numref}`fr_fig1`显示，在英国和法国，政府债务的利息支出（即所谓的 “还本付息”）占政府税收收入的比例都很高。

{numref}`fr_fig2`向我们展示了在和平时期，尽管利息支出巨大，英国仍然能够平衡预算。

但正如我们在下一张图中看到的，在1788年法国大革命前夕，在英国行之有效的财政*重力法则*在法国并不奏效。

```{code-cell} ipython3
# 从 Excel 文件中读取数据
data1 = pd.read_excel(fig_3_url, sheet_name='Sheet1', 
          usecols='C:F', skiprows=5, nrows=30, header=None)

data1.replace(0, np.nan, inplace=True)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "法国的政府支出和税收收入"
    name: fr_fig3
---
# 绘制数据
plt.figure()

plt.plot(range(1759, 1789, 1), data1.iloc[:, 0], '-x', linewidth=0.8)
plt.plot(range(1759, 1789, 1), data1.iloc[:, 1], '--*', linewidth=0.8)
plt.plot(range(1759, 1789, 1), data1.iloc[:, 2], 
         '-o', linewidth=0.8, markerfacecolor='none')
plt.plot(range(1759, 1789, 1), data1.iloc[:, 3], '-*', linewidth=0.8)

plt.text(1775, 610, '总开支', fontsize=10)
plt.text(1773, 325, '军用', fontsize=10)
plt.text(1773, 220, '民用加偿债', fontsize=10)
plt.text(1773, 80, '偿债', fontsize=10)
plt.text(1785, 500, '收入', fontsize=10)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.ylim([0, 700])
plt.ylabel('百万里弗')

plt.tight_layout()
plt.show()
```

{numref}`fr_fig3`显示，在1788年法国大革命前夕，政府支出超过了税收收入。  

这种支出超过收入的情况在法国支持美国独立战争期间及之后尤为严重，主要是由于政府债务利息支出的持续增长。

这种债务动态的发展过程将在后续课程{doc}`unpleasant`中作为"令人不愉快的算术"进行详细讨论。

{cite}`sargent_velde1995`指出，统治法国至1788年的旧制度存在一些根深蒂固的制度特征，这些特征使政府难以实现预算平衡。

当时存在着强大的既得利益集团，他们阻碍了政府缩小支出和收入差距的任何尝试。具体而言，政府在以下三个方面都难以有所作为：

* 增加税收收入
* 削减非利息支出
* 通过债务重组或部分违约来降低利息负担

法国当时的制度安排使得三个群体能够阻止任何不利于他们的预算调整：

* 纳税人
* 政府支出的受益者
* 政府债权人（即政府债券持有人）

这种情况与1720年左右路易十四发动战争后的情况形成了鲜明对比。当时面对债务危机，政府选择了牺牲债权人的利益，通过拖欠债务来减少利息支出，从而平衡预算。

但到了1789年，政府债权人的力量明显增强。这迫使路易十六不得不召集三级会议，希望通过修改宪法来增加税收或削减开支，以实现预算平衡并减轻债务负担。

{cite}`sargent_velde1995`详细描述了法国大革命期间是如何应对这一挑战的。

## 教会资产的国有化与私有化

1789年，三级会议很快改组为国民议会。他们面临的首要任务就是解决财政危机 - 这也正是国王召集三级会议的原因。

值得注意的是，革命者们并非社会主义者或共产主义者。相反，他们尊重私有财产权，并且熟悉当时最先进的经济理论。

他们清楚地认识到，要偿还政府债务，要么增加收入，要么削减支出。

恰好的是，天主教会拥有大量创收资产。根据这些收入流的资本化价值估算，教会土地的价值几乎等于法国政府的全部债务。

这一巧合促成了一个三步走的债务偿还计划：

* 将教会土地收归国有
* 出售这些土地
* 用出售所得偿还政府债务

这一计划的理论基础来自亚当·斯密1776年出版的《国富论》{cite}`smith2010wealth`中关于"真实票据"的分析。斯密将"真实纸币"定义为以实物资产(如生产资本或存货)作为抵押的纸币。许多革命者都研读过这本书。

为实施这一计划，国民议会采取了一系列巧妙的制度安排。在无神论者主教塔列朗的提议下，议会首先将教会土地收归国有。

为了在不增加税收的情况下偿还债务，他们开始实施一项私有化计划。他们发行了一种名为"assignats"的纸币，持有者可以用它购买国有土地。由于这些纸币可以用来购买前教会土地，因此在某种程度上可以与银币等价。

财政部长内克尔和议会代表们希望通过这种新货币同时解决私有化和债务问题。他们设计了一个方案，通过拍卖没收的土地来筹集资金，并以政府出售土地为担保收回已发行的纸币。

这种"以税收支持的货币"方案将国民议会推向了当时货币理论的前沿。从辩论记录可以看出，议会成员们如何通过理论和历史经验来评估这一创新的影响：

* 他们引用了休谟和斯密的理论
* 他们借鉴了约翰·劳1720年的金融体系和美国近期的纸币经验教训
* 他们努力避免重蹈这些失败的覆辙

这个计划在最初两三年运行良好。但随后法国卷入了一场大规模战争，扰乱了计划的执行，从根本上改变了法国纸币的性质。{cite}`sargent_velde1995`对此有详细描述。

## 税收制度的改革

1789年，革命者们在国民议会中开始重塑法国的财政政策。由于债权人在议会中拥有强大影响力，他们希望履行政府的债务义务。

同时，他们着手改革税收制度和征税机制：

* 废除了多项税种
* 终结了古老的*税务承包*制度
  * 在这一制度下，政府将征税权私有化，让私人承包商征收税款并从中抽取佣金
  * 著名化学家拉瓦锡就是一位税务承包商，这也是他在1794年被送上断头台的原因之一

这些税收改革导致政府税收收入大幅下降，如下图所示。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "法国人均实际收入指数"
    name: fr_fig5
---
# 从Excel文件读取数据
data5 = pd.read_excel(dette_url, sheet_name='Debt', usecols='K', 
                    skiprows=41, nrows=120, header=None)

# 绘制数据
plt.figure()
plt.plot(range(1726, 1846), data5.iloc[:, 0], linewidth=0.8)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_facecolor('white')
plt.gca().tick_params(labelsize=12)
plt.xlim([1726, 1845])
plt.ylabel('1726 = 1', fontsize=12)

plt.tight_layout()
plt.show()
```

根据 {numref}`fr_fig5`，人均税收收入直到1815年，拿破仑·波拿巴被流放到圣赫勒拿岛并且路易十八恢复法国王位后，才得以回升至在1789年之前的水平。

* 从1799至1814年，拿破仑·波拿巴还有其他的收入来源—战利品和战争中打败的省份和国家支付的赔款
* 从1789至1799年，法国革命者寻求其他来源来筹集资源，以支付政府购买的货物和服务以及偿还法国政府债务。

如下图所示，在1789至1799年期间，政府开支大幅超过税收。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "支出（蓝色）和收入（橙色），（实际值）"
    name: fr_fig11
---
# 从Excel文件读取数据
data11 = pd.read_excel(assignat_url, sheet_name='Budgets',
        usecols='J:K', skiprows=22, nrows=52, header=None)

# 准备x轴数据
x_data = np.concatenate([
    np.arange(1791, 1794 + 8/12, 1/12),
    np.arange(1794 + 9/12, 1795 + 3/12, 1/12)
])

# 移除NaN数值
data11_clean = data11.dropna()

# 绘制数据
plt.figure()
h = plt.plot(x_data, data11_clean.values[:, 0], linewidth=0.8)
h = plt.plot(x_data, data11_clean.values[:, 1], '--', linewidth=0.8)

# 设置图表属性
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_facecolor('white')
plt.gca().tick_params(axis='both', which='major', labelsize=12)
plt.xlim([1791, 1795 + 3/12])
plt.xticks(np.arange(1791, 1796))
plt.yticks(np.arange(0, 201, 20))

# 设置y轴标签
plt.ylabel('百万里弗', fontsize=12)

plt.tight_layout()
plt.show()
```

面对这种财政困境，法国革命者通过多种方式弥补收支差额，其中一种方法是印发纸币并在市场上流通使用。

为了详细展示这一过程，我们下面将展示纸币的增发与各种商品和服务购买能力之间的关系，特别是军事物资和对士兵的支出。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "印刷纸币带来的收入"
    name: fr_fig24
---
# 从Ｅxcel中读取数据
data12 = pd.read_excel(assignat_url, sheet_name='seignor', 
         usecols='F', skiprows=6, nrows=75, header=None).squeeze()

# 创建图表并绘制
plt.figure()
plt.plot(pd.date_range(start='1790', periods=len(data12), freq='M'),
         data12, linewidth=0.8)


plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.axhline(y=472.42/12, color='r', linestyle=':')
plt.xticks(ticks=pd.date_range(start='1790', 
           end='1796', freq='AS'), labels=range(1790, 1797))
plt.xlim(pd.Timestamp('1791'),
         pd.Timestamp('1796-02') + pd.DateOffset(months=2))
plt.ylabel('百万里弗', fontsize=12)
plt.text(pd.Timestamp('1793-11'), 39.5, '1788年收入水平', 
         verticalalignment='top', fontsize=12)

plt.tight_layout()
plt.show()
```

{numref}`fr_fig24` 将 1789 年至 1796 年印钞所得的收入与古代政权在 1788 年获得的税收收入进行了比较。

以商品衡量，在 $t$ 时刻通过印制新钞所获得的收入等于

$$
\frac{M_{t+1} - M_t}{p_t}
$$

其中

* $M_t$ 是以里弗为单位的 在$t$ 时间的纸币存量
* $p_t$ 是在$t$ 时间每里弗的商品为单位的在 $t$ 时间的价格水平
* $M_{t+1} - M_t$ 是 在$t$ 时间内印制的新钞数量

从图中可以看到，1793-1794年期间印钞收入出现了显著的激增。这主要是由于公共安全委员会采取了一系列强制措施,要求公民必须接受纸币作为支付手段。

到了1797年前后，印钞收入开始急剧下降并最终停止。这标志着政府不再依赖印钞机来获取财政收入。

在这段时期，法国纸币持有者的地位和权利也在不断变化。这些变化产生了深远的影响，也从实践层面验证了当时革命者制定货币政策时所依据的理论。

接下来的图表展示了革命时期法国的物价水平变化。由于物价上涨幅度巨大，我们采用了对数刻度来更好地展示这一变化趋势。这个时期政府主要通过发行纸币来为各项支出融资。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "价格水平和黄金价格（对数标度）"
    name: fr_fig9
---
# 从Excel文件中读取数据
data7 = pd.read_excel(assignat_url, sheet_name='Data', 
          usecols='P:Q', skiprows=4, nrows=80, header=None)
data7a = pd.read_excel(assignat_url, sheet_name='Data', 
          usecols='L', skiprows=4, nrows=80, header=None)
# 创建图表并绘制
plt.figure()
x = np.arange(1789 + 10/12, 1796 + 5/12, 1/12)
h, = plt.plot(x, 1. / data7.iloc[:, 0], linestyle='--')
h, = plt.plot(x, 1. / data7.iloc[:, 1], color='r')

# 设置图表特征
plt.gca().tick_params(labelsize=12)
plt.yscale('log')
plt.xlim([1789 + 10/12, 1796 + 5/12])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 加入竖线
plt.axvline(x=1793 + 6.5/12, linestyle='-', linewidth=0.8, color='orange')
plt.axvline(x=1794 + 6.5/12, linestyle='-', linewidth=0.8, color='purple')

# 加入文字
plt.text(1793.75, 120, '“恐怖时期”', fontsize=12)
plt.text(1795, 2.8, '价格水平', fontsize=12)
plt.text(1794.9, 40, '黄金', fontsize=12)


plt.tight_layout()
plt.show()
```

我们将 {numref}`fr_fig9` 中的价格水平对数和 {numref}`fr_fig8` 中的实际余额 $\frac{M_t}{p_t}$ 划分为三个时期,对应不同的货币实验或制度。

第一个时期持续到1793年夏末，特点是实际余额稳步增长，通货膨胀温和。

第二个时期从恐怖时期开始，也以恐怖时期结束。这一时期的特点是实际余额维持在较高水平，约为25亿里弗，价格相对稳定。

第三个时期始于1794年7月下旬罗伯斯庇尔的倒台。在这一阶段，实际余额不断下降，价格快速上涨。

我们用三种不同的理论来解释这三个时期:

* *支持*或*真实票据*理论（该理论的经典演绎出自亚当·斯密{cite}`smith2010wealth`）
* 法律限制理论（{cite}`keynes1940pay`，{cite}`bryant1984price`）
* 经典恶性通货膨胀理论（{cite}`Cagan`）

```{注意}
根据{cite}`Cagan`采用的恶性通货膨胀的经验定义，
从通货膨胀率超过每月 50% 的月份开始，到通货膨胀率降至每月 50% 以下的月份结束，至少持续一年，*assignat* 从 1795 年 5 月到 12 月经历了恶性通货膨胀。
```

我们并不将这些理论视为竞争对手，而是将其视为关于政府票据发行的“如果-那么”的集合，每个理论都有其更接近现实条件的地方—即更接近满足”如果“的地方。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "转让的实际余额（用黄金和货币的形式）"
    name: fr_fig8
---
# 从Excel文件中读取数据
data7 = pd.read_excel(assignat_url, sheet_name='Data', 
        usecols='P:Q', skiprows=4, nrows=80, header=None)
data7a = pd.read_excel(assignat_url, sheet_name='Data', 
        usecols='L', skiprows=4, nrows=80, header=None)

# 创建图表并绘制
plt.figure()
h = plt.plot(pd.date_range(start='1789-11-01', periods=len(data7), freq='M'), 
            (data7a.values * [1, 1]) * data7.values, linewidth=1.)
plt.setp(h[1], linestyle='--', color='red')

plt.vlines([pd.Timestamp('1793-07-15'), pd.Timestamp('1793-07-15')], 
           0, 3000, linewidth=0.8, color='orange')
plt.vlines([pd.Timestamp('1794-07-15'), pd.Timestamp('1794-07-15')], 
           0, 3000, linewidth=0.8, color='purple')

plt.ylim([0, 3000])

# 设置图表属性
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_facecolor('white')
plt.gca().tick_params(labelsize=12)
plt.xlim(pd.Timestamp('1789-11-01'), pd.Timestamp('1796-06-01'))
plt.ylabel('百万里弗', fontsize=12)

# 添加文本注释
plt.text(pd.Timestamp('1793-09-01'), 200, '“恐怖时期”', fontsize=12)
plt.text(pd.Timestamp('1791-05-01'), 750, '黄金水平', fontsize=12)
plt.text(pd.Timestamp('1794-10-01'), 2500, '真实价值', fontsize=12)

plt.tight_layout()
plt.show()
```

图{numref}`fr_fig104`中的三个聚集点描绘了不同的实际余额-通货膨胀关系。

只有第三个时期的点具有我们现在熟悉的二十世纪恶性通货膨胀的逆向关系。

* 时期 1：（“*真实票据* 时期）：1791 年 1 月至 1793 年 7 月

* 时期 2：（“恐怖时期”）：1793 年 8 月 - 1794 年 7 月

* 时期 3：（“经典凯根恶性通货膨胀”）：1794 年 8 月 - 1796 年 3 月

```{code-cell} ipython3
def fit(x, y):

    b = np.cov(x, y)[0, 1] / np.var(x)
    a = y.mean() - b * x.mean()

    return a, b
```

```{code-cell} ipython3
# 加载数据
caron = np.load('datasets/caron.npy')
nom_balances = np.load('datasets/nom_balances.npy')

infl = np.concatenate(([np.nan], 
      -np.log(caron[1:63, 1] / caron[0:62, 1])))
bal = nom_balances[14:77, 1] * caron[:, 1] / 1000
```

```{code-cell} ipython3
# 分为三个时期将 y 对 x 进行回归
a1, b1 = fit(bal[1:31], infl[1:31])
a2, b2 = fit(bal[31:44], infl[31:44])
a3, b3 = fit(bal[44:63], infl[44:63])

# 分为三个时期将 y 对 x 进行回归
a1_rev, b1_rev = fit(infl[1:31], bal[1:31])
a2_rev, b2_rev = fit(infl[31:44], bal[31:44])
a3_rev, b3_rev = fit(infl[44:63], bal[44:63])
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "通货膨胀与实际余额"
    name: fr_fig104
---
plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 第一个子样本
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', 
         color='blue', label='真实票据时期')

# 第二个子样本
plt.plot(bal[31:44], infl[31:44], '+', color='red', label='恐怖时期')

# 第三个子样本
plt.plot(bal[44:63], infl[44:63], '*', 
        color='orange', label='经典凯根恶性通货膨胀')

plt.xlabel('实际余额')
plt.ylabel('通货膨胀')
plt.legend()

plt.tight_layout()
plt.show()
```

从 {numref}`fr_fig104` 中可以看出，三个不同时期的数据点呈现出截然不同的通货膨胀与实际余额之间的关系。

其中，只有第三个时期的数据点展现出了一个负相关关系 - 这与我们从20世纪的恶性通货膨胀案例中所熟悉的规律相符。

为了更清晰地展示这些关系，我们将对三个时期分别进行线性回归分析。

在此之前，我们先剔除恐怖时期初期的一些异常观测值，然后重新绘制散点图。

```{code-cell} ipython3
# 分为三个时期将 y 对 x 进行回归
a1, b1 = fit(bal[1:31], infl[1:31])
a2, b2 = fit(bal[31:44], infl[31:44])
a3, b3 = fit(bal[44:63], infl[44:63])

# 分为三个时期将 y 对 x 进行回归
a1_rev, b1_rev = fit(infl[1:31], bal[1:31])
a2_rev, b2_rev = fit(infl[31:44], bal[31:44])
a3_rev, b3_rev = fit(infl[44:63], bal[44:63])
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "通货膨胀与实际余额"
    name: fr_fig104b
---
plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 第一个子样本
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', color='blue', label='真实票据时期')

# 第二个子样本
plt.plot(bal[34:44], infl[34:44], '+', color='red', label='恐怖时期')

# 第三个子样本
plt.plot(bal[44:63], infl[44:63], '*', color='orange', label='经典凯根恶性通货膨胀')

plt.xlabel('实际余额')
plt.ylabel('通货膨胀')
plt.legend()

plt.tight_layout()
plt.show()
```

现在让我们把*真实票据*时期的通货膨胀对实际余额进行回归，并绘制回归线。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "通货膨胀与实际余额"
    name: fr_fig104c
---
plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 第一个子样本
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', 
        color='blue', label='真实票据时期')
plt.plot(bal[1:31], a1 + bal[1:31] * b1, color='blue')

# 第二个子样本
plt.plot(bal[31:44], infl[31:44], '+', color='red', label='恐怖时期')

# 第三个子样本
plt.plot(bal[44:63], infl[44:63], '*', 
        color='orange', label='经典凯根恶性通货膨胀')

plt.xlabel('实际余额')
plt.ylabel('通货膨胀')
plt.legend()

plt.tight_layout()
plt.show()
```

{numref}`fr_fig104c` 中的回归线反映了一个有趣的现象: 尽管政府大量发行纸币，但价格水平仅温和上涨。这与*真实票据*理论的预测相符。

这是因为在这一时期，政府发行的纸币实际上代表了对教会土地的所有权要求。人们相信这些纸币背后有实际资产作为支撑。

然而,到了这一时期的末尾,情况发生了变化。政府一方面继续印制纸币,另一方面却停止了教会土地的出售。这导致价格水平开始上涨,而实际余额反而下降。

为了维持纸币的流通,政府不得不采取强制措施,通过法律手段要求民众接受和持有这些纸币。

接下来,我们将分析恐怖时期的数据,对实际余额进行回归分析并绘制回归线。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "通货膨胀与实际余额"
    name: fr_fig104d
---
plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 第一个子样本
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', 
        color='blue', label='真实票据时期')

# 第二个子样本
plt.plot(bal[31:44], infl[31:44], '+', color='red', label='恐怖时期')
plt.plot(a2_rev + b2_rev * infl[31:44], infl[31:44], color='red')

# 第三个子样本
plt.plot(bal[44:63], infl[44:63], '*', 
        color='orange', label='经典凯根恶性通货膨胀')

plt.xlabel('实际余额')
plt.ylabel('通货膨胀')
plt.legend()

plt.tight_layout()
plt.show()
```

{numref}`fr_fig104d` 中的回归线揭示了一个有趣的现象：在恐怖时期，尽管政府大量增发纸币，但物价水平却出现了轻微上涨甚至下跌。

这表明在恐怖时期，政府通过严格的法律限制和财政压制手段，成功地维持了纸币的价值。

然而，随着恐怖统治在1794年7月结束，这种强制措施也随之瓦解。人们开始寻求其他方式来进行交易和储存价值，导致了严重的通货膨胀。

接下来的两张图表展示了这一经典的恶性通货膨胀时期的情况。

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "通货膨胀和实际余额"
    name: fr_fig104e
---
plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 第一个子样本
plt.plot(bal[1:31], infl[1:31], 'o', markerfacecolor='none', 
        color='blue', label='真实票据时期')

# 第二个子样本
plt.plot(bal[31:44], infl[31:44], '+', color='red', label='恐怖时期')

# 第三个子样本
plt.plot(bal[44:63], infl[44:63], '*', 
    color='orange', label='经典凯根恶性通货膨胀')
plt.plot(bal[44:63], a3 + bal[44:63] * b3, color='orange')
plt.xlabel('实际余额')
plt.ylabel('通货膨胀')
plt.legend()
plt.tight_layout()
plt.show()
```

上面两张图分别从不同角度展示了通货膨胀与实际余额之间的关系。一张图以实际余额作为自变量，通货膨胀作为因变量；另一张图则反过来，以通货膨胀作为自变量，实际余额作为因变量。

从两张图中都可以看出它们之间存在明显的负相关关系。这种负相关正是凯根 {cite}`Cagan`在研究恶性通货膨胀时发现的典型特征。


```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "通货膨胀与实际余额"
    name: fr_fig104f
---
plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 第一个子样本
plt.plot(bal[1:31], infl[1:31], 'o', 
    markerfacecolor='none', color='blue', label='真实票据时期')

# 第二个子样本
plt.plot(bal[31:44], infl[31:44], '+', color='red', label='恐怖时期')

# 第三个子样本
plt.plot(bal[44:63], infl[44:63], '*', 
        color='orange', label='经典凯根恶性通货膨胀')
plt.plot(a3_rev + b3_rev * infl[44:63], infl[44:63], color='orange')

plt.xlabel('实际余额')
plt.ylabel('通货膨胀')
plt.legend()

plt.tight_layout()
plt.show()
```

{numref}`fr_fig104e`展示了在恶性通货膨胀期间，通货膨胀率与实际货币余额之间的回归关系。

## 恶性通货膨胀的终结

根据{cite}`sargent_velde1995`的记载，1797年法国革命政府采取了一系列措施来终结通货膨胀：

  * 宣布2/3的国家债务无效
  * 由此消除了政府的利息赤字负担
  * 停止印制纸币
  * 改用金银币作为流通货币

1799年，拿破仑·波拿巴就任第一执政。在随后的15年里，他主要依靠从征服地区掠夺的资源来维持法国政府的开支。

## 理论启示

本讲为我们研究通货膨胀理论及其背后的政府货币和财政政策奠定了基础。

我们介绍了一个*货币主义的物价水平理论*，这个理论将在{doc}`cagan_ree`中得到深入探讨。

同时，本讲的内容也为后续的{doc}`money_inflation`和{doc}`unpleasant`两章做了必要的铺垫。
