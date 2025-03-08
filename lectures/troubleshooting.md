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

(troubleshooting)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 故障排除

本页面旨在帮助读者解决执行讲座代码时遇到的错误。

## 修复您的本地环境

讲座的基本假设是：只要

1. 在Jupyter笔记本中执行代码，并且
1. 笔记本在安装了最新版Anaconda Python的机器上运行。

按照[这节课](https://python-programming.quantecon.org/getting_started.html)中的指引，你应该已经安装了Anaconda了吧？ 

假设你已经安装了，我们读者最常见的问题是他们的Anaconda发行版不是最新的。

[这里有一篇挺有用的文章](https://www.anaconda.com/blog/keeping-anaconda-date)
描述了如何更新Anaconda。

另一个选项则是卸载Anaconda并重新安装。

你还需要更新外部代码库，例如[QuantEcon.py](https://quantecon.org/quantecon-py)。

为此，你可以

* 在命令行中使用 conda install -y quantecon，或
* 在Jupyter笔记本中执行 !conda install -y quantecon。

如果您的本地环境依然无法工作，你可以做两件事。

首先，你可以通过点击每个教程提供的启动笔记本图标改用云端运行

```{image} _static/lecture_specific/troubleshooting/launch.png

```

其次，你可以开一个[`Issue`](https://github.com/QuantEcon/lecture-intro.zh-cn/issues)，
这样我们可以一起帮忙尝试修复你的本地设置。

我们都很喜欢收到课程的反馈，所以请随时与我们联系！

## 报告问题

上文提到，其中一种联系我们的方式是通过我们的[问题跟踪器](https://github.com/QuantEcon/lecture-intro.zh-cn/issues)提出问题。

请尽可能具体。告诉我们问题所在，以及尽可能多的关于你本地设置的细节（比如环境变量、操作系统、Python版本等）。

如果你能提供一个最小的可重现代码片段，那就更好了。

最后，您可以直接向[contact@quantecon.org](mailto:contact@quantecon.org)提供反馈。