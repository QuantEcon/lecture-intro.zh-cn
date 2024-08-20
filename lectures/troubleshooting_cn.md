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

本页面专门为在运行课程代码时遇到错误的读者提供帮助。

## 修复你的本地环境

课程的基本假设是，只要满足以下条件，课程中的代码就应该可以执行：

1. 它在一个 Jupyter 笔记本中执行，并且
1. 笔记本在装有最新版本 Anaconda Python 的机器上运行。

你已经按照[这节课](https://python-programming.quantecon.org/getting_started.html)中的说明安装了 Anaconda，是吗？

假设你已经安装了，那么我们读者最常遇到的问题是他们的 Anaconda 版本不是最新的。

[这是一篇有用的文章](https://www.anaconda.com/blog/keeping-anaconda-date)
关于如何更新 Anaconda。

另一个选项是简单地移除 Anaconda 并重新安装。

你还需要保持外部代码库（例如 [QuantEcon.py](https://quantecon.org/quantecon-py)）的更新。

为此，你可以

* 在命令行中使用 conda install -y quantecon，或
* 在 Jupyter 笔记本中执行 !conda install -y quantecon。

如果你的本地环境仍然无法工作，你可以做两件事。

首先，你可以改用远程机器，通过点击每节课中提供的启动笔记本图标来实现。

```{image} _static/lecture_specific/troubleshooting/launch.png

```

其次，你可以报告问题，我们会尝试修复你的本地设置。

我们喜欢收到关于课程的反馈，所以请不要犹豫与我们联系。

## 报告问题

一种反馈方式是通过我们的[问题跟踪器](https://github.com/QuantEcon/lecture-python/issues)提出问题。

请尽量具体。告诉我们问题所在，以及尽可能多地提供有关你本地设置的详细信息。

另一种反馈选项是使用我们的[论坛](https://discourse.quantecon.org/)。

最后，你可以直接反馈到[contact@quantecon.org](mailto:contact@quantecon.org)。