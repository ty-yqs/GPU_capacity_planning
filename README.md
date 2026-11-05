# GPU 容量规划

深圳理工大学“概率论与数理统计”探究项目。根据连续 14 天的分钟请求数和文本、图像请求的 GPU 服务时间，估计工作日高峰（17:00–21:59）的一分钟超载概率，并确定满足 1% 服务目标的 GPU 配置。

开源仓库：[ty-yqs/GPU_capacity_planning](https://github.com/ty-yqs/GPU_capacity_planning)

## 主要结论

每块 GPU 按 80% 利用率规划，有效容量为每分钟 48 GPU 秒。以下结果使用 A01 数据、成对抽取的高峰分钟计数、经验服务时间分布和每情景 60,000 次模拟。

| 情景 | 最少 GPU | 临界配置的一分钟超载率 |
| --- | ---: | ---: |
| 当前工作日高峰 | 5 | 0.872% |
| 文本和图像到达规模均增长 30% | 7 | 0.317% |
| 总请求数不变、图像占比升至 35% | 11 | 0.805% |
| 敏感性测试：5% 的分钟，两类到达均额外增长 50% | 6 | 0.470% |

**建议现在部署 6 块 GPU。** 汇总数据下 5 块刚好满足目标，但按前 5 个工作日单独模拟时，其超载率为 1.182%，已超过 1%；此外，经验数据不能覆盖未出现的极端服务时间。若两类到达规模均增长 30%，应在增长发生前扩容至至少 7 块。图像占比达到 35% 时，仅增加 GPU 虽可达标，也应同时评估图像请求的预热、批处理、模型压缩和分级路由。

这些是基于给定数据和模型的估计，并非未来超载率的保证。论文说明了泊松假设检验、冷启动对右尾的影响、正态近似误差及关键假设。

## 文件

| 路径 | 内容 |
| --- | --- |
| `GPU_capacity_planning.pdf` | 课程任务书 |
| `request_count.csv` | 20,160 个连续分钟的文本与图像请求数 |
| `service_time.csv` | 4,000 条文本、1,600 条图像请求的 GPU 服务时间 |
| `experiments/analyze.py` | 数据分析、模拟及论文图表数据生成脚本 |
| `experiments/results.json` | 固定种子的分析和模拟结果 |
| `latex/main.tex` | 论文 LaTeX 源码 |
| `latex/fig_data/` | 编译论文所需的箱线图与直方图数据 |
| `latex/main.pdf` | 编译后的论文 |

## 编译论文

安装带有 XeLaTeX、`latexmk`、`ctex`、`pgfplots` 和 TeX Gyre 字体的 TeX Live / MacTeX，然后在本项目目录执行：

```bash
cd latex
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

已有的 `latex/fig_data/` 包含编译所需数据；**只编译论文无需运行 Python，也无需网络连接**。论文采用 Fandol 中文字体，作者信息、页眉页脚和其他版式与同课程的 `naive_bayes/latex/main.tex` 对齐。

## 复现实验

需要 Python 3、NumPy 和 pandas。在本项目目录执行：

```bash
python3 -m pip install numpy pandas
python3 experiments/analyze.py
```

脚本读取根目录两份 CSV，更新 `experiments/results.json` 和 `latex/fig_data/`。随机种子为 `20261105`；各模拟情景分别运行 60,000 次，每批 400 次。脚本限制 NumPy 底层计算库为单线程，避免占用过多系统资源。统计检验的泊松模拟另为每类 2,000 次。

主模拟将同一分钟的文本、图像请求数成对抽样，保留其相关性和过度离散，再分别有放回抽取服务时间。增长 30% 的情景对两类计数添加条件均值为原计数 30% 的泊松增量；图像占比 35% 的情景在保持实测总请求数分布的条件下进行二项分配。服务时间样本没有分钟时间戳，模型假设其能代表高峰请求；论文把这一点列为最需要后续核验的假设。
