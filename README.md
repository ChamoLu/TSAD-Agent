# TSAD Agent

TSAD Agent 是一个本地运行的**时序异常检测 + LLM 分析**系统。检测层负责产生异常分数、异常窗口和指标；LLM 负责把这些结果解释成可读的分析结论。


![image](https://github.com/ChamoLu/TSAD-Agent/blob/main/docs/assets/tsad-agent-workflow.svg)


当前版本：
- 检测层：`MtsCID`
- 示例数据集：`PSM`
- 后端：`FastAPI`
- 前端：本地 HTML 页面
- LLM：`DeepSeek API`

## 项目意义

传统 TSAD 方法通常只输出分数、阈值和标签，开发者还需要自己理解异常窗口、变量贡献和排查方向。TSAD Agent 在检测模型之上增加了一个统一的结果解释层，让用户可以在网页中完成检测、查看图表，并直接向 LLM 询问异常原因、变量贡献和排查建议。

本项目不替代 MtsCID、Anomaly Transformer、TranAD 等检测方法，而是为这些方法提供一个更易使用的交互和分析框架。

## 整体流程

```text
时序数据 CSV / PSM
        ↓
检测层：MtsCID 当前已实现，后续可扩展更多 TSAD 方法
        ↓
异常证据：分数、阈值、窗口、指标、变量贡献
        ↓
LLM 分析：解释结果、回答问题、给出排查建议
        ↓
本地网页：配置检测、查看图表、对话分析
```



## 目录结构

```text
.
├── MtsCID/                         # MtsCID 代码、数据和 checkpoint
├── app/
│   ├── main.py                     # FastAPI 入口
│   ├── services/                   # 检测、摘要、DeepSeek 调用
│   └── static/                     # 前端页面
├── prompts/
│   └── deepseek_tsad_system_prompt.md
├── docs/assets/
│   └── tsad-agent-workflow.svg
├── requirements.txt
└── README.md
```



## 快速运行

推荐使用 Python 3.8 到 3.10。Linux 服务器、macOS、WSL2 均可。

```bash
cd /path/to/TSAD

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

配置 DeepSeek：

```bash
export DEEPSEEK_API_KEY="你的 DeepSeek API Key"
```

启动服务：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

浏览器打开：

```text
http://localhost:8000
```

如果代码运行在远程服务器，推荐使用端口转发：

```bash
ssh -L 8000:127.0.0.1:8000 user@server_ip
```

然后仍然在本地浏览器访问：

```text
http://localhost:8000
```



## 页面使用

1. 在欢迎页点击“开始使用”。
2. 左侧选择检测方法、数据集和参数。
3. 点击“运行检测”。
4. 中间查看异常分数、指标和异常窗口。
5. 右侧选择 LLM 模型，并围绕检测结果提问。

使用案例：

```text
这个异常窗口为什么异常？
哪些变量贡献最大？
下一步应该怎么排查？
```



## 扩展新的 TSAD 方法

当前只实现了 `MtsCID`。如果要接入新方法，建议新增一个 detector service，并统一输出：

```python
{
    "scores": scores,
    "pred_labels": pred_labels,
    "true_labels": true_labels,
    "threshold": threshold,
    "metrics": metrics_dict,
    "variable_errors": variable_errors,
}
```

只要新方法能输出这些字段，就可以复用现有的前端展示和 LLM 分析逻辑。

## 注意事项

- LLM 的回答是基于模型证据的解释，不等同于真实业务根因。
- 当前默认数据集是 PSM，默认检测层是 MtsCID。
- 不要直接双击 HTML 文件，请通过 FastAPI 服务访问页面。
- 没有 GPU 也能运行，但推理会更慢。



## 致谢与引用

本项目的检测层基于 MtsCID，并参考了 TSAD 领域中许多优秀方法。感谢 MtsCID、Anomaly Transformer、TranAD、USAD 等工作对时序异常检测社区的贡献。

如果使用本项目中的 MtsCID 检测能力，请优先引用 MtsCID 原论文和原始仓库：

```bibtex
@article{xie2025multivariate,
  title={Multivariate Time Series Anomaly Detection by Capturing Coarse-Grained Intra-and Inter-Variate Dependencies},
  author={Xie, Yongzheng and Zhang, Hongyu and Babar, Muhammad Ali},
  journal={arXiv preprint arXiv:2501.16364},
  year={2025}
}
```

相关链接：

- MtsCID: [https://github.com/ilwoof/MtsCID](https://github.com/ilwoof/MtsCID)
- MtsCID arXiv: [https://arxiv.org/abs/2501.16364](https://arxiv.org/abs/2501.16364)
- Anomaly Transformer: [https://arxiv.org/abs/2110.02642](https://arxiv.org/abs/2110.02642)
- TranAD: [https://www.vldb.org/pvldb/vol15/p1201-tuli.pdf](https://www.vldb.org/pvldb/vol15/p1201-tuli.pdf)

