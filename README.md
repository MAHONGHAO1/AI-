# AI 智选基金助手

> 基于 Streamlit 和通义千问的个人基金投资分析系统  
> 自动抓取净值数据，多维度量化评分，AI 生成投资建议，一站式交互看板

## ✨ 主要功能

- 📈 **自动数据抓取**：支持 AkShare / 东方财富双数据源，自动重试与降级（模拟数据兜底）
- 🧮 **多维度评分模型**：参考晨星体系，五维度加权（收益能力30%、风险控制25%、风险调整收益25%、稳定性10%、相对排名10%），输出 0–10 分与晨星评级
- 🤖 **AI 投资建议**：调用通义千问大模型（兼容 OpenAI API），同时生成简短建议与结构化详细分析（趋势判断、操作建议、风险提示、适合人群），API 不可用时自动降级为本地规则
- 📊 **交互看板**：Streamlit 构建，支持基金筛选、收益-回撤散点图、五维雷达图、定投回测模拟器
- 📄 **报告导出**：一键导出 Markdown 文本报告与 HTML 可视化报告

## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone (https://github.com/MAHONGHAO1/AI-/new/main?filename=README.md)
cd AI-Fund-Assistant
```

### 2. 安装依赖（自动安装，也可手动）
```bash
pip install -r requirements.txt
```
或使用清华镜像加速：
```bash
pip install streamlit pandas numpy requests plotly openai lxml tabulate akshare -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 运行应用
```bash
streamlit run ai_investment_advisor.py
```

浏览器将自动打开 `http://localhost:8501`。

### 4. （可选）配置通义千问 API Key
- 在侧边栏「设置」中输入你的 `DASHSCOPE_API_KEY`，即可启用真实 AI 建议
- 不配置时自动使用本地 mock 建议，不影响核心功能

## 🧱 技术栈

- **前端/交互**：Streamlit, Plotly
- **数据处理**：Pandas, NumPy, Requests
- **数据源**：AkShare, 东方财富公开接口
- **AI 模型**：通义千问（qwen-plus / qwen-max），兼容 OpenAI SDK
- **语言**：Python 3.14

## 📂 文件结构

```
.
├── ai_investment_advisor.py   # 主程序（含所有功能）
├── requirements.txt           # 依赖列表
├── README.md                  # 本文件
└── output/                    # 生成的报告默认保存在桌面
```

## 📝 注意事项

- 数据均来自公开接口，仅供个人学习与研究，不构成投资建议。
- 首次运行会自动安装依赖，请保持网络畅通。
- 部分基金接口有访问频率限制，代码已内置重试与降级机制。
- 如需长期使用，建议申请自己的通义千问 API Key（免费额度足够个人使用）。

## 🛠️ 未来计划

- [ ] 支持更多基金数据源（天天基金、蛋卷等）
- [ ] 增加组合优化建议（基于均值-方差模型）
- [ ] 支持多账户持仓对比
- [ ] 部署到 Streamlit Cloud 提供在线演示



## 👤 作者

麻洪豪  
- 邮箱：3438519642@qq.com  
## 🙏 致谢

- [AkShare](https://www.akshare.xyz/) 提供金融数据接口
- [Streamlit](https://streamlit.io/) 提供便捷的 Web 应用框架
- 通义千问提供大模型能力
```
