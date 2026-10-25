# Leaf 模型家族 / Leaf Model Family

**中文** | [English](#english)

四个自训练的极小语言模型，全部 **<0.1B 参数**（25.8M–29.5M）、**200K 上下文**、**64K 最大输出**，
在消费级 CPU（AMD Ryzen 5 7530U 核显笔记本，无独立显卡）上完成训练与推理，
通过一个纯 Python API 服务器提供 **OpenAI 兼容接口**。

| 模型 | 参数量 | 类型 | 特点 |
|---|---|---|---|
| **leaf1.0** | 25.8M | 直答 | 通用问答、两位数加减 100%、长文检索 92%、多轮对话 |
| **leaf-things1.0** | 25.8M | 思考 | 先输出推理过程（`reasoning_content`）再作答，竖式思维链 |
| **leaf-flash** | 29.5M | 多模态 | 图像理解（颜色/形状/数字识别）+ 思考，视觉前置分析架构 |
| **leaf-code** | 25.8M | 代码 | Python 函数编写/输出预测/代码解释/补全改错（模板级） |

中英双语。训练与推理均不依赖 GPU：CPU 单线程推理 20–96 tok/s（int8 量化权重仅 19–31 MB）。

---

## English

Four self-trained tiny language models, all **under 0.1B parameters** (25.8M–29.5M),
with **200K context** and **64K max output**, fully trained and inference-tested on a
consumer CPU laptop (AMD Ryzen 5 7530U iGPU, no discrete GPU), served through a pure-Python
**OpenAI-compatible API**.

| Model | Params | Type | Highlights |
|---|---|---|---|
| **leaf1.0** | 25.8M | Direct | General QA, 100% two-digit arithmetic, 92% long-context retrieval, multi-turn chat |
| **leaf-things1.0** | 25.8M | Thinking | Chain-of-thought in `reasoning_content` (vertical-form math reasoning) |
| **leaf-flash** | 29.5M | Multimodal | Image understanding (colors/shapes/digits) + thinking, vision-preprocessing architecture |
| **leaf-code** | 25.8M | Code | Python function writing / output prediction / code explanation / bugfix (template-level) |

Bilingual (Chinese/English). No GPU needed: single-thread CPU inference runs 20–96 tok/s
(INT8 quantized weights are only 19–31 MB).

---

## 模型文件 / Model Files (`gguf/`)

| 文件 / File | 大小 / Size | 说明 / Notes |
|---|---|---|
| `leaf1.0-q4_k_m.gguf` | 19.3 MB | 最小，老 CPU 首选 / smallest, best for old CPUs |
| `leaf1.0-q8_0.gguf` | 31.2 MB | 推荐 / recommended |
| `leaf1.0-f16.gguf` | 58.5 MB | 可用官方 llama-quantize 再量化 / requantizable |
| `leaf-things1.0-q8_0.gguf` | 31.2 MB | 思考模型 / thinking model |
| `leaf-flash-q8_0.gguf` | 31.2 MB | 多模态文本侧 / multimodal (text side) |
| `leaf-code-q8_0.gguf` | 31.2 MB | 代码模型 / code model |

全部为标准 GGUF（llama 架构），已在真实 llama.cpp (b11853) 上验证。

All are standard GGUF (llama architecture), verified on real llama.cpp (b11853).

## 快速开始 / Quick Start

### llama.cpp

```bash
llama-cli -m gguf/leaf1.0-q4_k_m.gguf -p "1+1=?" -n 100 -t 1
# OpenAI 兼容服务 / OpenAI-compatible server
llama-server -m gguf/leaf1.0-q4_k_m.gguf --host 127.0.0.1 --port 8080
```

官方 llama-bench 实测 / official llama-bench (Ryzen 5 7530U):
Q8_0 单线程 tg128 **508 tok/s**；Q4_K_M 单线程 **730 tok/s**。
2008 年的 Atom 单核性能按 1/10 折算仍有 **~70 tok/s**。

### Ollama

```bash
ollama create xiaoqiang -f Modelfile
ollama run xiaoqiang
```

### 纯 Python API 服务器 / Pure-Python API Server

```bash
pip install numpy tokenizers
python api_server.py            # 双模型/四模型 OpenAI 兼容 API
# 浏览器打开 http://127.0.0.1:8000/ 有可视化测试页
```

`models/` 目录已包含 int8 权重，开箱即用（需要 torch 时自动切换 torch 后端）。
The `models/` folder ships with INT8 weights — works out of the box
(automatically uses the torch backend if torch is installed).

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"leaf1.0","messages":[{"role":"user","content":"你好"}]}'
```

API 鉴权默认关闭；如需开启，把 `api_server.py` 中的 `AUTH_DISABLED` 改为 `False`
并在 `api.json` 里配置密钥（热加载）。
Auth is disabled by default; set `AUTH_DISABLED = False` and edit `api.json` to enable.

### 思考模型响应 / Thinking model response

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "reasoning_content": "个位5+6=11，写1进1；十位8+6+1=15。",
      "content": "85+66=151"
    }
  }]
}
```

---

## 训练过程 / Training Summary

全部在消费级 CPU/核显上完成（DirectML 加速 iGPU 训练），无云端、无独显。
All stages ran on consumer CPU/iGPU (DirectML for the AMD iGPU) — no cloud, no discrete GPU.

1. 下载基座 [MiniMind2-Small](https://huggingface.co/jingyaogong/MiniMind2-Small)（25.8M）
2. 多阶段中英 SFT：人设统一 / 英文增强 / 数学全覆盖（含进位借位）/ 长文检索 / 知识扩充 / 多轮对话
3. 专项阶段：数学记忆、开放式生成恢复、代码专项（leaf-code）、图像观察注入（leaf-flash）
4. 压缩：INT8（自研逐通道对称量化）与 GGUF Q8_0/Q4_K_M

方法论：小模型查表记忆需要「每个算式全覆盖 × 多问法」；混合批次会稀释记忆信号，
需要专项阶段 + 回放恢复；记忆型 SFT 保存最终权重而非最优验证权重。

Methodology: tiny models need full-coverage × multi-phrasing data for lookup memorization;
mixed batches dilute the signal (use specialized phases + replay); save final weights for
memorization-style SFT instead of best-validation checkpoints.

## 已知限制 / Known Limitations

- 0.026B–0.03B 的参数量决定能力边界：超出训练分布的任务会出错
  / Out-of-distribution tasks fail — these are template-level models
- 长上下文为机制容量（200K），训练长度 ≤1.6K，超长召回会退化
  / 200K context is mechanical; quality is best within ~1.6K tokens
- leaf-flash 的视觉能力针对合成图形域，真实照片不适用
  / leaf-flash vision targets the synthetic graphics domain, not natural photos

## 致谢 / Attribution

基座模型 / Base model: [MiniMind2-Small](https://huggingface.co/jingyaogong/MiniMind2-Small)
by [jingyaogong](https://github.com/jingyaogong/minimind) (Apache-2.0)。
本仓库的继续训练、量化与 API 实现基于其架构。/ Continued training, quantization and the
API implementation in this repo build on that architecture.

## 许可证 / License

Apache-2.0
