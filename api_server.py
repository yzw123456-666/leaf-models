# -*- coding: utf-8 -*-
"""Leaf API 服务器 —— 双模型 + API Key 鉴权 + OpenAI 兼容。

模型:
  leaf1.0        直答模型 (原小强 v4)
  leaf-things1.0 思考模型 (生成 <think>推理</think> 后作答, API 拆分为
                 reasoning_content + content 两个字段返回)

鉴权: 只有 scripts/api.json 中列出的 key 可通行 (Bearer Token),
     文件可随时手工编辑, 服务器热加载立即生效。/health 无需鉴权。

用法:
  python scripts/api_server.py                     # 双模型, 0.0.0.0:8000
  python scripts/api_server.py --port 9000

请求示例:
  curl http://127.0.0.1:8000/v1/chat/completions \
       -H "Authorization: Bearer leaf-sk-123456" -H "Content-Type: application/json" \
       -d '{"model":"leaf-things1.0","messages":[{"role":"user","content":"17+26=?"}]}'
"""
import argparse
import io
import json
import os
import re
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))          # 本文件所在目录
ROOT = os.path.dirname(BASE)                                # 项目根 (项目模式)
API_JSON = os.path.join(BASE, "api.json")
sys.path.insert(0, BASE)  # 独立部署模式: 同级目录可直接导入引擎模块

MAX_SEQ = 204800       # 200K 上下文
MAX_OUT = 65536        # 64K 最大输出
DEFAULT_MAX_TOKENS = 512

# 模型注册表: id -> (权重目录, 是否思考模型, 别名)
# 双模式: 本目录下存在 models/ 即为独立部署 (权重直接放模型目录), 否则用项目 exports/
if os.path.isdir(os.path.join(BASE, "models")):
    _M = lambda *p: os.path.join(BASE, "models", *p)
    MODELS_SPEC = {
        "leaf1.0": {"dir": _M("leaf1.0"), "npz": _M("leaf1.0", "weights_int8.npz"),
                    "think": False, "aliases": ["xiaoqiang-26m", "tinyllm-26m", "mini", "小强", "小微",
                                "leaf-1.0", "Leaf-1.0", "leaf 1.0"]},
        "leaf-things1.0": {"dir": _M("leaf-things1.0"),
                           "npz": _M("leaf-things1.0", "weights_int8.npz"),
                           "think": True, "aliases": ["leaf-things", "leaf-things-1.0"]},
        "leaf-flash": {"dir": _M("leaf-flash"),
                       "npz": _M("leaf-flash", "weights_int8.npz"),
                       "think": True, "preproc": True, "aliases": ["flash"]},
        "leaf-code": {"dir": _M("leaf-code"), "npz": _M("leaf-code", "weights_int8.npz"),
                      "think": True, "aliases": ["code"]},
    }
else:
    _E = lambda *p: os.path.join(ROOT, "exports", *p)
    MODELS_SPEC = {
        "leaf1.0": {"dir": _E("tinyllm-ft"),
                    "npz": _E("tinyllm-ft-int8", "weights_int8.npz"),
                    "think": False, "aliases": ["xiaoqiang-26m", "tinyllm-26m", "mini", "小强", "小微",
                                "leaf-1.0", "Leaf-1.0", "leaf 1.0"]},
        "leaf-things1.0": {"dir": _E("leaf-things"),
                           "npz": _E("leaf-things-int8", "weights_int8.npz"),
                           "think": True, "aliases": ["leaf-things", "leaf-things-1.0"]},
        "leaf-flash": {"dir": _E("leaf-flash"),
                       "npz": _E("leaf-flash", "weights_int8.npz"),
                       "think": True, "preproc": True, "aliases": ["flash"]},
        "leaf-code": {"dir": _E("leaf-code"), "npz": _E("leaf-code", "weights_int8.npz"),
                      "think": True, "aliases": ["code"]},
    }

_models = {}           # id -> (NPTinyLlama, TinyTokenizer)
_locks = {}            # id -> Lock
_stats = {"requests": 0, "tokens": 0, "start": time.time(), "rejected": 0}

# ---------------- 鉴权 (api.json 热加载) ----------------
_key_cache = {"mtime": -1, "keys": set()}


def load_api_keys():
    mtime = os.path.getmtime(API_JSON) if os.path.exists(API_JSON) else -1
    if mtime != _key_cache["mtime"]:
        try:
            with open(API_JSON, encoding="utf-8") as f:
                data = json.load(f)
            keys = set(data.get("api_keys", []))
            _key_cache.update(mtime=mtime, keys=keys)
            print(f"[api.json] 已加载 {len(keys)} 个 API Key")
        except Exception as e:
            print(f"[api.json] 解析失败({e}), 沿用上一次的 {len(_key_cache['keys'])} 个 key")
            _key_cache["mtime"] = mtime
    return _key_cache["keys"]


AUTH_DISABLED = True  # 已按要求关闭 API 鉴权; 改回 False 并配置 api.json 可重新启用


def check_auth(header):
    """返回 (ok, key)。AUTH_DISABLED=True 时所有请求放行。"""
    if AUTH_DISABLED:
        return True, "anonymous"
    if not header or not header.startswith("Bearer "):
        return False, None
    key = header[7:].strip()
    return (key in load_api_keys()), key


def lan_ips():
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return sorted(ips)


def resolve_model(name):
    """请求里的 model 名 -> 标准 id; 未知返回 None (大小写不敏感)"""
    if not name:
        return "leaf1.0"  # 默认
    n = str(name).strip()
    if n in MODELS_SPEC:
        return n
    for mid, spec in MODELS_SPEC.items():
        als = [a.lower() for a in spec["aliases"]] + [mid.lower()]
        if n.lower() in als:
            return mid
    return None


class NumpyBackend:
    """NumPy int8 引擎 (无 torch 依赖, 内存最小)"""
    name = "numpy-int8"

    def __init__(self, dir_path, npz_path):
        from tiny_numpy_engine import NPTinyLlama
        from tinyllm.tokenizer import TinyTokenizer
        self.model = NPTinyLlama(npz_path, os.path.join(dir_path, "config.json"))
        self.tok = TinyTokenizer(os.path.join(dir_path, "tokenizer.json"))
        self.MAX_POS = 204800

    def generate_deltas(self, messages, max_tokens, temperature, top_k, top_p, rep):
        from tiny_numpy_engine import sample_token
        model, tok = self.model, self.tok
        ids = tok.encode_chat(messages, add_generation_prompt=True, non_thinking=False)
        ids = ids[-(self.MAX_POS - max_tokens - 8):]
        model.reset()
        for t in ids[:-1]:
            model.forward(t)
        logits = model.forward(ids[-1])
        buf, emitted = [], 0
        for _ in range(max_tokens):
            nxt = sample_token(logits, temperature, top_k, top_p, rep, gen_ids=ids)
            if len(buf) < 6 and nxt == tok.eos_id:
                lg = logits.astype(np.float64).copy()
                lg[tok.eos_id] = -1e12
                nxt = int(lg.argmax())
            if nxt == tok.eos_id or model.pos >= self.MAX_POS - 1:
                break
            buf.append(nxt)
            full = tok.decode(buf)
            safe = full[:-1] if full.endswith("\ufffd") else full
            if len(safe) > emitted:
                yield safe[emitted:]
                emitted = len(safe)
            logits = model.forward(nxt)


class TorchBackend:
    """torch 引擎: 整段预填充 + 逐 token KV cache 解码; 设备可为 CPU 或 DirectML GPU"""
    name = "torch-fp32"

    def __init__(self, dir_path, device):
        import torch
        from safetensors.torch import load_file
        from tinyllm.model import TinyLlama, TinyConfig
        from tinyllm.tokenizer import TinyTokenizer
        torch.set_num_threads(1)  # batch-1 解码单线程最快
        cfg = TinyConfig.from_json(os.path.join(dir_path, "config.json"))
        m = TinyLlama(cfg)
        sd = load_file(os.path.join(dir_path, "model.safetensors"))
        sd = {k.removeprefix("model."): v for k, v in sd.items()}
        m.load_state_dict(sd, strict=False)
        m.eval().to(device)
        self.torch = torch
        self.model = m
        self.device = device
        self.dev_name = "GPU(DirectML)" if device != "cpu" else "CPU"
        self.tok = TinyTokenizer(os.path.join(dir_path, "tokenizer.json"))
        self.MAX_POS = 204800

    def generate_deltas(self, messages, max_tokens, temperature, top_k, top_p, rep):
        t = self.torch
        tok = self.tok
        ids = tok.encode_chat(messages, add_generation_prompt=True, non_thinking=False)
        ids = ids[-(self.MAX_POS - max_tokens - 8):]
        with t.no_grad():
            cur = t.tensor([ids], dtype=t.long, device=self.device)
            logits, past = self.model(cur)          # 整段预填充
            gen, buf, emitted = [], [], 0
            for _ in range(max_tokens):
                lg = logits[:, -1, :].float().cpu()  # 采样在 CPU (DML 不支持 multinomial)
                if len(gen) < 6:
                    lg[0, tok.eos_id] = -1e9  # 最少输出保护: 开头禁止结束符
                if rep and rep != 1.0:
                    for tid in set(ids + gen):
                        v = lg[0, tid]
                        lg[0, tid] = v / rep if v > 0 else v * rep
                if temperature <= 0:
                    nxt = int(lg.argmax(-1).item())
                else:
                    p = t.softmax(lg / temperature, dim=-1)[0]
                    if top_k and top_k < p.shape[0]:
                        cut = t.topk(p, top_k).values.min().item()
                        p = t.where(p < cut, t.zeros_like(p), p)
                        p = p / p.sum()
                    if top_p < 1.0:
                        sp, si = t.sort(p, descending=True)
                        cum = t.cumsum(sp, 0)
                        kill = si[(cum - sp) > top_p]
                        p[kill] = 0.0
                        p = p / p.sum()
                    nxt = int(t.multinomial(p, 1).item())
                if nxt == tok.eos_id or len(ids) + len(gen) >= self.MAX_POS - 1:
                    break
                gen.append(nxt)
                buf.append(nxt)
                full = tok.decode(buf)
                safe = full[:-1] if full.endswith("\ufffd") else full
                if len(safe) > emitted:
                    yield safe[emitted:]
                    emitted = len(safe)
                cur = t.tensor([[nxt]], dtype=t.long, device=self.device)
                logits, past = self.model(cur, past)


def load_models(args=None):
    use_torch = (not args or args.backend in ("auto", "torch"))
    if use_torch:
        try:
            import torch  # noqa
        except Exception:
            use_torch = False
            print("[后端] torch 不可用 -> numpy")
    dml_dev = None
    if use_torch and args and not args.cpu_only:
        try:
            import torch_directml
            dml_dev = torch_directml.device()
            print("[GPU] DirectML:", torch_directml.device_name(0))
        except Exception:
            pass
    gpu_mid = args.gpu_model if (args and dml_dev is not None) else None
    for mid, spec in MODELS_SPEC.items():
        d = spec["dir"]
        npz = spec["npz"]
        have_torch_w = os.path.exists(os.path.join(d, "model.safetensors"))
        have_npz = os.path.exists(npz)
        if not (os.path.exists(os.path.join(d, "config.json")) and (have_torch_w or have_npz)):
            print(f"[跳过] {mid}: 权重不存在 ({d})")
            continue
        t0 = time.perf_counter()
        dev_desc = "CPU"
        try:
            if use_torch and have_torch_w:
                dev = dml_dev if mid == gpu_mid else "cpu"
                be = TorchBackend(d, dev)
                dev_desc = be.dev_name
            elif have_npz:
                be = NumpyBackend(d, npz)
                dev_desc = "CPU(int8)"
            else:
                raise RuntimeError("无可用权重")
        except Exception as e:
            if have_npz:
                be = NumpyBackend(d, npz)
                dev_desc = "CPU(int8,回退)"
            else:
                print(f"[失败] {mid}: {e}")
                continue
        _models[mid] = (be, MODELS_SPEC[mid]["think"])
        _locks[mid] = threading.Lock()
        print(f"[加载] {mid}: {be.name} @ {dev_desc} ({time.perf_counter()-t0:.2f}s)")


# ---------------- 思维链流式拆分状态机 ----------------
class ThinkSplitter:
    """把生成文本流拆成 (reasoning_delta, content_delta)。
    识别 <think>...</think> 标记; 思考模型标记前的杂讯计入 content(容错)。"""

    def __init__(self, has_think=True):
        self.has_think = has_think
        self.state = "pre" if has_think else "post"
        self.buf = ""

    def feed(self, text):
        """返回 [(kind, delta)] kind in {'reasoning','content'}"""
        out = []
        if not self.has_think:
            if text:
                out.append(("content", text))
            return out
        self.buf += text
        while True:
            if self.state == "pre":
                i = self.buf.find("<think>")
                if i >= 0:
                    if self.buf[:i]:
                        out.append(("content", self.buf[:i]))
                    self.buf = self.buf[i + 7:]
                    self.state = "thinking"
                    continue
                keep = max(0, len(self.buf) - 7)
                if keep:
                    out.append(("content", self.buf[:keep]))
                    self.buf = self.buf[keep:]
                break
            elif self.state == "thinking":
                i = self.buf.find("</think>")
                if i >= 0:
                    if self.buf[:i]:
                        out.append(("reasoning", self.buf[:i]))
                    self.buf = self.buf[i + 8:].lstrip("\n")
                    self.state = "post"
                    continue
                keep = max(0, len(self.buf) - 8)
                if keep:
                    out.append(("reasoning", self.buf[:keep]))
                    self.buf = self.buf[keep:]
                break
            else:  # post
                if self.buf:
                    out.append(("content", self.buf))
                    self.buf = ""
                break
        return out


THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)


def split_think(text):
    """非流式: 整段拆分 -> (reasoning, content)"""
    m = THINK_RE.search(text)
    if not m:
        return "", text
    return m.group(1).strip(), (text[m.end():]).strip()


# ---------------- 图像消息解析 (leaf-flash 多模态) ----------------
def extract_images(body, messages):
    """从 OpenAI content 数组或 Ollama images 字段提取 base64 图像 -> PIL"""
    import base64
    from PIL import Image
    imgs = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = part.get("image_url", {})
                    url = url.get("url", "") if isinstance(url, dict) else str(url)
                    if "base64," in url:
                        b64 = url.split("base64,", 1)[1]
                        imgs.append(Image.open(io.BytesIO(base64.b64decode(b64))))
    for b64 in body.get("images") or []:
        try:
            imgs.append(Image.open(io.BytesIO(base64.b64decode(b64))))
        except Exception:
            pass
    return imgs


def flatten_text(messages):
    """content 可能是数组 -> 拼出纯文本消息"""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c
                         if isinstance(p, dict) and p.get("type") == "text")
        out.append({"role": m.get("role", "user"), "content": c or ""})
    return out


# ---------------- HTTP Handler ----------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stdout.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _unauthorized(self):
        _stats["rejected"] += 1
        self._json({"error": {"message": "无效的 API Key。请在请求头携带 api.json 中列出的密钥: "
                                         "Authorization: Bearer <你的key>",
                              "type": "invalid_api_key", "code": 401}}, 401)

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n <= 0:
            print(f"[请求] {self.path} 无请求体 (Content-Length=0)", flush=True)
            return {}
        raw = self.rfile.read(n)
        try:
            body = json.loads(raw.decode("utf-8"))
            if os.environ.get("LEAF_DEBUG"):
                print(f"[请求体] {self.path} <- {raw.decode('utf-8')[:400]}", flush=True)
            return body
        except Exception as e:
            print(f"[请求体解析失败] {self.path}: {e} | 原始前200字节: "
                  f"{raw[:200]!r}", flush=True)
            return {}

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _norm_path(self):
        """兼容客户端 Base URL 已带 /v1 的写法: /v1/v1/chat/... -> /v1/chat/..."""
        p = self.path
        while p.startswith("/v1/v1/"):
            p = p[3:]
        return p

    def do_GET(self):
        self.path = self._norm_path()
        if self.path == "/health":
            mem = 0
            try:
                from tiny_numpy_engine import process_mem_mb
                mem = process_mem_mb()
            except Exception:
                pass
            backends = {mid: _models[mid][0].name for mid in _models}
            self._json({"status": "ok", "models": sorted(_models.keys()),
                        "backends": backends,
                        "context": MAX_SEQ, "max_output_tokens": MAX_OUT,
                        "mem_mb": round(mem),
                        "uptime_s": int(time.time() - _stats["start"]),
                        "requests": _stats["requests"], "tokens": _stats["tokens"],
                        "rejected_auth": _stats["rejected"]})
            return
        if self.path in ("/", "/test", "/test.html"):
            import io as _io
            page = os.path.join(BASE, "test.html")
            if not os.path.exists(page):
                self._json({"error": {"message": "test.html not found"}}, 404)
                return
            body = _io.open(page, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/v1/models":
            if not check_auth(self.headers.get("Authorization"))[0]:
                self._unauthorized()
                return
            self._json({"object": "list", "data": [
                {"id": mid, "object": "model", "owned_by": "local",
                 "created": int(_stats["start"])} for mid in sorted(_models)]})
            return
        self._json({"error": {"message": f"unknown path {self.path}", "code": 404}}, 404)

    def do_POST(self):
        self.path = self._norm_path()
        if self.path not in ("/v1/chat/completions", "/v1/completions", "/api/chat"):
            self._json({"error": {"message": f"unknown path {self.path}", "code": 404}}, 404)
            return
        ok, _ = check_auth(self.headers.get("Authorization"))
        if not ok:
            self._unauthorized()
            return
        body = self._read_body()
        try:
            mid = resolve_model(body.get("model"))
            if mid is None or mid not in _models:
                self._json({"error": {"message": f"未知模型 {body.get('model')!r}, "
                                                 f"可用: {sorted(_models)}",
                                      "code": 404}}, 404)
                return
            if self.path == "/v1/completions":
                self._completions(mid, body)
            else:
                self._chat(mid, body, ollama_style=(self.path == "/api/chat"))
        except BrokenPipeError:
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                self._json({"error": {"message": str(e), "code": 500}}, 500)
            except Exception:
                pass

    # ---------- 生成核心 ----------
    def _generate_deltas(self, mid, messages, max_tokens, temperature, top_k, top_p, rep):
        backend, think = _models[mid]
        _stats["requests"] += 1
        produced = 0
        with _locks[mid]:
            for piece in backend.generate_deltas(messages, max_tokens, temperature,
                                                 top_k, top_p, rep):
                produced += len(piece)
                _stats["tokens"] += 1
                yield piece
        if produced == 0:  # 兜底: 保证绝不返回空内容
            yield "（我没能理解这个问题，请换个问法再试试。）"

    def _chat(self, mid, body, ollama_style=False):
        messages = body.get("messages") or []
        if not messages:
            self._json({"error": {"message": "messages required"}}, 400)
            return
        stream = bool(body.get("stream"))
        max_tokens = min(int(body.get("max_tokens") or DEFAULT_MAX_TOKENS), MAX_OUT)
        temperature = float(body.get("temperature", 0.6))
        top_k = int(body.get("top_k", 32))
        top_p = float(body.get("top_p", 0.9))
        rep = float(body.get("repeat_penalty", 1.1))
        think = _models[mid][1]
        preproc_reasoning = None
        if MODELS_SPEC[mid].get("preproc"):
            from tinyllm.vision_preproc import describe
            imgs = extract_images(body, messages)
            messages = flatten_text(messages)
            if imgs:
                obs = describe(imgs[0])
                preproc_reasoning = obs
                last = messages[-1]
                if last["role"] == "user":
                    last["content"] = ("阅读下面的图像观察记录：" + chr(10) + obs +
                                       chr(10) + chr(10) + "问题：" + last["content"])
                else:
                    last["content"] = "图像观察：" + obs + chr(10) + last["content"]
        rid = "chatcmpl-" + uuid.uuid4().hex[:12]
        created = int(time.time())
        echo_model = body.get("model") or mid   # 响应回显请求的模型名 (OpenAI 行为)

        gen = self._generate_deltas(mid, messages, max_tokens, temperature, top_k, top_p, rep)

        if not stream:
            raw = "".join(gen)
            reasoning, content = split_think(raw) if think else ("", raw)
            if preproc_reasoning:
                reasoning = (preproc_reasoning + chr(10) + reasoning).strip()
            msg = {"role": "assistant", "content": content}
            if reasoning:
                msg["reasoning_content"] = reasoning
            if ollama_style:
                resp = {"model": echo_model, "message": msg, "eval_count": len(raw), "done": True}
            else:
                resp = {"id": rid, "object": "chat.completion", "created": created,
                        "model": echo_model, "choices": [{"index": 0, "finish_reason": "stop",
                                                   "message": msg}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": len(raw),
                                  "total_tokens": len(raw)}}
            self._json(resp)
            return

        # SSE 流式
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self._cors()
        self.end_headers()

        def sse(obj):
            self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8"))
            self.wfile.flush()

        splitter = ThinkSplitter(think)
        if preproc_reasoning:
            splitter.has_think = False  # 思考来自前置视觉分析, 不再解析 <think>
        if not ollama_style:
            sse({"id": rid, "object": "chat.completion.chunk", "created": created,
                 "model": echo_model,
                 "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
        if preproc_reasoning:
            sse({"id": rid, "object": "chat.completion.chunk", "created": created,
                 "model": mid,
                 "choices": [{"index": 0,
                              "delta": {"reasoning_content": preproc_reasoning},
                              "finish_reason": None}]})
        n = 0
        for piece in gen:
            n += 1
            for kind, delta in splitter.feed(piece):
                if not delta:
                    continue
                if kind == "reasoning":
                    if ollama_style:
                        sse({"model": mid, "message": {"role": "assistant", "thinking": delta},
                             "done": False})
                    else:
                        sse({"id": rid, "object": "chat.completion.chunk", "created": created,
                             "model": echo_model, "choices": [{"index": 0,
                                                        "delta": {"reasoning_content": delta},
                                                        "finish_reason": None}]})
                else:
                    if ollama_style:
                        sse({"model": mid, "message": {"role": "assistant", "content": delta},
                             "done": False})
                    else:
                        sse({"id": rid, "object": "chat.completion.chunk", "created": created,
                             "model": echo_model, "choices": [{"index": 0, "delta": {"content": delta},
                                                        "finish_reason": None}]})
        # 冲刷 splitter 里残留的缓冲 (结束补一刀)
        for kind, delta in splitter.feed("\n"):
            if delta:
                if ollama_style:
                    sse({"model": mid, "message": {"role": "assistant", "content": delta},
                         "done": False})
                else:
                    sse({"id": rid, "object": "chat.completion.chunk", "created": created,
                         "model": echo_model, "choices": [{"index": 0, "delta": {"content": delta},
                                                    "finish_reason": None}]})
        if ollama_style:
            sse({"model": mid, "message": {"role": "assistant", "content": ""},
                 "done": True, "eval_count": n})
        else:
            sse({"id": rid, "object": "chat.completion.chunk", "created": created,
                 "model": mid, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    def _completions(self, mid, body):
        prompt = body.get("prompt")
        if not prompt:
            self._json({"error": {"message": "prompt required"}}, 400)
            return
        messages = [{"role": "user", "content": str(prompt)}]
        max_tokens = min(int(body.get("max_tokens") or 128), MAX_OUT)
        temperature = float(body.get("temperature", 0.6))
        gen = self._generate_deltas(mid, messages, max_tokens, temperature, 32, 0.9, 1.1)
        raw = "".join(gen)
        self._json({"id": "cmpl-" + uuid.uuid4().hex[:12], "object": "text_completion",
                    "created": int(time.time()), "model": mid,
                    "choices": [{"index": 0, "text": raw, "finish_reason": "stop"}]})


def main():
    ap = argparse.ArgumentParser(description="Leaf API 服务器 (双模型 + Key 鉴权)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--backend", choices=["auto", "torch", "numpy"], default="auto",
                    help="auto: 有 torch 用 torch(GPU+CPU 各扛一模型), 否则 numpy")
    ap.add_argument("--cpu-only", action="store_true", help="禁用 GPU")
    ap.add_argument("--gpu-model", default="leaf1.0",
                    help="放到 GPU 上的模型 (另一个在 CPU)")
    args = ap.parse_args()

    load_api_keys()
    load_models(args)
    if not _models:
        print("没有任何可用模型, 退出")
        return

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\nLeaf API 已启动 (Ctrl+C 退出)  可用模型: {sorted(_models)}")
    print(f"  本机:   http://127.0.0.1:{args.port}/v1/chat/completions")
    for ip in lan_ips():
        print(f"  局域网: http://{ip}:{args.port}/v1/chat/completions")
    print(f"\n调用示例 (key 来自 scripts/api.json):")
    print(f'  curl http://127.0.0.1:{args.port}/v1/chat/completions \\\n'
          f'       -H "Authorization: Bearer leaf-sk-123456" \\\n'
          f'       -H "Content-Type: application/json" \\\n'
          f'       -d \'{{"model":"leaf-things1.0","messages":[{{"role":"user","content":"17+26=?"}}]}}\'')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
