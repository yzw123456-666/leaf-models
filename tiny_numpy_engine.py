# -*- coding: utf-8 -*-
"""纯 NumPy 轻量推理引擎 (无 PyTorch 依赖, 老旧 CPU / 小内存专用)。

依赖: pip install numpy tokenizers  (合计约 30MB, 无需 torch)
权重: exports/tinyllm-ft-int8/weights_int8.npz (int8 量化, 约 26MB)

  python scripts/tiny_numpy_engine.py --chat                 # 对话 (省内存模式)
  python scripts/tiny_numpy_engine.py --bench                # 测速
  python scripts/tiny_numpy_engine.py --chat --fp32-cache    # 快速模式 (权重常驻 fp32, 103MB)

原理: 权重保持 int8; 每次矩阵乘法时 numpy 自动把 int8 提升为 float32
再调用 BLAS sgemm, 逐输出通道 scale 在乘法后乘回。省内存模式以少量
逐 token 转换开销换取 4 倍更小的权重常驻内存。
"""
import argparse
import ctypes
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from tinyllm.tokenizer import TinyTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FT_DIR = os.path.join(ROOT, "exports", "tinyllm-ft")
NPZ = os.path.join(ROOT, "exports", "tinyllm-ft-int8", "weights_int8.npz")
MAX_SEQ = 204800  # 200K 上下文 (KV 缓存虚拟内存, 按需提交)


class NPTinyLlama:
    def __init__(self, npz_path, config_path, fp32_cache=False):
        z = np.load(npz_path)
        cfg = json.load(open(config_path, encoding="utf-8"))
        self.H = cfg["hidden_size"]
        self.L = cfg["num_hidden_layers"]
        self.nh = cfg["num_attention_heads"]
        self.nkv = cfg["num_key_value_heads"]
        self.hd = cfg["head_dim"]
        self.eps = cfg["rms_norm_eps"]
        self.n_rep = self.nh // self.nkv
        self.scale = 1.0 / np.sqrt(self.hd)
        self.vocab = cfg["vocab_size"]
        self.fp32_cache = fp32_cache

        self.emb_q = z["model.embed_tokens.weight.q"]
        self.emb_s = z["model.embed_tokens.weight.s"].astype(np.float32)

        def W(name):
            wq = z[name + ".q"]
            s = z[name + ".s"].astype(np.float32)  # [out]
            if fp32_cache:
                return wq.astype(np.float32) * s[:, None]
            return (wq, s)

        self.emb_w = (self.emb_q.astype(np.float32) * self.emb_s[:, None]) if fp32_cache else None
        self.layers = []
        for i in range(self.L):
            p = f"model.layers.{i}."
            self.layers.append({
                "ln1": z[p + "input_layernorm.weight"].astype(np.float32),
                "ln2": z[p + "post_attention_layernorm.weight"].astype(np.float32),
                "q": W(p + "self_attn.q_proj.weight"),
                "k": W(p + "self_attn.k_proj.weight"),
                "v": W(p + "self_attn.v_proj.weight"),
                "o": W(p + "self_attn.o_proj.weight"),
                "g": W(p + "mlp.gate_proj.weight"),
                "u": W(p + "mlp.up_proj.weight"),
                "d": W(p + "mlp.down_proj.weight"),
            })
        self.norm = z["model.norm.weight"].astype(np.float32)

        # RoPE 按需计算 (支持 128K 超长上下文, 不预存大表)
        self.inv_freq = (1.0 / (cfg["rope_theta"] ** (
            np.arange(0, self.hd, 2, dtype=np.float32) / self.hd))).astype(np.float32)

        # KV cache
        self.cache_k = np.zeros((self.L, self.nkv, MAX_SEQ, self.hd), dtype=np.float32)
        self.cache_v = np.zeros((self.L, self.nkv, MAX_SEQ, self.hd), dtype=np.float32)
        self.pos = 0

    # ---- 基础算子 ----
    def _mm(self, w, x):
        """w: 缓存时 fp32 [out,in]; 否则 (int8 [out,in], scale [out]) ; x: [in] fp32"""
        if self.fp32_cache:
            return w @ x
        wq, s = w
        return (x @ wq.T) * s

    def _rms(self, x, g):
        return x * g / np.sqrt(np.mean(x * x) + self.eps)

    def _rope(self, x, p):
        """x: [hd] 单头; p 为绝对位置"""
        freqs = self.inv_freq * np.float32(p)
        emb = np.concatenate([freqs, freqs])
        c = np.cos(emb).astype(np.float32)
        s = np.sin(emb).astype(np.float32)
        x1, x2 = x[: self.hd // 2], x[self.hd // 2:]
        return np.concatenate([x1 * c[: self.hd // 2] - x2 * s[: self.hd // 2],
                               x1 * s[self.hd // 2:] + x2 * c[self.hd // 2:]])

    def forward(self, tok_id):
        p = self.pos
        x = (self.emb_w[tok_id] if self.fp32_cache
             else self.emb_q[tok_id].astype(np.float32) * self.emb_s[tok_id])
        for li, lyr in enumerate(self.layers):
            h = self._rms(x, lyr["ln1"])
            q = self._mm(lyr["q"], h).reshape(self.nh, self.hd)
            k = self._mm(lyr["k"], h).reshape(self.nkv, self.hd)
            v = self._mm(lyr["v"], h).reshape(self.nkv, self.hd)
            for hh in range(self.nh):
                q[hh] = self._rope(q[hh], p)
            for hh in range(self.nkv):
                k[hh] = self._rope(k[hh], p)
                self.cache_k[li, hh, p] = k[hh]
                self.cache_v[li, hh, p] = v[hh]
            T = p + 1
            out = np.zeros((self.nh, self.hd), dtype=np.float32)
            for hh in range(self.nh):
                kvh = hh // self.n_rep
                K = self.cache_k[li, kvh, :T]            # [T,hd]
                V = self.cache_v[li, kvh, :T]
                sc = (K @ q[hh]) * self.scale            # [T]
                sc -= sc.max()
                e = np.exp(sc)
                pr = e / e.sum()
                out[hh] = pr @ V
            x = x + self._mm(lyr["o"], out.reshape(-1))
            h = self._rms(x, lyr["ln2"])
            g = self._mm(lyr["g"], h)
            u = self._mm(lyr["u"], h)
            x = x + self._mm(lyr["d"], (g / (1 + np.exp(-g)) * u))
        x = self._rms(x, self.norm)
        if self.fp32_cache:
            logits = self.emb_w @ x
        else:
            logits = (x @ self.emb_q.T) * self.emb_s
        self.pos += 1
        return logits

    def reset(self):
        self.pos = 0


def sample_token(logits, temperature=0.7, top_k=32, top_p=0.9, rep_penalty=1.1, gen_ids=None):
    logits = logits.astype(np.float64)
    if rep_penalty != 1.0 and gen_ids:
        for t in set(gen_ids):
            if logits[t] > 0:
                logits[t] /= rep_penalty
            else:
                logits[t] *= rep_penalty
    if temperature <= 0:
        return int(np.argmax(logits))
    logits /= temperature
    logits -= logits.max()
    pr = np.exp(logits)
    pr /= pr.sum()
    if top_k and top_k < pr.shape[0]:
        cut = np.partition(pr, -top_k)[-top_k]
        pr[pr < cut] = 0
        pr /= pr.sum()
    if top_p < 1.0:
        order = np.argsort(-pr)
        cum = np.cumsum(pr[order])
        kill = order[(cum - pr[order]) > top_p]
        pr[kill] = 0
        pr /= pr.sum()
    return int(np.random.choice(pr.shape[0], p=pr))


def process_mem_mb():
    if os.name != "nt":
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3
        except Exception:
            return 0.0
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(pmc)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.windll.psapi
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_uint32]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return 0.0
    return pmc.WorkingSetSize / 1e6


def generate(model, tok, prompt_ids, max_new=200, temperature=0.7, rep_penalty=1.0):
    model.reset()
    for t in prompt_ids[:-1]:
        model.forward(t)
    logits = model.forward(prompt_ids[-1])
    gen = []
    for _ in range(max_new):
        nxt = sample_token(logits, temperature, rep_penalty=rep_penalty, gen_ids=prompt_ids + gen)
        if nxt == tok.eos_id:
            break
        gen.append(nxt)
        if model.pos >= MAX_SEQ - 1:
            break
        logits = model.forward(nxt)
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--fp32-cache", action="store_true", help="权重常驻fp32(103MB), 速度更快")
    ap.add_argument("--prompt", default="介绍一下人工智能")
    ap.add_argument("--temperature", type=float, default=0.6)
    args = ap.parse_args()

    t0 = time.perf_counter()
    model = NPTinyLlama(NPZ, os.path.join(FT_DIR, "config.json"), fp32_cache=args.fp32_cache)
    tok = TinyTokenizer(os.path.join(FT_DIR, "tokenizer.json"))
    print(f"加载 {time.perf_counter()-t0:.2f}s | 模式: {'fp32快速' if args.fp32_cache else 'int8省内存'}"
          f" | 进程内存 {process_mem_mb():.0f} MB")

    if args.bench:
        ids = tok.encode_chat([{"role": "user", "content": args.prompt}],
                              add_generation_prompt=True, non_thinking=False)
        generate(model, tok, ids, max_new=16, temperature=0.0)  # 预热
        model.reset()
        t0 = time.perf_counter()          # 预填充
        for t in ids[:-1]:
            model.forward(t)
        logits = model.forward(ids[-1])
        t_pre = time.perf_counter() - t0
        gen = []
        t0 = time.perf_counter()          # 逐 token 解码
        for _ in range(128):
            nxt = sample_token(logits, 0.0)
            if nxt == tok.eos_id:
                break
            gen.append(nxt)
            logits = model.forward(nxt)
        t_dec = time.perf_counter() - t0
        print(f"预填充 {len(ids)} token: {len(ids)/t_pre:.1f} tok/s | "
              f"解码 {len(gen)} token: {len(gen)/t_dec:.2f} tok/s | "
              f"进程内存 {process_mem_mb():.0f} MB")
        print("样例输出:", tok.decode(gen)[:120])

    if args.chat:
        print("开始对话 (输入 quit 退出)。")
        history = []
        while True:
            try:
                user = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user:
                continue
            if user.lower() in ("quit", "exit", "q"):
                break
            history.append({"role": "user", "content": user})
            if len(history) > 8:
                history = history[-6:]
            ids = tok.encode_chat(history, add_generation_prompt=True, non_thinking=False)
            ids = ids[-(MAX_SEQ - 210):]
            t0 = time.perf_counter()
            gen = generate(model, tok, ids, max_new=200, rep_penalty=1.1,
                           temperature=args.temperature)
            dt = time.perf_counter() - t0
            text = tok.decode(gen)
            print(f"小微: {text}")
            print(f"    [{len(gen)/dt:.1f} tok/s]")
            history.append({"role": "assistant", "content": text})
        print(f"\n进程内存: {process_mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
