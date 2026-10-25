# -*- coding: utf-8 -*-
"""基于 tokenizers 库的极简封装 (不依赖 transformers)。

特殊 token: 0=<|endoftext|>  1=<|im_start|>(bos)  2=<|im_end|>(eos)
对话格式与 MiniMind2 chat_template 一致。
"""
import json
from tokenizers import Tokenizer

IM_START, IM_END = "<|im_start|>", "<|im_end|>"


class TinyTokenizer:
    def __init__(self, tokenizer_json):
        self.tok = Tokenizer.from_file(tokenizer_json)
        self.bos_id, self.eos_id, self.pad_id = 1, 2, 0

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)

    def encode_chat(self, messages, add_generation_prompt=False, non_thinking=True):
        """按 MiniMind2 模板拼接; 返回 (ids, assistant_start_index)"""
        s = ""
        if not any(m["role"] == "system" for m in messages):
            s += f"{IM_START}system\nYou are a helpful assistant{IM_END}\n"
        for m in messages:
            if m["role"] in ("system", "user"):
                s += f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n"
            elif m["role"] == "assistant":
                s += f"{IM_START}assistant\n{m['content']}"
        if add_generation_prompt:
            s += f"{IM_START}assistant\n"
            if non_thinking:
                s += "<think>\n\n</think>\n\n"
        return self.encode(s)

    def apply_train_template(self, messages):
        """训练用(单轮): assistant 回复结尾追加 <|im_end|>, 返回 (ids, prompt_len)"""
        s = ""
        prompt = ""
        if not any(m["role"] == "system" for m in messages):
            prompt += f"{IM_START}system\nYou are a helpful assistant{IM_END}\n"
        for m in messages:
            if m["role"] in ("system", "user"):
                prompt += f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n"
            elif m["role"] == "assistant":
                prompt += f"{IM_START}assistant\n"
                if m.get("non_thinking", True):
                    prompt += "<think>\n\n</think>\n\n"
                s = prompt + m["content"] + IM_END
        ids = self.encode(s)
        return ids, len(self.encode(prompt))

    def encode_train(self, messages):
        """训练用(支持多轮): 返回 (ids, spans), spans 为每个 assistant
        回复 (含 <|im_end|>) 的 [起,止) 区间, 损失只算这些区间。"""
        ids_all, spans = [], []
        pos = 0
        if not any(m["role"] == "system" for m in messages):
            t = self.encode(f"{IM_START}system\nYou are a helpful assistant{IM_END}\n")
            ids_all += t
            pos += len(t)
        for m in messages:
            if m["role"] in ("system", "user"):
                t = self.encode(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n")
                ids_all += t
                pos += len(t)
            elif m["role"] == "assistant":
                t = self.encode(f"{IM_START}assistant\n")
                ids_all += t
                pos += len(t)
                t = self.encode(m["content"] + IM_END)
                spans.append((pos, pos + len(t)))
                ids_all += t
                pos += len(t)
                t = self.encode("\n")
                ids_all += t
                pos += len(t)
        return ids_all, spans

    def save_vocab_info(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab_size": self.tok.get_vocab_size(), "bos": self.bos_id,
                       "eos": self.eos_id}, f, ensure_ascii=False, indent=2)


def load_tokenizer(model_dir):
    import os
    return TinyTokenizer(os.path.join(model_dir, "tokenizer.json"))
