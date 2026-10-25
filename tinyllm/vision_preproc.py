# -*- coding: utf-8 -*-
"""leaf-flash 视觉前置分析器: 纯 numpy+PIL 的确定性图像理解。
针对本项目的合成图像域 (颜色形状/数字), 输出结构化中文描述给 LLM。
架构: 经典 CV 前置 + LLM 思考回答 (0.1B 模型的实用多模态方案)。"""
import numpy as np
from PIL import Image

COLORS = {"红色": (220, 40, 40), "蓝色": (40, 80, 220), "绿色": (30, 160, 60),
          "黄色": (230, 190, 20), "紫色": (140, 40, 180), "橙色": (240, 130, 20)}
SHAPES = ["圆形", "方块", "三角形"]


def _nearest_color(rgb):
    best, bd = None, 1e9
    for name, c in COLORS.items():
        d = sum((int(a) - int(b)) ** 2 for a, b in zip(rgb, c))
        if d < bd:
            best, bd = name, d
    return best


def _components(mask):
    """简单 4 邻域连通域, 返回 [(x0,y0,x1,y1,area)]"""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    ys, xs = np.where(mask)
    coords = set(zip(ys.tolist(), xs.tolist()))
    for y0, x0 in list(coords):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        pts = []
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        ys2 = [p[0] for p in pts]
        xs2 = [p[1] for p in pts]
        out.append((min(xs2), min(ys2), max(xs2), max(ys2), len(pts)))
    return out


def _classify_shape(w, h, area, mask):
    """按轮廓特征分类: 圆形(面积/外接矩形比高且边缘平滑), 方块(占比~0.78+直角), 三角形(占比~0.5)"""
    fill = area / max(w * h, 1)
    # 三角形填充率 ~0.5, 方块 ~1.0, 圆 ~0.785
    if fill < 0.62:
        return "三角形"
    if fill > 0.9:
        return "方块"
    return "圆形"


def analyze(img):
    """分析图像 -> (facts 列表, 摘要)。支持: 彩色形状(1-3个, 左中右) / 黑色数字"""
    img = img.convert("RGB")
    a = np.asarray(img).astype(int)
    facts = []
    digit_found = None
    # 1) 深色像素 -> 数字判定 (黑灰且占比小)
    dark = (a.max(axis=2) < 100)
    if dark.sum() > 30:
        digit_found = _recognize_digit(a, dark)
    # 2) 各颜色连通域
    found = []
    for cname, c in COLORS.items():
        diff = np.abs(a - np.array(c)).sum(axis=2)
        mask = diff < 150
        if mask.sum() < 60:
            continue
        for (x0, y0, x1, y1, area) in _components(mask):
            if area < 60:
                continue
            shape = _classify_shape(x1 - x0 + 1, y1 - y0 + 1, area, mask)
            found.append({"color": cname, "shape": shape,
                          "cx": (x0 + x1) / 2, "area": area})
    found.sort(key=lambda f: f["cx"])
    if digit_found is not None:
        facts.append(f"图中写着一个手写数字，识别结果是 {digit_found}。")
    if found:
        pos = ["左边", "中间", "右边"] if len(found) == 3 else \
            (["左边", "右边"] if len(found) == 2 else ["图中"])
        parts = []
        for f, p in zip(found, pos):
            parts.append(f"{p}有一个{f['color']}的{f['shape']}")
        facts.append("图中从左到右：" + "，".join(parts) + "。")
        summary = "；".join(f"{f['color']}{f['shape']}" for f in found)
        facts.append(f"图中物体清单：{summary}。")
    if not facts:
        facts.append("图中没有检测到明确的彩色形状或数字。")
    return facts


def _recognize_digit(a, dark):
    """模板匹配: 用 PIL 字体渲染 0-9 与暗像素图比对 (本项目数字为标准字体)"""
    from PIL import Image, ImageDraw, ImageFont
    h, w = dark.shape
    ys, xs = np.where(dark)
    if len(ys) == 0:
        return None
    sub = dark[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    best, bd = None, 1e18
    try:
        f = ImageFont.truetype("arial.ttf", 44)
    except Exception:
        f = ImageFont.load_default()
    for d in range(10):
        t = Image.new("L", (w, h), 0)
        dr = ImageDraw.Draw(t)
        try:
            dr.text((xs.min(), ys.min() - 6), str(d), fill=255, font=f)
        except Exception:
            continue
        ta = np.asarray(t) > 100
        if ta.sum() == 0:
            continue
        # 对齐: 同尺寸比较 IoU
        ys2, xs2 = np.where(ta)
        if len(ys2) == 0:
            continue
        t2 = ta[ys2.min():ys2.max() + 1, xs2.min():xs2.max() + 1]
        # resize sub to t2 shape
        im = Image.fromarray((sub * 255).astype(np.uint8)).resize(
            (t2.shape[1], t2.shape[0]))
        s2 = np.asarray(im) > 100
        inter = (s2 & t2).sum()
        union = (s2 | t2).sum()
        iou = inter / max(union, 1)
        if iou < bd:
            bd, best = iou, d
    # bd 是最小 iou 的坑: 重取最大
    best_score, best = -1, None
    for d in range(10):
        t = Image.new("L", (w, h), 0)
        dr = ImageDraw.Draw(t)
        try:
            dr.text((xs.min(), ys.min() - 6), str(d), fill=255, font=f)
        except Exception:
            continue
        ta = np.asarray(t) > 100
        if ta.sum() == 0:
            continue
        ys2, xs2 = np.where(ta)
        t2 = ta[ys2.min():ys2.max() + 1, xs2.min():xs2.max() + 1]
        im = Image.fromarray((sub * 255).astype(np.uint8)).resize(
            (t2.shape[1], t2.shape[0]))
        s2 = np.asarray(im) > 100
        iou = (s2 & t2).sum() / max((s2 | t2).sum(), 1)
        if iou > best_score:
            best_score, best = iou, d
    return best if best_score > 0.3 else None


def describe(img):
    """返回观察文本 (作为 reasoning_content / 上下文注入)"""
    return "。".join(analyze(img))
