#!/usr/bin/env python3
"""Render captured terminal output into a terminal-window PNG.

Usage:  some-command | python3 tools/render.py out.png "window title"

The text you pipe in is drawn as-is, so only feed it output you really captured.
A line starting with "$ " is drawn as a typed command.
"""
import re
import sys

from PIL import Image, ImageDraw, ImageFont

# (regular, bold, ttc_index_regular, ttc_index_bold)
FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 0, 0),
    ("/usr/share/fonts/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf", 0, 0),
    ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Menlo.ttc", 0, 1),
    ("C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/consolab.ttf", 0, 0),
]
S, SIZE = 2, 15  # 2x for crisp screenshots
BG, FG, DIM = (22, 24, 31), (214, 219, 230), (125, 133, 150)
GREEN, AMBER, RED, CYAN, PINK = (126, 231, 135), (255, 203, 107), (255, 123, 114), (121, 210, 255), (240, 160, 255)


def load_fonts(size):
    for reg, bold, i_reg, i_bold in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(reg, size, index=i_reg), ImageFont.truetype(bold, size, index=i_bold)
        except OSError:
            continue
    raise SystemExit("No monospace font found. Edit FONT_CANDIDATES in tools/render.py.")


def color(line):
    if re.match(r"^(model|policy|\s+cold start|GPU=)", line):
        return CYAN, True
    if line.startswith("==") or line.startswith("Little"):
        return PINK, True
    if re.search(r"\bOK\b", line):
        return GREEN, False
    if re.search(r"\bFAIL\b", line):
        return RED, False
    if re.search(r"\bSKIP\b|does not fit", line):
        return AMBER, False
    return FG, False


def render(lines, title, path):
    f, fb = load_fonts(SIZE * S)
    cw = f.getlength("M")
    lh = int(SIZE * S * 1.45)
    cols = max(len(line) for line in lines)
    pad, bar = 22 * S, 38 * S
    w, h = int(cols * cw) + 2 * pad, bar + 2 * pad + lh * len(lines)
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, bar], fill=(36, 39, 50))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        r = 7 * S
        cx = pad + i * 22 * S + r
        d.ellipse([cx - r, bar // 2 - r, cx + r, bar // 2 + r], fill=c)
    tf, _ = load_fonts(13 * S)
    d.text((w // 2 - tf.getlength(title) / 2, bar // 2 - 8 * S), title, font=tf, fill=DIM)
    y = bar + pad
    for line in lines:
        if line.startswith("$ "):
            d.text((pad, y), "$ ", font=fb, fill=GREEN)
            d.text((pad + 2 * cw, y), line[2:], font=f, fill=(255, 255, 255))
        else:
            c, bold = color(line)
            d.text((pad, y), line, font=fb if bold else f, fill=c)
        y += lh
    img.save(path, optimize=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    render(sys.stdin.read().rstrip("\n").split("\n"), sys.argv[2], sys.argv[1])
