import ast
import base64
import json
import mimetypes
import os
import re
import uuid
from collections import defaultdict
from io import BytesIO
from math import hypot

import requests
from PIL import Image, ImageDraw

from module.geometric_constraint_correction_module.gccm_tools import _to_data_url, _decode_data_url_to_image
from utils.fonts import get_font, additional_colors
from utils.json_tools import clean_json_str, parse_json, decode_json_points

# === 新增：将框 JSON 送入第二阶段的辅助函数 ===
def load_boxes(json_text: str):
    """将第一阶段返回的框 JSON 解析为 list[{'bbox_2d':[x1,y1,x2,y2],'label':'...'}]"""
    body = parse_json(json_text)
    try:
        data = json.loads(body)
    except Exception:
        # 兜底：某些模型会缺少结尾
        end_idx = body.rfind('"}') + len('"}')
        data = json.loads(body[:end_idx] + "]")
    if not isinstance(data, list):
        data = [data]
    cleaned = []
    for d in data:
        if "bbox_2d" in d and "label" in d and isinstance(d["bbox_2d"], (list, tuple)) and len(d["bbox_2d"]) == 4:
            try:
                x1,y1,x2,y2 = [int(v) for v in d["bbox_2d"]]
            except Exception:
                continue
            if 0 <= x1 <= 999 and 0 <= y1 <= 999 and 0 <= x2 <= 999 and 0 <= y2 <= 999:
                cleaned.append({"bbox_2d":[x1,y1,x2,y2], "label":str(d["label"])})
    return cleaned

def normalize_shape_name(name: str) -> str:
    """与 plot_points.parse_label 的别名对齐，用于生成实例编号前缀"""
    alias = {
        "rectangle": "rect",
        "square": "rect",
        "parallelogram": "para",
        "rhombus": "rhom",
        "trapezoid": "trap",
        "pentagon": "penta",
        "hexagon": "hexa",
        "heptagon": "hepta",
        "octagon": "octa",
        "polygon": "poly",
        "line_segment": "line",
    }
    return alias.get(name, name)

def build_points_prompt_with_boxes(boxes):
    """
    把已检测到的框清单（含类别、实例编号、bbox）塞进第二阶段的 prompt，
    要求“只在这些框内标点，编号与清单一致”。
    """
    prompt_points = (
        '你是一名视觉关键点标注助手。请在一张由 Manim 生成的数学/几何场景图片中，'
        '为每个已识别的几何图形输出关键点，并返回 JSON 数组（只输出数组本体，无多余文字/Markdown）。'
        '【实例编号规则】按“从上到下、同一行从左到右”的顺序，为同类形状依次编号为 #1,#2,#3...，'
        '确保与检测框顺序一致。'
        '【坐标系】使用 0~999 归一化坐标；每点格式 {"point_2d":[x,y],"label":"..."}；x、y 为整数。'
        '【各类别关键点命名】'
        'triangle: 三顶点(顺时针) -> triangle_v1..v3；'
        'square/rectangle: 四顶点(顺时针) -> rect_v1..v4；'
        'parallelogram: 四顶点(顺时针) -> para_v1..v4；'
        'rhombus: 四顶点(顺时针) -> rhom_v1..v4；'
        'trapezoid: 四顶点(顺时针) -> trap_v1..v4；'
        'kite: 四顶点(顺时针) -> kite_v1..v4；'
        'pentagon: 五顶点(顺时针) -> penta_v1..v5；'
        'hexagon: 六顶点(顺时针) -> hexa_v1..v6；'
        'heptagon: 七顶点(顺时针) -> hepta_v1..v7；'
        'octagon: 八顶点(顺时针) -> octa_v1..v8；'
        'polygon(>8): 所有顶点(顺时针) -> poly_v1..vN；'
        'circle: 圆心与圆上一点 -> circle_center, circle_rpoint；'
        'ellipse: 中心与主轴端点之一 -> ellipse_center, ellipse_axis；'
        'line_segment: 两端点 -> line_p1, line_p2；'
        'arrow: 起点与箭头尖 -> arrow_tail, arrow_head；'
        'axis: 原点与正向刻度点 -> axis_origin, axis_pos；'
        'text: 文本近似中心 -> text_center。'
        '【多实例标注】若同类目标有多个，在上述标签末尾加 #编号，例如 rect_v1#1..rect_v4#1；'
        '第二个矩形为 rect_v1#2..rect_v4#2；其他类别以此类推。'
        '【一致性】若某目标已能明确为具体类别（如 square），请使用对应前缀（此处统一用 rect_ 作为四角形的关键点前缀即可），'
        '不要输出冲突类别的关键点。忽略水印/装饰元素/相机边框。'
    )
    per_class_counter = defaultdict(int)
    annotated = []  # [(cls_norm, inst_id, raw_label, bbox)]

    for b in boxes:
        raw_label = str(b["label"])
        cls_norm = normalize_shape_name(raw_label)
        per_class_counter[cls_norm] += 1
        inst_id = per_class_counter[cls_norm]
        annotated.append((cls_norm, inst_id, raw_label, b["bbox_2d"]))

    lines = []
    lines.append("【已检测目标清单】按下列顺序与编号输出关键点，仅在对应 bbox 内标注：")
    for cls_norm, inst_id, raw_label, (x1,y1,x2,y2) in annotated:
        lines.append(f"- {raw_label} -> 实例 #{inst_id}，bbox_2d=[{x1},{y1},{x2},{y2}]")

    constraints = (
        "【强制约束】\n"
        "1) 实例编号与上表严格一致（同类各自独立编号）。\n"
        "2) 关键点标签后缀必须带 #编号（如 rect_v1#2、triangle_v3#1 等）。\n"
        "3) 只在对应 bbox 范围内找点；若不确定则该实例留空，不要臆造。\n"
        "4) 坐标使用 0~999 归一化整型；仅输出纯 JSON 数组（不含任何解释文本）。\n"
    )

    return prompt_points + "\n" + "\n".join(lines) + "\n" + constraints

# =========================
# 3) 可视化：矩形框
# =========================
def plot_bounding_boxes(im: Image.Image, bounding_boxes: str):
    """
    绘制检测框。输入 JSON 为 [{"bbox_2d":[x1,y1,x2,y2],"label":"..."}]
    坐标为 0-999 归一化。
    """
    img = im
    width, height = img.size
    print("Image size:", img.size)
    draw = ImageDraw.Draw(img)
    font = get_font(14)

    colors = [
        'red','green','blue','yellow','orange','pink','purple','brown','gray','beige',
        'turquoise','cyan','magenta','lime','navy','maroon','teal','olive','coral',
        'lavender','violet','gold','silver',
    ] + additional_colors

    bbox_str = parse_json(bounding_boxes)
    try:
        dets = ast.literal_eval(bbox_str)
    except Exception:
        end_idx = bbox_str.rfind('"}') + len('"}')
        dets = ast.literal_eval(bbox_str[:end_idx] + "]")
    if not isinstance(dets, list):
        dets = [dets]

    for i, item in enumerate(dets):
        if "bbox_2d" not in item:
            continue
        x1n, y1n, x2n, y2n = item["bbox_2d"]
        color = colors[i % len(colors)]
        # 归一化 -> 绝对像素
        x1 = int(x1n / 1000 * width)
        y1 = int(y1n / 1000 * height)
        x2 = int(x2n / 1000 * width)
        y2 = int(y2n / 1000 * height)
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1

        draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=3)
        label = item.get("label", "")
        if label:
            draw.text((x1 + 8, y1 + 6), label, fill=color, font=font)

    #img.show()

# =========================
# 4) 可视化：关键点（增强版：自动连线/闭合）
# =========================
def parse_label(label: str):
    """
    解析标签，返回 (shape_norm, instance_id:int, vertex_tag:str)
    约定示例：
      triangle_v1#1 -> ("triangle", 1, "v1")
      rect_v3#2     -> ("rect", 2, "v3")
      penta_v5#1    -> ("penta", 1, "v5")
      poly_v7#3     -> ("poly", 3, "v7")
      circle_center#1 -> ("circle", 1, "center")
      line_p1#1 / line_p2#1 -> ("line", 1, "p1"/"p2")
      arrow_tail#1 / arrow_head#1 -> ("arrow", 1, "tail"/"head")
      axis_origin#1 / axis_pos#1 -> ("axis", 1, "origin"/"pos")
    """
    inst = 1
    if '#' in label:
        try:
            inst = int(label.split('#')[-1])
        except:
            inst = 1
    base = label.split('#')[0]
    if '_' in base:
        shape, vertex = base.split('_', 1)
    else:
        shape, vertex = base, ""

    alias = {
        "rectangle": "rect",
        "square": "rect",
        "parallelogram": "para",
        "rhombus": "rhom",
        "trapezoid": "trap",
        "pentagon": "penta",
        "hexagon": "hexa",
        "heptagon": "hepta",
        "octagon": "octa",
        "polygon": "poly",
        "line_segment": "line",
    }
    shape_norm = alias.get(shape, shape)
    return shape_norm, inst, vertex

def connect_groups_and_shapes(draw, groups, color_map):
    """按组连线/闭合多边形或绘制特殊几何（circle/line/arrow/axis）。"""
    closed_shapes = {"triangle","rect","para","rhom","trap","kite","penta","hexa","hepta","octa","poly"}
    line_like = {"line","arrow","axis"}

    for key, pts in groups.items():
        shape_norm, inst = key
        color = color_map.get(key, "white")

        def v_index(p):
            m = re.search(r'^v(\d+)$', p["vertex"])
            return int(m.group(1)) if m else 10**9

        ordered = sorted(pts, key=v_index)

        if shape_norm == "circle":
            cands = {p["vertex"]: p for p in ordered}
            if "center" in cands and "rpoint" in cands:
                cx, cy = cands["center"]["pt"]
                rx, ry = cands["rpoint"]["pt"]
                r = hypot(rx - cx, ry - cy)
                draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=color, width=2)
            continue

        if shape_norm in line_like:
            name_order = {"line": ["p1","p2"], "arrow": ["tail","head"], "axis": ["origin","pos"]}
            order = name_order.get(shape_norm, [])
            line_pts = []
            for name in order:
                for p in pts:
                    if p["vertex"] == name:
                        line_pts.append(p["pt"])
                        break
            if len(line_pts) >= 2:
                draw.line([line_pts[0], line_pts[1]], fill=color, width=2)
            continue

        if shape_norm in closed_shapes:
            poly = [p["pt"] for p in ordered if re.match(r'^v\d+$', p["vertex"])]
            if len(poly) >= 3:
                draw.line(poly + [poly[0]], fill=color, width=2)
            continue

def plot_points(im: Image.Image, text: str):
    """
    绘制关键点 + 标签，并自动把同一形状实例的顶点连线/闭合。
    输入为 [{"point_2d":[x,y], "label":"..."}]，坐标为 0-999 归一化。
    """
    img = im
    width, height = img.size
    draw = ImageDraw.Draw(img)
    font = get_font(14)

    colors = [
        'red','green','blue','yellow','orange','pink','purple','brown','gray',
        'beige','turquoise','cyan','magenta','lime','navy','maroon','teal',
        'olive','coral','lavender','violet','gold','silver',
    ] + additional_colors

    points, labels = decode_json_points(text)
    print("Parsed points:", points)
    print("Parsed descriptions:", labels)
    DEBUG_SHOW = False

    if not points:
        if DEBUG_SHOW:
            img.show()
        return

    groups = {}    # (shape_norm, inst) -> [{"label","pt","vertex"}]
    color_map = {} # (shape_norm, inst) -> color

    for i, (nx, ny) in enumerate(points):
        label = labels[i] if i < len(labels) else ""
        color = colors[i % len(colors)]

        px = int(nx) / 1000 * width
        py = int(ny) / 1000 * height
        r = 2
        draw.ellipse([(px - r, py - r), (px + r, py + r)], fill=color)
        if label:
            draw.text((px - 20, py + 6), label, fill=color, font=font)

        if label:
            shape_norm, inst, vertex = parse_label(label)
            key = (shape_norm, inst)
            groups.setdefault(key, []).append({"label": label, "pt": (px, py), "vertex": vertex})
            color_map.setdefault(key, color)

    connect_groups_and_shapes(draw, groups, color_map)
    #img.show()




