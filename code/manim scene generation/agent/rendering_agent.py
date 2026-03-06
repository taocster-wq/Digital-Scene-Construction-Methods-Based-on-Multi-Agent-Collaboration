import logging
from typing import Union, List, Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
import os

from PIL import Image

from config import cfg
from generation.generate import rendering_agent
from module.geometric_constraint_correction_module.gccm_tools import to_data_url, decode_data_url_to_image
from module.geometric_constraint_correction_module.run import load_boxes, \
    build_points_prompt_with_boxes, plot_bounding_boxes, plot_points
from module.ssr import ssr_store

from utils.code_tools import extract_python_code
from agent.base_agent import BaseAgent
from utils.json_tools import clean_json_str
from task_prompts import load_all_prompts

prompts = load_all_prompts(cfg.PROMPT_BASE_DIR)


# 渲染智能体智能体类
class RenderingAgent(BaseAgent):
    """
    renderingAgent 类用于与 GPT API 交互，生成 Manim 动画代码。
    """

    """
    渲染智能体 Agent：构造时注入 client，内部创建可复用的 LC Agent。
    """

    def __init__(self, client, agent_name: str = "rendering_agent",
                 global_system_prompt: str = "你是一个专业的动画生成助理。需要生成视频时请调用工具 generate_video_mcp。",
                 tools: Optional[List] = None):
        super().__init__(agent_name)
        self.client = client
        self.agent_name = agent_name
        self.global_system_prompt = global_system_prompt
        self.tools = tools or []
        self.agent = self._create_agent()

    def _create_agent(self):
        return create_agent(
            model=self.client,
            tools=self.tools,
            system_prompt=self.global_system_prompt,
        )
    # 渲染智能体 生成 场景代码 和 初始几何结构信息
    async def code_generation(self, task_name="code_generation"):
        try:

            user_input = f"""
                       topic: {ssr_store.get_val("topic")} \n
                       description: {ssr_store.get_val("description")} \n
                       scene_plan: {ssr_store.get_val("scene_plan")} \n
                       scene_vision_storyboard: {ssr_store.get_val("scene_vision_storyboard")} \n
                       scene_implementation: {ssr_store.get_val("scene_implementation")} \n
                       scene_technical_implementation: {ssr_store.get_val("scene_technical_implementation")} \n
                       rag_information: {ssr_store.get_val("rag_information")} \n
                       "geometric_parameter_control_module_information": {ssr_store.get_val("geometric_parameter_control_module_information")} \n
                       scene_animation: {ssr_store.get_val("scene_animation")} \n
                       scene_narration: {ssr_store.get_val("scene_narration")} \n
            """
            system_prompt = prompts[f"{self.agent_name}_prompts"][task_name]
            response_content = await self.execute_task(self.agent, system_prompt=system_prompt, user_input=user_input, json_mode=False,image_list=None)
            print("渲染智能体 code_generation 响应内容: {}".format(response_content))
            if response_content:
                scene_code = extract_python_code(response_content)  # 提取响应中的 Python 代码

                user_input = user_input + f"\nscene_code:\n{scene_code}\n"
                geometric_structure_extraction = await self.geometric_structure_extraction(user_input)
                return scene_code,geometric_structure_extraction
            else:
                logging.error(
                    "Failed to 场景代码 和 初始几何结构信息."
                )
                return ""
        except Exception as e:
            logging.error(f"渲染智能体生成 场景代码 和 初始几何结构信息 失败: {e}")
            return ""


    # 渲染智能体 生成 修复后场景代码 和 校正后几何结构信息
    # geometric_structure_extraction 函数 提取几何结构校正模块信息
    async def fix_error(self, task_name="fix_error"):
        try:
            user_input = f"""
                topic: {ssr_store.get_val("topic")}\n
                description: {ssr_store.get_val("description")}\n
                scene_plan: {ssr_store.get_val("scene_plan")} \n
                scene_vision_storyboard: {ssr_store.get_val("scene_vision_storyboard")} \n
                scene_implementation: {ssr_store.get_val("scene_implementation")} \n
                scene_animation: {ssr_store.get_val("scene_animation")} \n
                scene_narration: {ssr_store.get_val("scene_narration")} \n
                scene_code：{ssr_store.get_val("scene_code")} \n
                error_message：\n{ssr_store.get_val("error_message")}\n
                geometric_constraint_correction_module_information：{ssr_store.get_val("geometric_constraint_correction_module_information")}\n
                rag_information：{ssr_store.get_val("rag_information")}\n
            """
            system_prompt = prompts[f"{self.agent_name}_prompts"][task_name]
            response_content = await self.execute_task(self.agent, system_prompt=system_prompt, user_input=user_input,
                                                       json_mode=False, image_list=None)
            if response_content:
                scene_code = extract_python_code(response_content)  # 提取响应中的 Python 代码
                geometric_structure_extraction_corrected = await self._geometric_structure_extraction(user_input)
                return scene_code,geometric_structure_extraction_corrected
            else:
                logging.error(
                    "Failed to 修复后场景代码 和 校正后几何结构信息."
                )
                return ""
        except Exception as e:
            logging.error(f"渲染智能体生成 修复后场景代码 和 校正后几何结构信息 失败: {e}")
            return ""

    # 渲染智能体 调用 几何约束校正模块 生成 几何约束校正模块信息
    async def get_geometric_constraint_correction_module_information(self,
                                                task_name="get_geometric_constraint_correction_module_information"):
        try:
            image_list=ssr_store.get_val("base64_list")
            animation_code=ssr_store.get_val("scene_code")
            geometric_structure_extraction=ssr_store.get_val("geometric_structure_extraction")
            system_prompt = """你是 **Geometric Constraint Correction Module (GCCM)**。你的任务是审核 Manim 动画帧，检查是否存在图像错位、缺失或其他问题，并结合 Manim 代码提出修改建议。请仔细检查每一帧图像，确保其与 Manim 代码中的几何结构和动画描述一致。对于发现的问题，请提供具体的修改建议，以便改进动画效果。你的输出应包含审核结果和反思建议，格式为纯文本，不包含任何代码块或多余的标记。"""
            all_image_list, image_list_base64 = await self.process_image_list(image_list)
            user_input = f"""
            这是manim代码：{animation_code}，请检查动画帧是否存在图像错位、缺失或其他问题，请结合manim代码提出修改建议。
            这个是初始几何信息：{geometric_structure_extraction}
            """
            response_content = await self.execute_task_list(self.agent, system_prompt=system_prompt,
                                                            user_input=user_input,
                                                            json_mode=False, image_list=all_image_list)
            if response_content:
                return response_content  # 返回审核和反思结果
            else:
                logging.error(
                    "Failed to review images and reflect on code."
                )  # 记录审核失败的错误
                return "渲染智能体出现问题，请稍后再试。"
        except Exception as e:
            logging.error(f"Error in get_geometric_constraint_correction_module_information: {e}")
            return "渲染智能体出现问题，请稍后再试。"

    # 渲染智能体 生成 初始/校正后 几何结构信息 （修复场景代码所使用，属于内部函数，不向外公开）
    async def _geometric_structure_extraction(self, user_input, task_name="geometric_structure_extraction"):
        try:
            user_input = user_input + f"""scene_code: {ssr_store.get_val("scene_code")}\n"""
            system_prompt = geometric_structure_extraction_system_prompt
            response_content = await self.execute_task(self.agent, system_prompt=system_prompt,
                                                       user_input=user_input,
                                                       json_mode=True, image_list=None)
            if response_content:
                return response_content  # 返回审核和反思结果
            else:
                logging.error(
                    "Failed to 初始/校正后 几何结构信息."
                )  # 记录审核失败的错误
                return "渲染智能体出现问题，请稍后再试。"
        except Exception as e:
            logging.error(f"Error in 初始/校正后 几何结构信息: {e}")
            return "渲染智能体出现问题，请稍后再试。"

    # 渲染智能体 审核图片内容（框）并反思代码（调用几何约束校正模块时使用，属于内部函数，不向外公开）
    async def _review_images_by_boxes_and_reflect_on_code(self, image_url,
                                                         task_name="review_images_by_boxes_and_reflect_on_code"):
        try:
            system_prompt = review_images_by_boxes_and_reflect_on_code_system_prompt
            user_input = """请你仔细检查图片内容，完成任务"""
            response_content = await self.execute_task_single(self.agent, system_prompt=system_prompt,
                                                              user_input=user_input,
                                                              json_mode=False, image_url=image_url)
            if response_content:
                return response_content  # 返回审核和反思结果
            else:
                logging.error(
                    "review_images_by_boxes_and_reflect_on_code."
                )  # 记录审核失败的错误
                return "渲染智能体出现问题，请稍后再试。"
        except Exception as e:
            logging.error(f"Error in review_images_by_boxes_and_reflect_on_code: {e}")
            return "渲染智能体出现问题，请稍后再试。"
    # 渲染智能体 审核图片内容（点）并反思代码（调用几何约束校正模块时使用，属于内部函数，不向外公开）
    async def _review_images_by_points_and_reflect_on_code(self, image_url, system_prompt,
                                                          task_name="review_images_by_points_and_reflect_on_code"):
        try:
            user_input = """请你仔细检查图片内容，完成任务"""
            response_content = await self.execute_task_single(self.agent, system_prompt=system_prompt,
                                                              user_input=user_input,
                                                              json_mode=False, image_url=image_url)
            if response_content:
                return response_content  # 返回审核和反思结果
            else:
                logging.error(
                    "Failed to review_images_by_points_and_reflect_on_code."
                )  # 记录审核失败的错误
                return "渲染智能体出现问题，请稍后再试。"
        except Exception as e:
            logging.error(f"Error in review_images_by_points_and_reflect_on_code: {e}")
            return "渲染智能体出现问题，请稍后再试。"

    # 子类实现父类的钩子方法
    def call_gpt_api(self, messages, agent, session_id, **kwargs):
        """
        同步调用 OpenAI 或 Azure API，返回 ChatCompletionMessage。
        """
        try:
            result = agent.invoke(
                {"messages": messages},
                config={"configurable": {"session_id": session_id}} if session_id else None,
            )
            messages = result['messages']
            ai_message = next((m for m in messages if isinstance(m, AIMessage)), None)
            return ai_message
        except Exception as e:
            logging.error(f"API 调用失败: {e}")
            return None


    async def process_image_list(
                                self,
                                 image_list,
                                 output_dir="./outputs",
                                 thumbnail=(640, 640),
                                 name_key="name"):
        """
        逐帧处理：每张图单独【先框→再点】；
        返回：bbox_point_list（json+路径） 和 image_list_base64（含三种base64url）。
        """
        os.makedirs(output_dir, exist_ok=True)

        bbox_point_list = []
        image_list_base64 = []  # 每个元素包含原图、bbox图、points图的 base64 data url

        for idx, item in enumerate(image_list, start=1):
            url = item.get("url") if isinstance(item, dict) else str(item)
            if not url:
                print(f"⚠️ 第 {idx} 张无 url，跳过")
                continue

            # === 第一次检测框 ===
            boxes_response = await rendering_agent.review_images_by_boxes_and_reflect_on_code(url)
            boxes_response_json = clean_json_str(boxes_response)
            det_boxes = load_boxes(boxes_response)
            points_prompt = build_points_prompt_with_boxes(det_boxes)

            # === 第二次关键点 ===
            points_response = await rendering_agent.review_images_by_points_and_reflect_on_code(url,points_prompt)
            points_response_json = clean_json_str(points_response)

            # === 解码原图 ===
            try:
                data_url = to_data_url(url)
                base = decode_data_url_to_image(data_url).convert("RGBA")
            except Exception as e:
                print(f"⚠️ 第 {idx} 张图片解码失败：{e}，跳过");
                continue

            base.thumbnail(list(thumbnail), Image.Resampling.LANCZOS)

            # === 文件名 ===
            name = item.get(name_key) if isinstance(item, dict) else None
            if not name:
                try:
                    basename = os.path.basename(url.split("?")[0])
                    name = os.path.splitext(basename)[0] or f"frame_{idx:03d}"
                except:
                    name = f"frame_{idx:03d}"
            ext = ".png"

            # === 各版本图 ===
            from io import BytesIO
            import base64

            def _img_to_data_url(img, fmt="PNG"):
                buf = BytesIO()
                img.save(buf, format=fmt)
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                mime = f"image/{fmt.lower()}"
                return f"data:{mime};base64,{b64}"

            # 原图
            orig_url = _img_to_data_url(base)

            # 画框图
            img_bbox = base.copy()
            plot_bounding_boxes(img_bbox, boxes_response)
            bbox_url = _img_to_data_url(img_bbox)

            # 关键点图
            img_pts = base.copy()
            plot_points(img_pts, points_response)
            pts_url = _img_to_data_url(img_pts)

            bbox_point_list.append({
                f"{idx}": {
                    "bbox": {"bbox_json": boxes_response_json},
                    "points": {"points_json": points_response_json}
                }
            })

            image_list_base64.append({
                "index": idx,
                "name": name,
                "original": {"url": orig_url},
                "bbox": {"url": bbox_url},
                "points": {"url": pts_url},
            })

            print(f"✅ 第 {idx} 张完成（base64url生成成功）")

        return bbox_point_list, image_list_base64

geometric_structure_extraction_system_prompt = """
你是 **Geometric Structure Extractor (Keypoint & Edge Extractor)**。你的输出将用于后续误差计算与多智能体框架对比评估，因此必须稳定、可复现、结构化、可机器解析。你只能输出一个 JSON 对象，禁止输出任何其它内容（包括解释、markdown、标题、代码块围栏、表格、自然语言、前后缀文字、空行）。你的回复必须能被 `json.loads()` 直接解析。

输入（固定）：

* topic: `{topic}`（必填）
* description: `{description}`（必填）
* scene_code: `{scene_code}`（可选，可能为空字符串、缺失、或仅包含空白）

单条输出约束（强制）：

* 一次只处理并输出一个 topic 的结果。
* 若输入包含多个 topic/列表/批量数据：只处理第一个条目；或只处理与 `{topic}` 精确匹配的条目（若 `{topic}` 已明确给出），其余忽略。
* 最终只输出一个 JSON 对象（不是数组，不是双列表）。

任务（只做抽取/推测）：

* 从 scene_code（若提供且可静态解析）并结合 topic/description（用于对象选择与补充规则）抽取/推测图形的关键点坐标 points 与边长 edges.length。
* 你只做提取/推测，不做误差计算、对比、评价，不输出角度/面积/证明/约束结论/字幕排版/动画信息等任何无关内容。

输出格式（必须严格一致，标准 JSON）：
你的最终输出必须且只允许包含如下字段结构（字段名必须一致）：
{
"schema_version": "geo-kp-edge-1.0",
"unit": "manim_unit",
"topic": "<copy topic verbatim>",
"description": "<copy description verbatim>",
"mode": "from_code" 或 "hypothesized",
"objects": [
{
"id": "OBJ_1",
"type": "<string>",
"points": { "<pointName>": [x,y,z] 或 null, ... },
"edges": [
{ "id": "<edgeId>", "p1": "<pointName>", "p2": "<pointName>", "length": <number|null>, "source": "code_direct|computed|hypothesized" }
],
"source": "code" 或 "hypothesized",
"confidence": <number 0..1>,
"notes": "<one short sentence>"
}
]
}

标准 JSON 硬规则（必须遵守）：

* 必须使用双引号；禁止注释；禁止尾随逗号；禁止 NaN/Infinity；禁止 Python 的 None/True/False。
* 所有数值必须为有限实数（float/int 均可），不得输出字符串数字。
* 坐标统一为 [x,y,z]，z 默认 0。
* 不可确定的点坐标必须为 null，禁止臆造；并在 notes 用一句话说明原因。
* edges.length 只有在两端点坐标都已知时才能是数值，否则必须为 null。
* 边长计算用欧氏距离：sqrt((x2-x1)^2+(y2-y1)^2+(z2-z1)^2)。
* 只输出与主题强相关且最少够用的对象：主图形优先，其次必要辅助线（如对角线被明确提到）。
* 必须忽略所有非几何主体：Text/MathTex/Tex、字幕框、背景矩形、安全框、排版辅助对象、装饰物。
* 命名必须稳定：相同输入必须产生相同点名/边名/对象顺序/边顺序。
* 最终输出禁止包含任何额外字符：禁止 ```json 围栏、禁止任何前缀或后缀、禁止换行外的任何附加文本。

模式判定（强制）：

* 仅当 scene_code 为非空且非纯空白字符串，并且包含白名单几何构造关键字之一（"Polygon("、"Line("、"Square("、"Rectangle("、"Circle("）时，mode 才能为 "from_code"。
* 否则 mode 必须为 "hypothesized"。
* mode="from_code" 时，允许 objects 中同时存在 source="code" 与 source="hypothesized" 的对象（仅在“补充 hypothesized 对象”规则触发时）；但每个对象的 source 必须真实反映来源。
* mode="hypothesized" 时，objects 中所有对象的 source 必须为 "hypothesized"。

命名与排序规范（强制，可复现）：

* objects 顺序：主图形永远是 OBJ_1；若必须输出多个对象（仅在 Euler 类主题无代码推测时），固定 OBJ_1=axes，OBJ_2=circle_like_polygon。
* points 命名（按 type）：

  * right_triangle: O,U,R
  * triangle: A,B,C
  * kite_quadrilateral: T,R,B,L
  * quadrilateral: A,B,C,D
  * axes: X1,X2,Y1,Y2
  * circle_like_polygon: O,E,N,W,S
  * segment/line: A,B（或保留上下文已有点名）
* edges 命名：edge.id 固定为 p1+p2（例如 OU、TR、X1X2）。
* 多边形边顺序固定：

  * triangle: AB, BC, CA
  * quadrilateral: AB, BC, CD, DA
  * kite_quadrilateral: TR, RB, BL, LT
* 对角线/辅助线顺序：在多边形边之后输出；kite 对角线固定先 TB 再 RL。

edges.source 语义（强制统一）：

* "code_direct": 端点坐标可从 scene_code 静态解析得到
* "hypothesized": 端点坐标来自无代码时的规范化推测库
* "computed": 仅在 from_code 下，当端点来自代码解析但边是由闭合关系补齐出来时可用；若不确定则用 "code_direct"

三源融合规则（scene_code + topic + description，强制）：

* 当 mode="from_code" 时，你必须同时利用 scene_code、topic、description 来决定输出哪些对象，但坐标与长度必须遵守以下优先级与约束：

  1. 对象选择：先从 scene_code 静态抽取白名单几何对象候选；再用 topic/description 判断主题强相关核心结构；最终只保留“与主题强相关且最少够用”的对象。
  2. 坐标来源优先级：若 scene_code 可静态解析出某点坐标，必须用该坐标；若 scene_code 无法静态确定该点，点必须为 null，禁止用 topic/description 去补坐标。
  3. 边长规则不变：仅当两端点坐标都已知（非 null）才计算 length，否则 length=null；禁止用 topic/description 的符号量（a,b,c,d1,d2）去填数值长度。
  4. 允许补充 hypothesized 对象的唯一条件：仅当 scene_code 中不存在任何能表达主题核心结构的可解析对象，且 topic/description 明确命中关键词映射（见 hypothesized 规则）时，才允许额外增加一个 source="hypothesized" 的核心模板对象；该对象必须使用默认规范化坐标库；confidence 必须 ≤ 0.6；notes 必须一句话说明已补充模板原因。
  5. 符号变量（a,b,c,d1,d2）处理：禁止输出 a=,b=,c=,d1=,d2= 等字段或语义映射；这些只允许影响“是否需要输出相应核心结构对象”，不得影响坐标与边长数值。

A) from_code：静态抽取规则（不执行代码，仅文本静态）
你只能抽取以下白名单几何类型：

* Polygon(...)
* Line(p1,p2)
* Square(side_length=...)
* Rectangle(width=..., height=...)
* Circle(radius=...)（谨慎：仅当圆心/半径可静态确定时才输出离散近似点，否则点为 null 或不输出该对象）

可静态解析的点表达式（严格白名单）：

* ORIGIN, UP, DOWN, LEFT, RIGHT
* 标量乘法：k*UP / k*RIGHT 等（k 必须为显式数值）
* 线性组合：上述项用 + 或 - 连接（如 LEFT+UP*0.5、2*UP+2*RIGHT）
  禁止解析：
* 任何函数调用/变量引用/运行时方法：get_vertices()、point_at_angle(...)、对象属性读取、未定义变量等。一旦端点含这些，端点坐标为 null。

坐标映射（写死）：

* ORIGIN=[0,0,0], UP=[0,1,0], DOWN=[0,-1,0], LEFT=[-1,0,0], RIGHT=[1,0,0]

允许的静态变换：

* shift(白名单向量)：对该对象所有已知 points 做平移。
  禁止/不安全变换（遇到则相关点置 null 并降低 confidence）：
* rotate、scale、move_to(非 ORIGIN 且不可静态确定)、依赖运行时状态的变换。

Polygon 处理：

* 若 Polygon 的每个顶点都可静态解析为坐标，则按出现顺序命名 A,B,C,...；若 topic/description 明确是 right_triangle 且顶点恰好构成直角结构，仍不得改名为 O,U,R，必须遵守 Polygon 命名为 A,B,C 的稳定规则。
* edges：输出闭合边并计算 length；triangle 为 AB,BC,CA；quadrilateral 为 AB,BC,CD,DA；其他多边形按顺序连边并闭合，edge.id 使用 p1+p2。
* 若任何顶点不可解析，则该顶点为 null；涉及该顶点的边 length 必须为 null。

Line 处理：

* 两端点都可静态解析则命名 A,B 并计算 length；否则对应点为 null，length=null。

Square/Rectangle 顶点推断（仅当没有不安全变换且中心可确定为 ORIGIN）：

* Square(side_length=s)：
  A=[s/2,s/2,0], B=[s/2,-s/2,0], C=[-s/2,-s/2,0], D=[-s/2,s/2,0]；edges：AB,BC,CD,DA
* Rectangle(width=w,height=h)：
  A=[w/2,h/2,0], B=[w/2,-h/2,0], C=[-w/2,-h/2,0], D=[-w/2,h/2,0]；edges：AB,BC,CD,DA
* 若 side_length/width/height 不可静态取值，则点为 null，边长为 null。

from_code 的 confidence 规则：

* 若主对象所有 points 均非 null 且所有边长可算：confidence ∈ [0.85,0.95]
* 若存在部分 null 或出现不安全变换：confidence ∈ [0.4,0.8]
* 若几乎不可解析：confidence ∈ [0.1,0.3]

B) hypothesized：无代码推测规则（只用 topic/description）

* mode 必须为 "hypothesized"
* objects 中所有对象 source 必须为 "hypothesized"
* confidence 必须在 [0.2,0.6]
* 使用“默认规范化坐标库”生成 points，并计算 edges.length（所有点已知则必须计算所有边长）

关键词到图形映射（hypothesized 必用，写死）：

* 命中：Pythagorean / a^2+b^2=c^2 / right triangle / hypotenuse / 勾股 / 直角三角形 => 输出一个对象 type="right_triangle"
* 命中：kite / Kite_Quadrilateral / 风筝 / perpendicular diagonals / bisected / 对角线 => 输出一个对象 type="kite_quadrilateral"；若命中对角线关键词则加 TB, RL
* 命中：Euler / e^(ix) / cos / sin / complex plane / unit circle / π => 输出两个对象：OBJ_1 type="axes"，OBJ_2 type="circle_like_polygon"
* 若未命中任何映射 => 输出一个对象 type="unknown"

默认规范化坐标库（必须使用）：

* right_triangle：
  points：O=[0,0,0], U=[0,2,0], R=[2,0,0]
  edges：OU, OR, UR
* kite_quadrilateral：
  points：T=[0,1,0], R=[1,0.5,0], B=[0,-1,0], L=[-1,0.5,0]
  edges：TR, RB, BL, LT
  可选对角线：TB, RL
* axes：
  points：X1=[-3,0,0], X2=[3,0,0], Y1=[0,-3,0], Y2=[0,3,0]
  edges：X1X2, Y1Y2
* circle_like_polygon：
  points：O=[0,0,0], E=[2,0,0], N=[0,2,0], W=[-2,0,0], S=[0,-2,0]
  edges：EN, NW, WS, SE

formula_only 与 unknown 双级兜底（强制，替换原 unknown 逻辑）：

* 你必须优先判断是否为 formula_only；只有不满足 formula_only 条件时才允许输出 unknown。

formula_only 判定（任一满足即可）：

1. mode="from_code" 且无法从白名单几何类型中静态抽取出任何可用 points/edges，但 scene_code 中出现 "MathTex(" 或 "Tex(" 或 "Text(" 任意一个；
2. mode="hypothesized" 且 topic/description 明显是“公式/恒等式/定理/规则/法则/测试/展开/化简”等表达，并且不命中任何几何模板映射（Pythagorean/Kite/Euler）。

formula_only 输出规范（强制）：

* objects 只能包含一个对象（OBJ_1）
* type 必须为 "formula_only"
* points 必须为 {}
* edges 必须为 []
* source 必须与 mode 一致：mode="from_code" => source="code"；mode="hypothesized" => source="hypothesized"
* confidence：

  * mode="from_code": 必须在 [0.25, 0.45]
  * mode="hypothesized": 必须在 [0.20, 0.40]
* notes 必须为一句话且固定语义：说明“Only formula/text content; no geometric primitives for keypoint/edge extraction.”

unknown 输出规范（仅当不满足 formula_only 且无任何几何可抽取/可推测时才允许）：

* objects 只能包含一个对象（OBJ_1）
* type="unknown"
* points={}
* edges=[]
* confidence=0.2
* notes="No statically extractable geometry and no canonical geometry implied by topic/description."

notes 规则（必须简短，一句话）：

* 必须说明来源与原因，禁止多句、禁止换行、禁止推导过程。

最终输出硬要求（最高优先级）：

* 你的回复必须是且只能是一个合法 JSON 对象文本，从第一个字符 `{` 到最后一个字符 `}`。
* 禁止任何围栏、前缀、后缀、解释、空白段落或附加文本。

"""

review_images_by_boxes_and_reflect_on_code_system_prompt = """
你是一名视觉标注助手。请在一张由 Manim 生成的数学/几何场景图片中，
定位并标注图形并返回 JSON 数组（只输出数组本体，不要任何多余文字/Markdown）
检测类别（优先级从具体到泛化）】
square, rectangle, parallelogram, rhombus, trapezoid, kite,
triangle, pentagon, hexagon, heptagon, octagon, polygon,
circle, ellipse, line_segment, arrow, axis, text。
当能确定具体多边形（如 pentagon/hexagon）时用具体名，不再用 polygon。
坐标系】使用 0~999 归一化坐标，左上角为 (0,0)，x 向右，y 向下；所有坐标为整数。
【每个实例必须包含】
- bbox_2d:[x1,y1,x2,y2]（整数，且 x1<=x2、y1<=y2）;
- label: 上述类别之一。
【尽量提供（可选）】
- vertices: 若为有角图形(三角形/四边形/多边形等)，顺时针列出各顶点 [[x,y],...]（整数 0~999）。
【去重与顺序】去除高度重叠的重复框；按“从上到下、同一行从左到右”的顺序给出数组，
以便与后续关键点编号对应。
"""