import base64
import logging
import os
from io import BytesIO
from typing import List, Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from PIL import Image

from config import cfg
from multi_agent_collaborative_scene_generation_system.module.gccm import (
    load_boxes,
    build_points_prompt_with_boxes,
    plot_bounding_boxes,
    plot_points,
    to_data_url,
    decode_data_url_to_image,
)
from multi_agent_collaborative_scene_generation_system.model.ssrm import ssrm
from multi_agent_collaborative_scene_generation_system.utils import (
    extract_python_code,
    clean_json_str,
)
from multi_agent_collaborative_scene_generation_system.multi_agent import (
    BaseAgent,
    load_all_prompts,
)


prompts = load_all_prompts(cfg.PROMPT_BASE_DIR)


class RenderingAgent(BaseAgent):
    def __init__(
        self,
        client,
        agent_name: str = "rendering_agent",
        global_system_prompt: str = "You are a professional animation generation assistant.",
        tools: Optional[List] = None,
    ):
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

    async def code_generation(self, task_name: str = "code_generation"):
        try:
            user_input = f"""
topic: {ssrm.get_val("topic")}
description: {ssrm.get_val("description")}
scene_plan: {ssrm.get_val("scene_plan")}
scene_vision_storyboard: {ssrm.get_val("scene_vision_storyboard")}
scene_implementation: {ssrm.get_val("scene_implementation")}
scene_technical_implementation: {ssrm.get_val("scene_technical_implementation")}
rag_information: {ssrm.get_val("rag_information")}
geometric_parameter_control_module_information: {ssrm.get_val("geometric_parameter_control_module_information")}
scene_animation: {ssrm.get_val("scene_animation")}
scene_narration: {ssrm.get_val("scene_narration")}
""".strip()

            system_prompt = prompts[f"{self.agent_name}_prompts"][task_name]

            response_content = await self.execute_task(
                self.agent,
                system_prompt=system_prompt,
                user_input=user_input,
                json_mode=False,
                image_list=None,
            )

            print(f"RenderingAgent code_generation response: {response_content}")

            if not response_content:
                logging.error(
                    "Failed to generate scene code and geometric structure information."
                )
                return ""

            scene_code = extract_python_code(response_content)

            extraction_input = user_input + f"\nscene_code:\n{scene_code}\n"
            geometric_structure_extraction = await self._geometric_structure_extraction(
                extraction_input
            )

            return scene_code, geometric_structure_extraction

        except Exception as e:
            logging.error(
                "RenderingAgent failed to generate scene code and initial "
                f"geometric structure information: {e}"
            )
            return ""

    async def fix_error(self, task_name: str = "fix_error"):
        try:
            user_input = f"""
topic: {ssrm.get_val("topic")}
description: {ssrm.get_val("description")}
scene_plan: {ssrm.get_val("scene_plan")}
scene_vision_storyboard: {ssrm.get_val("scene_vision_storyboard")}
scene_implementation: {ssrm.get_val("scene_implementation")}
scene_animation: {ssrm.get_val("scene_animation")}
scene_narration: {ssrm.get_val("scene_narration")}
scene_code: {ssrm.get_val("scene_code")}
error_message:
{ssrm.get_val("error_message")}
geometric_constraint_correction_module_information: {ssrm.get_val("geometric_constraint_correction_module_information")}
rag_information: {ssrm.get_val("rag_information")}
""".strip()

            system_prompt = prompts[f"{self.agent_name}_prompts"][task_name]

            response_content = await self.execute_task(
                self.agent,
                system_prompt=system_prompt,
                user_input=user_input,
                json_mode=False,
                image_list=None,
            )

            if not response_content:
                logging.error(
                    "Failed to generate fixed scene code and corrected geometric "
                    "structure information."
                )
                return ""

            scene_code = extract_python_code(response_content)

            extraction_input = user_input + f"\nfix_error_code:\n{scene_code}\n"
            geometric_structure_extraction_corrected = (
                await self._geometric_structure_extraction(extraction_input)
            )

            return scene_code, geometric_structure_extraction_corrected

        except Exception as e:
            logging.error(
                "RenderingAgent failed to generate fixed scene code and corrected "
                f"geometric structure information: {e}"
            )
            return ""

    async def get_geometric_constraint_correction_module_information(
        self,
        task_name: str = "get_geometric_constraint_correction_module_information",
    ):
        try:
            image_list = ssrm.get_val("base64_list") or []
            animation_code = ssrm.get_val("scene_code")
            geometric_structure_extraction = ssrm.get_val(
                "geometric_structure_extraction"
            )

            system_prompt = (
                "你是 **Geometric Constraint Correction Module (GCCM)**。"
                "你的任务是审核 Manim 动画帧，检查是否存在图像错位、缺失或其他问题，"
                "并结合 Manim 代码提出修改建议。请仔细检查每一帧图像，确保其与 "
                "Manim 代码中的几何结构和动画描述一致。对于发现的问题，请提供具体的修改建议，"
                "以便改进动画效果。你的输出应包含审核结果和反思建议，格式为纯文本，"
                "不包含任何代码块或多余的标记。"
            )

            bbox_point_list, image_list_base64 = await self.process_image_list(
                image_list
            )

            user_input = f"""
这是 manim 代码：
{animation_code}

这是初始几何信息：
{geometric_structure_extraction}

这是逐帧检测得到的边界框和关键点信息：
{bbox_point_list}

请检查动画帧是否存在图像错位、元素缺失、几何结构不一致或其他问题，并结合 manim 代码提出修改建议。
""".strip()

            response_content = await self.execute_task_list(
                self.agent,
                system_prompt=system_prompt,
                user_input=user_input,
                json_mode=False,
                image_list=image_list_base64,
            )

            if response_content:
                return response_content

            logging.error("Failed to review images and reflect on code.")
            return "RenderingAgent encountered an issue. Please try again later."

        except Exception as e:
            logging.error(
                "Error in get_geometric_constraint_correction_module_information: "
                f"{e}"
            )
            return "RenderingAgent encountered an issue. Please try again later."

    async def _geometric_structure_extraction(
        self,
        user_input,
        task_name: str = "geometric_structure_extraction",
    ):
        try:
            user_input = user_input + f"\nscene_code: {ssrm.get_val('scene_code')}\n"
            system_prompt = geometric_structure_extraction_system_prompt

            response_content = await self.execute_task(
                self.agent,
                system_prompt=system_prompt,
                user_input=user_input,
                json_mode=True,
                image_list=None,
            )

            if response_content:
                return response_content

            logging.error("Failed to extract geometric structure information.")
            return "RenderingAgent encountered an issue. Please try again later."

        except Exception as e:
            logging.error(f"Error in geometric structure extraction: {e}")
            return "RenderingAgent encountered an issue. Please try again later."

    async def _review_images_by_boxes_and_reflect_on_code(
        self,
        image_url,
        task_name: str = "review_images_by_boxes_and_reflect_on_code",
    ):
        try:
            system_prompt = review_images_by_boxes_and_reflect_on_code_system_prompt
            user_input = "Please carefully inspect the image content and complete the task."

            response_content = await self.execute_task_single(
                self.agent,
                system_prompt=system_prompt,
                user_input=user_input,
                json_mode=False,
                image_url=image_url,
            )

            if response_content:
                return response_content

            logging.error("Failed to review images by boxes.")
            return "RenderingAgent encountered an issue. Please try again later."

        except Exception as e:
            logging.error(f"Error in review_images_by_boxes_and_reflect_on_code: {e}")
            return "RenderingAgent encountered an issue. Please try again later."

    async def _review_images_by_points_and_reflect_on_code(
        self,
        image_url,
        system_prompt,
        task_name: str = "review_images_by_points_and_reflect_on_code",
    ):
        try:
            user_input = "Please carefully inspect the image content and complete the task."

            response_content = await self.execute_task_single(
                self.agent,
                system_prompt=system_prompt,
                user_input=user_input,
                json_mode=False,
                image_url=image_url,
            )

            if response_content:
                return response_content

            logging.error("Failed to review images by points.")
            return "RenderingAgent encountered an issue. Please try again later."

        except Exception as e:
            logging.error(f"Error in review_images_by_points_and_reflect_on_code: {e}")
            return "RenderingAgent encountered an issue. Please try again later."

    def call_gpt_api(self, messages, agent, session_id, **kwargs):
        try:
            result = agent.invoke(
                {"messages": messages},
                config={
                    "configurable": {
                        "session_id": session_id,
                    }
                }
                if session_id
                else None,
            )

            messages = result["messages"]
            ai_message = next((m for m in messages if isinstance(m, AIMessage)), None)

            return ai_message

        except Exception as e:
            logging.error(f"API call failed: {e}")
            return None

    async def process_image_list(
        self,
        image_list,
        output_dir: str = "./outputs",
        thumbnail=(640, 640),
        name_key: str = "name",
    ):
        os.makedirs(output_dir, exist_ok=True)

        bbox_point_list = []
        image_list_base64 = []

        for idx, item in enumerate(image_list, start=1):
            url = item.get("url") if isinstance(item, dict) else str(item)

            if not url:
                print(f"Frame {idx} has no url. Skipped.")
                continue

            boxes_response = await self._review_images_by_boxes_and_reflect_on_code(
                url
            )
            boxes_response_json = clean_json_str(boxes_response)

            det_boxes = load_boxes(boxes_response)
            points_prompt = build_points_prompt_with_boxes(det_boxes)

            points_response = await self._review_images_by_points_and_reflect_on_code(
                url,
                points_prompt,
            )
            points_response_json = clean_json_str(points_response)

            try:
                data_url = to_data_url(url)
                base = decode_data_url_to_image(data_url).convert("RGBA")
            except Exception as e:
                print(f"Frame {idx} image decoding failed: {e}. Skipped.")
                continue

            base.thumbnail(list(thumbnail), Image.Resampling.LANCZOS)

            name = item.get(name_key) if isinstance(item, dict) else None

            if not name:
                try:
                    basename = os.path.basename(url.split("?")[0])
                    name = os.path.splitext(basename)[0] or f"frame_{idx:03d}"
                except Exception:
                    name = f"frame_{idx:03d}"

            def _img_to_data_url(img, fmt: str = "PNG"):
                buf = BytesIO()
                img.save(buf, format=fmt)
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                mime = f"image/{fmt.lower()}"
                return f"data:{mime};base64,{b64}"

            orig_url = _img_to_data_url(base)

            img_bbox = base.copy()
            plot_bounding_boxes(img_bbox, boxes_response)
            bbox_url = _img_to_data_url(img_bbox)

            img_pts = base.copy()
            plot_points(img_pts, points_response)
            pts_url = _img_to_data_url(img_pts)

            bbox_point_list.append(
                {
                    f"{idx}": {
                        "bbox": {
                            "bbox_json": boxes_response_json,
                        },
                        "points": {
                            "points_json": points_response_json,
                        },
                    }
                }
            )

            image_list_base64.append(
                {
                    "index": idx,
                    "name": name,
                    "original": {
                        "url": orig_url,
                    },
                    "bbox": {
                        "url": bbox_url,
                    },
                    "points": {
                        "url": pts_url,
                    },
                }
            )

            print(f"Frame {idx} completed. Base64 URLs generated successfully.")

        return bbox_point_list, image_list_base64

geometric_structure_extraction_system_prompt = """
You are a Geometric Structure Extractor (Keypoint & Edge Extractor). Your output will be used for subsequent error calculation and multi-agent framework comparison, so it must be stable, reproducible, structured, and machine-parseable. You must output only one JSON object. Do not output anything else, including explanations, Markdown, titles, code fences, tables, natural language, prefixes, suffixes, or blank paragraphs. Your response must be directly parseable by json.loads().

Fixed input:

* topic: `{topic}` required
* description: `{description}` required
* scene_code: `{scene_code}` optional; it may be an empty string, missing, or whitespace only

Single-output constraints:

* Process and output the result for only one topic at a time.
* If the input contains multiple topics, lists, or batch data, process only the first item; or process only the item that exactly matches `{topic}` if `{topic}` is explicitly provided. Ignore the rest.
* The final output must be one JSON object only. It must not be an array or a pair of lists.

Task:

* Extract or infer the geometric keypoint coordinates in points and edge lengths in edges.length from scene_code if it is provided and statically parseable, while using topic and description only for object selection and supplementary rules.
* Only perform extraction or inference. Do not calculate errors, compare results, evaluate quality, output angles, areas, proofs, constraint conclusions, subtitle layout, animation information, or any irrelevant content.

Required output format:
Your final output must strictly follow this field structure and field names:
{
  "schema_version": "geo-kp-edge-1.0",
  "unit": "manim_unit",
  "topic": "<copy topic verbatim>",
  "description": "<copy description verbatim>",
  "mode": "from_code" or "hypothesized",
  "objects": [
    {
      "id": "OBJ_1",
      "type": "<string>",
      "points": {
        "<pointName>": [x, y, z] or null
      },
      "edges": [
        {
          "id": "<edgeId>",
          "p1": "<pointName>",
          "p2": "<pointName>",
          "length": <number|null>,
          "source": "code_direct|computed|hypothesized"
        }
      ],
      "source": "code" or "hypothesized",
      "confidence": <number from 0 to 1>,
      "notes": "<one short sentence>"
    }
  ]
}

Strict JSON rules:

* Use double quotes only. Do not use comments, trailing commas, NaN, Infinity, or Python-style None/True/False.
* All numeric values must be finite real numbers. Do not output numbers as strings.
* Coordinates must use [x, y, z]. The default z value is 0.
* If a point coordinate cannot be determined, it must be null. Do not invent coordinates. Explain the reason briefly in notes.
* edges.length may be numeric only when both endpoint coordinates are known. Otherwise it must be null.
* Edge length must be calculated by Euclidean distance: sqrt((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2).
* Output only the most relevant and minimal objects related to the topic. Prioritize the main geometric figure, then necessary auxiliary lines such as explicitly mentioned diagonals.
* Ignore all non-geometric elements, including Text, MathTex, Tex, subtitle boxes, background rectangles, safe frames, layout helpers, and decorative objects.
* Naming must be stable. The same input must produce the same point names, edge names, object order, and edge order.
* The final output must not contain any extra characters. Do not use JSON fences, prefixes, suffixes, or any additional text.

Mode selection:

* mode may be "from_code" only when scene_code is non-empty, not whitespace-only, and contains at least one geometry construction keyword from this whitelist: "Polygon(", "Line(", "Square(", "Rectangle(", "Circle(".
* Otherwise, mode must be "hypothesized".
* When mode="from_code", objects may include both source="code" and source="hypothesized" objects, but only when the supplementary hypothesized object rule is triggered. Each object's source must truthfully reflect its origin.
* When mode="hypothesized", all objects must have source="hypothesized".

Naming and ordering rules:

* Object order: the main figure is always OBJ_1. If multiple objects must be output only for Euler-like topics without code, use OBJ_1=axes and OBJ_2=circle_like_polygon.
* Point naming by type:
  * right_triangle: O, U, R
  * triangle: A, B, C
  * kite_quadrilateral: T, R, B, L
  * quadrilateral: A, B, C, D
  * axes: X1, X2, Y1, Y2
  * circle_like_polygon: O, E, N, W, S
  * segment/line: A, B, or preserve existing contextual point names
* edge.id must be p1+p2, such as OU or TR.
* Polygon edge order:
  * triangle: AB, BC, CA
  * quadrilateral: AB, BC, CD, DA
  * kite_quadrilateral: TR, RB, BL, LT
* Diagonal or auxiliary-line order: output them after polygon edges. For kite diagonals, output TB first, then RL.

edges.source semantics:

* "code_direct": endpoint coordinates can be statically parsed from scene_code
* "hypothesized": endpoint coordinates come from the default normalized template library
* "computed": use only in from_code mode when endpoints come from code parsing but an edge is completed by closure; if uncertain, use "code_direct"

Three-source fusion rules: scene_code + topic + description:

* When mode="from_code", use scene_code, topic, and description together to decide which objects to output, but coordinates and lengths must follow these priorities and constraints:
  1. Object selection: first extract candidate whitelist geometric objects from scene_code; then use topic and description to identify the topic-relevant core structure; finally keep only the topic-relevant and minimal objects.
  2. Coordinate priority: if a point coordinate can be statically parsed from scene_code, use it. If it cannot be statically determined from scene_code, set the point to null. Do not use topic or description to fill coordinates.
  3. Edge length rule remains unchanged: calculate length only when both endpoint coordinates are known. Otherwise, length must be null. Do not use symbolic quantities such as a, b, c, d1, or d2 from topic/description as numeric lengths.
  4. The only condition for adding a supplementary hypothesized object: if scene_code contains no parseable object that expresses the topic's core structure, and topic/description explicitly matches the keyword mapping in the hypothesized rules, then one additional source="hypothesized" core template object may be added. It must use the default normalized coordinate library. Its confidence must be <= 0.6. Its notes must briefly explain why a template was added.
  5. Symbolic variables such as a, b, c, d1, d2 must not be output as fields or semantic mappings. They may only influence whether the corresponding core structure object should be output. They must not influence numeric coordinates or lengths.

A) from_code static extraction rules:
Do not execute code. Only perform static text parsing.
You may extract only the following whitelist geometry types:

* Polygon(...)
* Line(p1, p2)
* Square(side_length=...)
* Rectangle(width=..., height=...)
* Circle(radius=...), cautiously; output discrete approximation points only when the center and radius can be statically determined. Otherwise set points to null or do not output the object.

Strict whitelist of statically parseable point expressions:

* ORIGIN, UP, DOWN, LEFT, RIGHT
* Scalar multiplication: k*UP, k*RIGHT, etc., where k must be an explicit number
* Linear combinations using + or - of the above terms, such as LEFT+UP*0.5 or 2*UP+2*RIGHT

Do not parse:

* Any function call, variable reference, runtime method, get_vertices(), point_at_angle(...), object attribute access, undefined variables, or runtime-dependent expression. If an endpoint contains these, set that endpoint coordinate to null.

Coordinate mapping:

* ORIGIN=[0,0,0], UP=[0,1,0], DOWN=[0,-1,0], LEFT=[-1,0,0], RIGHT=[1,0,0]

Allowed static transform:

* shift(whitelist vector): translate all known points of that object.

Unsafe transforms:

* rotate, scale, move_to with non-ORIGIN or non-statically determinable target, and any runtime-dependent transform. If encountered, set related points to null and lower confidence.

Polygon handling:

* If every Polygon vertex can be statically parsed, name vertices A, B, C, ... in appearance order. If topic/description clearly indicates a right triangle and the vertices form a right-angle structure, still do not rename them to O, U, R. Polygon naming must remain A, B, C for stability.
* Output closed edges and calculate lengths. For triangle use AB, BC, CA. For quadrilateral use AB, BC, CD, DA. For other polygons, connect vertices in order and close the polygon. edge.id must be p1+p2.
* If any vertex cannot be parsed, set that vertex to null. Any edge involving that vertex must have length=null.

Line handling:

* If both endpoints can be statically parsed, name them A and B and calculate length. Otherwise set the corresponding point to null and length=null.

Square/Rectangle vertex inference:

* Only infer vertices when there is no unsafe transform and the center can be determined as ORIGIN.
* Square(side_length=s):
  A=[s/2,s/2,0], B=[s/2,-s/2,0], C=[-s/2,-s/2,0], D=[-s/2,s/2,0]; edges: AB, BC, CD, DA
* Rectangle(width=w,height=h):
  A=[w/2,h/2,0], B=[w/2,-h/2,0], C=[-w/2,-h/2,0], D=[-w/2,h/2,0]; edges: AB, BC, CD, DA
* If side_length, width, or height cannot be statically obtained, points must be null and edge lengths must be null.

from_code confidence rules:

* If all main-object points are non-null and all edge lengths are calculable: confidence must be in [0.85, 0.95].
* If some points are null or unsafe transforms appear: confidence must be in [0.4, 0.8].
* If almost nothing is parseable: confidence must be in [0.1, 0.3].

B) hypothesized rules:

* mode must be "hypothesized".
* All objects must have source="hypothesized".
* confidence must be in [0.2, 0.6].
* Use the default normalized coordinate library to generate points and calculate edges.length. If all points are known, all edge lengths must be calculated.

Keyword-to-geometry mapping:

* Match Pythagorean, Pythagorean theorem, a^2+b^2=c^2, right triangle, or hypotenuse: output one object with type="right_triangle".
* Match kite, Kite_Quadrilateral, perpendicular diagonals, bisected diagonals, or diagonal: output one object with type="kite_quadrilateral". If diagonal keywords are matched, add TB and RL.
* Match Euler, Euler's formula, e^(ix), cos, sin, complex plane, unit circle, or pi: output two objects: OBJ_1 type="axes", OBJ_2 type="circle_like_polygon".
* If no mapping is matched: output one object with type="unknown".

Default normalized coordinate library:

* right_triangle:
  points: O=[0,0,0], U=[0,2,0], R=[2,0,0]
  edges: OU, OR, UR
* kite_quadrilateral:
  points: T=[0,1,0], R=[1,0.5,0], B=[0,-1,0], L=[-1,0.5,0]
  edges: TR, RB, BL, LT
  optional diagonals: TB, RL
* axes:
  points: X1=[-3,0,0], X2=[3,0,0], Y1=[0,-3,0], Y2=[0,3,0]
  edges: X1X2, Y1Y2
* circle_like_polygon:
  points: O=[0,0,0], E=[2,0,0], N=[0,2,0], W=[-2,0,0], S=[0,-2,0]
  edges: EN, NW, WS, SE

formula_only and unknown fallback:

* Always check formula_only first. Only output unknown if formula_only conditions are not met.

formula_only conditions:

1. mode="from_code" and no usable points/edges can be statically extracted from whitelist geometry types, but scene_code contains any of "MathTex(", "Tex(", or "Text(".
2. mode="hypothesized" and topic/description is clearly about a formula, identity, theorem, rule, law, test, expansion, simplification, or symbolic expression, and it does not match any geometry template mapping.

formula_only output rules:

* objects must contain exactly one object: OBJ_1.
* type must be "formula_only".
* points must be {}.
* edges must be [].
* source must match mode: mode="from_code" => source="code"; mode="hypothesized" => source="hypothesized".
* confidence:
  * mode="from_code": must be in [0.25, 0.45].
  * mode="hypothesized": must be in [0.20, 0.40].
* notes must be one sentence with this fixed meaning: "Only formula/text content; no geometric primitives for keypoint/edge extraction."

unknown output rules:

* Output unknown only when formula_only is not satisfied and no geometry can be extracted or hypothesized.
* objects must contain exactly one object: OBJ_1.
* type must be "unknown".
* points must be {}.
* edges must be [].
* confidence must be 0.2.
* notes must be "No statically extractable geometry and no canonical geometry implied by topic/description."

notes rules:

* notes must be short and one sentence only. Do not include line breaks or reasoning steps.

Highest-priority final output requirement:

* Your response must be exactly one valid JSON object, from the first character `{` to the last character `}`.
* Do not output fences, prefixes, suffixes, explanations, blank paragraphs, or any additional text.
"""

review_images_by_boxes_and_reflect_on_code_system_prompt = """
You are a visual annotation assistant. In a mathematical or geometric scene image generated by Manim, locate and annotate graphical elements, then return a JSON array. Output only the array itself. Do not output any extra text or Markdown.

Detection classes, ordered from specific to general:
square, rectangle, parallelogram, rhombus, trapezoid, kite,
triangle, pentagon, hexagon, heptagon, octagon, polygon,
circle, ellipse, line_segment, arrow, axis, text.

When a specific polygon type can be determined, such as pentagon or hexagon, use the specific class name instead of polygon.

Coordinate system:
Use normalized coordinates from 0 to 999. The top-left corner is (0,0). x increases to the right, and y increases downward. All coordinates must be integers.

Each instance must include:
- bbox_2d: [x1, y1, x2, y2], where all values are integers and x1 <= x2, y1 <= y2.
- label: one of the detection classes listed above.

Optional when possible:
- vertices: for angular shapes such as triangles, quadrilaterals, and polygons, list the vertices clockwise as [[x,y], ...], using integer coordinates from 0 to 999.

Deduplication and ordering:
Remove highly overlapping duplicate boxes. Return instances ordered from top to bottom, and from left to right within the same row, so that they can be aligned with later keypoint numbering.
"""