from executor import apply_actions_emit_scene_plan
from geometry_analysis.geometry_core import load_geometry_data, GeometryAnalyzer

# steps = [
#     # 支柱
#     {"fn":"construct_line",
#      "params":{"mode":"point_dir_len", "P":[0.0,0.0], "angle_deg":270.0, "length":8.0},
#      "out_id":"pillar", "color":"#888888", "labels":False, "z":-1,
#      "t0":0.0, "dt":0.1},
#
#     # 0s 瞬时构造三角形（基准叶片）
#     {"fn":"construct_triangle",
#      "params":{"mode":"SSS","sides":{"AB":5,"BC":3,"CA":4},"orientation":"CCW"},
#      "out_id":"leaf_0deg", "color":"WHITE", "labels":True, "z":0,
#      "t0":0.0, "dt":0.0},
#
#     # ✅ 预倾斜：把基准叶片先“向左倾斜 45°”
#     # 这里我按“逆时针时针 45°(CW)”理解；若你想反向，direction 改 "CCW" 即可
#     {"fn":"rotate_triangle", "src_id":"leaf_0deg", "out_id":"leaf_0deg",
#      "in_place":True,                 # 原地修改同一个对象
#      "params":{"rotate":{"mode":"about_vertex","vertex":"A","deg":45,"direction":"CCW"}},
#      "color":"WHITE", "labels":True, "z":0,
#      "t0":0.2, "dt":0.3, "rate_func":"smooth"},
#
#     # 后续三个叶片：仍然基于（已倾斜后的）leaf_0deg 复制并旋转
#     {"fn":"rotate_triangle", "src_id":"leaf_0deg", "out_id":"leaf_90deg",
#      "in_place":False, "keep":True,
#      "params":{"rotate":{"mode":"about_vertex","vertex":"A","deg":90,"direction":"CCW"}},
#      "color":"RED", "labels":True, "z":1,
#      "t0":0.5, "dt":0.8, "rate_func":"linear"},
#
#     {"fn":"rotate_triangle", "src_id":"leaf_0deg", "out_id":"leaf_180deg",
#      "in_place":False, "keep":True,
#      "params":{"rotate":{"mode":"about_vertex","vertex":"A","deg":180,"direction":"CCW"}},
#      "color":"GREEN", "labels":True, "z":2,
#      "t0":1.2, "dt":0.8, "rate_func":"smooth"},
#
#     {"fn":"rotate_triangle", "src_id":"leaf_0deg", "out_id":"leaf_270deg",
#      "in_place":False, "keep":True,
#      "params":{"rotate":{"mode":"about_vertex","vertex":"A","deg":270,"direction":"CCW"}},
#      "color":"BLUE", "labels":True, "z":3,
#      "t0":2.0, "dt":0.8, "rate_func":"linear"},
# ]

# steps = [
#     # ① 第一个方形（边 AB：(-1,0)→(1,0)，默认往上）
#     {"fn":"construct_square",
#      "params":{"mode":"two_points_as_side","A":[-1.0,0.0],"B":[1.0,0.0]},
#      "out_id":"sq_ab_top", "color":"WHITE", "labels":True, "z":0,
#      "t0":0.0, "dt":0.4},
#
#     # ② 第二个方形：与 AB 共边，但“背离”第一个方形（参考点取第一个方形的中心(0,1)）
#     {"fn":"construct_square",
#      "params":{
#          "mode":"two_points_as_side",
#          "A":[-1.0,0.0], "B":[1.0,0.0],
#          "choose_by_point":{"point":[0.0,1.0], "relation":"away"}   # 或 "towards"
#      },
#      "out_id":"sq_ab_copy", "color":"YELLOW", "labels":True, "z":1,
#      "t0":0.5, "dt":0.4},
#
#     # ③ 绕 B 逆时针 90° 旋转
#     {"fn":"rotate_square", "src_id":"sq_ab_top", "out_id":"sq_ab_rot",
#      "params":{"rotate":{"mode":"about_vertex","which":"B","deg":45,"direction":"CCW"}},
#      "color":"GREEN", "labels":True, "z":2,
#      "t0":1.0, "dt":0.6, "rate_func":"smooth"}
# ]
# steps = [
#     # 1) 构造圆：圆心(0,0)，半径 2
#     {"fn":"construct_circle",
#      "params":{"mode":"center_radius","center":[0.0,0.0],"radius":2.0},
#      "out_id":"circle_0", "color":"WHITE", "labels":True, "z":0,
#      "t0":0.0, "dt":0.4},
#
#     # 2) 平移：整体移动 (+2,+1)
#     {"fn":"move_circle", "src_id":"circle_0", "out_id":"circle_1",
#      "params":{"move":{"mode":"by_vector","dx":2.0,"dy":1.0}},
#      "color":"YELLOW", "labels":True, "z":1,
#      "t0":0.5, "dt":0.4, "rate_func":"linear"},
#
#     # 3) 修改半径：保持圆心不变，把半径改到 3
#     {"fn":"set_circle_radius", "src_id":"circle_1", "out_id":"circle_2",
#      "params":{"radius":3.0, "mode":"keep_center"},
#      "color":"GREEN", "labels":True, "z":2,
#      "t0":1.0, "dt":0.4, "rate_func":"smooth"},
#
#     # 4) 导出折线近似：用 45° 粒度近似圆
#     {"fn": "export_as_polyline", "src_id": "circle_2", "out_id": "circle_2_poly",
#      "params": {"step_deg": 45},  # 每 45 度一个点 → 八段
#      "color": "RED", "labels": False, "z": 3,
#      "t0": 1.5, "dt": 0.3},
# # 导出圆心点
# {"fn":"export_center_point", "src_id":"circle_2", "out_id":"circle_2_center",
#  "color":"YELLOW", "labels":True, "z":3,
#  "t0":2.0, "dt":0.3},
#
# # 导出半径线段（角度 0°）
# {"fn":"export_radius_segment", "src_id":"circle_2", "out_id":"circle_2_radius",
#  "params":{"angle_deg":0.0},
#  "color":"BLUE", "labels":True, "z":3,
#  "t0":2.5, "dt":0.3}
# ]

# steps=[
#   {
#     "fn": "construct_rhombus",
#     "out_id": "rh_0",
#     "params": {
#       "mode": "diag_center_len_angle",
#       "center": [0.0, 0.0],
#       "diag1_length": 6.0,
#       "diag1_angle_deg": 30.0,
#       "diag2_length": 4.0
#     },
#     "t0": 0.0, "dt": 0.4, "color": "WHITE", "labels": True, "z": 0, "rate_func": "linear"
#   },
# ]

# steps =[
#     {"fn":"construct_line",
#      "params":{"mode":"point_dir_len", "P":[0.0,0.0], "angle_deg":270.0, "length":8.0},
#      "out_id":"pillar", "color":"#888888", "labels":False, "z":-1,
#      "t0":0.0, "dt":0.1},
#         {"fn":"construct_triangle",
#          "params":{"mode":"SSS","sides":{"AB":5,"BC":3,"CA":4},"orientation":"CCW"},
#          "out_id":"leaf_0deg", "color":"WHITE", "labels":True, "z":0,
#          "t0":0.0, "dt":0.0},
# ]
# plan = apply_actions_emit_scene_plan(
#     steps,
#     fps_hint=30,
#     zero_dt_policy="epsilon",
#     zero_dt_epsilon=1e-6,
#     nd=3,
#     force_str=True
# )
#动作步骤计划表
# print(plan)
# # 提取几何数据
# geometry_data = load_geometry_data(plan)
#
# # 创建分析器
# analyzer = GeometryAnalyzer(geometry_data)
#
# # 查找交点
# analyzer.find_all_intersections()
#
# # 打印交点
# # analyzer.print_intersections()
# #几何图形表
# print(analyzer.get_geometry_analysis_results())
