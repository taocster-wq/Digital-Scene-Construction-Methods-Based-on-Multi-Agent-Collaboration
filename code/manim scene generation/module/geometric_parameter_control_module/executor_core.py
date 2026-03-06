# executor_core.py
import math

def _quantize_dt(dt, fps_hint):
    if not fps_hint or fps_hint <= 0:
        return float(dt)
    base = 1.0 / float(fps_hint)
    q = round(dt / base) * base
    if 0.0 < q < base:
        q = base
    if q < 0:
        q = 0.0
    return float(q)

def _normalize_timing_fields(step, *, timeline_origin=0.0):
    t0 = step.get("t0")
    t1 = step.get("t1")
    dt = step.get("dt")

    rate_func = step.get("rate_func") or step.get("easing")
    group = step.get("group")

    raw_has_dt = (dt is not None)

    if dt is not None:
        dt = float(dt)
        if t0 is None and t1 is not None:
            t1 = float(t1); t0 = t1 - dt
        elif t0 is None and t1 is None:
            t0 = 0.0; t1 = t0 + dt
        else:
            t0 = float(t0) if t0 is not None else 0.0
            t1 = t0 + dt
    else:
        if t0 is None and t1 is None:
            t0, t1 = 0.0, 0.0
        else:
            t0 = float(t0) if t0 is not None else 0.0
            t1 = float(t1) if t1 is not None else t0
        dt = max(0.0, t1 - t0)

    t0 = float(t0) + float(timeline_origin)
    t1 = float(t1) + float(timeline_origin)
    return t0, t1, dt, rate_func, group, raw_has_dt

# 允许传入一个 set，里面是不要格式化的键
def _format_floats(obj, nd=3, eps=1e-10, skip_keys=None):
    if skip_keys is None:
        skip_keys = set()
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in skip_keys:
                out[k] = v  # 不格式化
            else:
                out[k] = _format_floats(v, nd, eps, skip_keys)
        return out
    if isinstance(obj, float):
        import math
        if not math.isfinite(obj):
            return str(obj)
        x = 0.0 if abs(obj) < eps else round(obj, nd)
        return f"{x:.{nd}f}"
    if isinstance(obj, (list, tuple)):
        return [_format_floats(v, nd, eps, skip_keys) for v in obj]
    return obj

def apply_actions_emit_scene_plan_generic(
    steps,
    *,
    registry,
    packer,
    speed_mag,
    fps_hint=30,
    zero_dt_policy="epsilon",
    zero_dt_epsilon=1e-6,
    nd=3,
    force_str=True,
    timeline_origin_seconds=0.0
):
    if steps is None or len(steps) == 0:
        return []
    objects = {}  # id -> last geometry object (any shape dict)
    plan = []

    for raw in steps:
        t0, t1, dt, rate_func, group, raw_has_dt = _normalize_timing_fields(
            raw, timeline_origin=timeline_origin_seconds
        )

        fn = raw["fn"]
        params = raw.get("params", {})
        src_id = raw.get("src_id")
        out_id = raw.get("out_id")

        in_place = bool(raw.get("in_place", False))
        keep = bool(raw.get("keep", True))

        color = raw.get("color")
        labels = raw.get("labels", True)
        z = int(raw.get("z", 0))
        tag = raw.get("tag")

        if fn not in registry:
            raise ValueError(f"未知函数: {fn}")
        fn_callable = registry[fn]

        # 若没显式 dt，尝试 speed→dt
        src_obj_for_speed = objects.get(src_id) if src_id in objects else None
        if not raw_has_dt and speed_mag is not None:
            kind, mag = speed_mag.magnitude(fn, params, src_obj_for_speed)
            if kind and mag is not None and "speed" in raw:
                sp = raw["speed"]
                lin_v = ang_w = scl_s = None
                if isinstance(sp, (int, float)):
                    lin_v = float(sp)
                elif isinstance(sp, dict):
                    if "linear" in sp: lin_v = float(sp["linear"])
                    if "angular_deg_per_sec" in sp: ang_w = float(sp["angular_deg_per_sec"])
                    if "scale_rate_per_sec" in sp: scl_s = float(sp["scale_rate_per_sec"])
                if   kind == "linear"  and lin_v and lin_v>0: dt = mag / lin_v
                elif kind == "angular" and ang_w and ang_w>0: dt = mag / ang_w
                elif kind == "logscale" and scl_s and scl_s>0: dt = mag / scl_s
                t1 = t0 + dt

        # dt 校验 & 量化
        if dt < 0:
            raise ValueError(f"时间非法：dt<0（fn={fn}, src_id={src_id}, out_id={out_id}, t0={t0}, t1={t1}, dt={dt}）")
        dt = _quantize_dt(dt, fps_hint)
        if dt == 0.0:
            if zero_dt_policy == "epsilon":
                dt = float(zero_dt_epsilon)
            elif zero_dt_policy == "error":
                raise ValueError(f"dt==0 不被允许（fn={fn}, src_id={src_id}, out_id={out_id}, t0={t0}）")

        # 执行
        if "construct" in fn:
            res = fn_callable(params)
            out_id = out_id or f"{fn}_0"
            objects[out_id] = res
            action = "Create"
            remove_src = None
            src_id_long = None
            dst_id_long = out_id
            in_place_flag = False
        else:
            if not src_id or src_id not in objects:
                raise ValueError(f"{fn}: 找不到源对象 src_id={src_id}")
            call_payload = {"from_construct": objects[src_id], **params}
            res = fn_callable(call_payload)

            if in_place:
                action = "Transform"
                remove_src = True
                out_id = src_id
                objects[out_id] = res
                src_id_long = src_id
                dst_id_long = out_id
                in_place_flag = True
            else:
                if not out_id:
                    raise ValueError(f"{fn}: 非原地需要 out_id")
                objects[out_id] = res
                if keep:
                    action = "Create"
                    remove_src = False
                else:
                    action = "Transform"
                    remove_src = True
                src_id_long = src_id
                dst_id_long = out_id
                in_place_flag = False

        # 打包几何（多形状支持）
        packed_geom = packer.pack(res)

        # 写入 plan
        plan.append({
            "function_name_to_execute_string": fn,
            "function_call_parameters_named_arguments_object": params,
            "source_object_identifier_string": src_id_long,
            "destination_object_identifier_string": dst_id_long,
            "should_modify_source_object_in_place_boolean": in_place_flag,

            "style_suggested_render_color_name_string": color,
            "style_should_show_vertex_labels_boolean": bool(labels),
            "style_layering_z_index_integer": z,
            "custom_tag_value_string_or_null": tag if (tag is None or isinstance(tag, str)) else str(tag),

            "render_action_kind_for_manim_string": action,
            "should_remove_source_after_transform_boolean": remove_src,

            **packed_geom,

            "timeline_t0_seconds_float": float(t0),
            "timeline_t1_seconds_float": float(t1),
            "timeline_run_time_seconds_float": float(dt),
            "timeline_rate_function_name_string_or_null": rate_func,
            "timeline_parallel_group_key_string_or_null": group,
        })

    if force_str:
        plan = [_format_floats(x, nd=nd,skip_keys={
            "timeline_t0_seconds_float",
            "timeline_t1_seconds_float",
            "timeline_run_time_seconds_float",
        }) for x in plan]

    return plan