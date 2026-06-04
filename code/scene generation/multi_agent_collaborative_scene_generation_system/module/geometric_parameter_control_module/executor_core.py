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

    raw_has_dt = dt is not None

    if dt is not None:
        dt = float(dt)

        if t0 is None and t1 is not None:
            t1 = float(t1)
            t0 = t1 - dt
        elif t0 is None and t1 is None:
            t0 = 0.0
            t1 = t0 + dt
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


def _format_floats(obj, nd=3, eps=1e-10, skip_keys=None):
    if skip_keys is None:
        skip_keys = set()

    if isinstance(obj, dict):
        output = {}

        for key, value in obj.items():
            if key in skip_keys:
                output[key] = value
            else:
                output[key] = _format_floats(value, nd, eps, skip_keys)

        return output

    if isinstance(obj, float):
        if not math.isfinite(obj):
            return str(obj)

        value = 0.0 if abs(obj) < eps else round(obj, nd)
        return f"{value:.{nd}f}"

    if isinstance(obj, (list, tuple)):
        return [
            _format_floats(value, nd, eps, skip_keys)
            for value in obj
        ]

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
    timeline_origin_seconds=0.0,
):
    if steps is None or len(steps) == 0:
        return []

    objects = {}
    plan = []

    for raw in steps:
        t0, t1, dt, rate_func, group, raw_has_dt = _normalize_timing_fields(
            raw,
            timeline_origin=timeline_origin_seconds,
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
            raise ValueError(f"Unknown function: {fn}")

        fn_callable = registry[fn]

        src_obj_for_speed = objects.get(src_id) if src_id in objects else None

        if not raw_has_dt and speed_mag is not None:
            kind, mag = speed_mag.magnitude(
                fn,
                params,
                src_obj_for_speed,
            )

            if kind and mag is not None and "speed" in raw:
                speed = raw["speed"]
                linear_speed = None
                angular_speed = None
                scale_speed = None

                if isinstance(speed, (int, float)):
                    linear_speed = float(speed)
                elif isinstance(speed, dict):
                    if "linear" in speed:
                        linear_speed = float(speed["linear"])

                    if "angular_deg_per_sec" in speed:
                        angular_speed = float(speed["angular_deg_per_sec"])

                    if "scale_rate_per_sec" in speed:
                        scale_speed = float(speed["scale_rate_per_sec"])

                if kind == "linear" and linear_speed and linear_speed > 0:
                    dt = mag / linear_speed
                elif kind == "angular" and angular_speed and angular_speed > 0:
                    dt = mag / angular_speed
                elif kind == "logscale" and scale_speed and scale_speed > 0:
                    dt = mag / scale_speed

                t1 = t0 + dt

        if dt < 0:
            raise ValueError(
                f"Invalid timing: dt < 0 "
                f"(fn={fn}, src_id={src_id}, out_id={out_id}, "
                f"t0={t0}, t1={t1}, dt={dt})"
            )

        dt = _quantize_dt(dt, fps_hint)

        if dt == 0.0:
            if zero_dt_policy == "epsilon":
                dt = float(zero_dt_epsilon)
            elif zero_dt_policy == "error":
                raise ValueError(
                    f"dt == 0 is not allowed "
                    f"(fn={fn}, src_id={src_id}, out_id={out_id}, t0={t0})"
                )

        if "construct" in fn:
            result = fn_callable(params)
            out_id = out_id or f"{fn}_0"
            objects[out_id] = result

            action = "Create"
            remove_src = None
            src_id_long = None
            dst_id_long = out_id
            in_place_flag = False
        else:
            if not src_id or src_id not in objects:
                raise ValueError(f"{fn}: source object not found: src_id={src_id}")

            call_payload = {
                "from_construct": objects[src_id],
                **params,
            }

            result = fn_callable(call_payload)

            if in_place:
                action = "Transform"
                remove_src = True
                out_id = src_id
                objects[out_id] = result
                src_id_long = src_id
                dst_id_long = out_id
                in_place_flag = True
            else:
                if not out_id:
                    raise ValueError(f"{fn}: out_id is required for non-in-place operations")

                objects[out_id] = result

                if keep:
                    action = "Create"
                    remove_src = False
                else:
                    action = "Transform"
                    remove_src = True

                src_id_long = src_id
                dst_id_long = out_id
                in_place_flag = False

        packed_geom = packer.pack(result)

        plan.append(
            {
                "function_name_to_execute_string": fn,
                "function_call_parameters_named_arguments_object": params,
                "source_object_identifier_string": src_id_long,
                "destination_object_identifier_string": dst_id_long,
                "should_modify_source_object_in_place_boolean": in_place_flag,
                "style_suggested_render_color_name_string": color,
                "style_should_show_vertex_labels_boolean": bool(labels),
                "style_layering_z_index_integer": z,
                "custom_tag_value_string_or_null": (
                    tag if tag is None or isinstance(tag, str) else str(tag)
                ),
                "render_action_kind_for_manim_string": action,
                "should_remove_source_after_transform_boolean": remove_src,
                **packed_geom,
                "timeline_t0_seconds_float": float(t0),
                "timeline_t1_seconds_float": float(t1),
                "timeline_run_time_seconds_float": float(dt),
                "timeline_rate_function_name_string_or_null": rate_func,
                "timeline_parallel_group_key_string_or_null": group,
            }
        )

    if force_str:
        plan = [
            _format_floats(
                item,
                nd=nd,
                skip_keys={
                    "timeline_t0_seconds_float",
                    "timeline_t1_seconds_float",
                    "timeline_run_time_seconds_float",
                },
            )
            for item in plan
        ]

    return plan