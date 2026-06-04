from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping
import copy
import gzip
import json
import os
import tempfile


def _get_ssrm_dir() -> Path:
    current_dir = Path(__file__).resolve().parent

    if current_dir.name == "source":
        return current_dir.parent

    return current_dir


DEFAULT_SSRM_STATE_PATH = _get_ssrm_dir() / "ssrm_state.dat"


DEFAULT_SSRM_SCHEMA = {
    "Semantic layer": {
        "topic": "",
        "description": "",
        "scene_plan": "",
        "scene_vision_storyboard": "",
        "scene_implementation": "",
        "scene_technical_implementation": "",
        "scene_animation": "",
        "scene_technical_implementation_extractor": "",
        "rag_information": "",
        "scene_narration": "",
        "scene_code": "",
        "fix_error_code": "",
        "error_message": "",
        "geometric_parameter_control_module_information": "",
        "geometric_constraint_correction_module_information": "",
    },
    "Geometric structure layer": {
        "geometric_structure_extraction": "",
        "geometric_structure_extraction_corrected": "",
    },
    "Visual representation layer": {
        "base64_list": [],
    },
}


MissingPolicy = Literal["skip", "none", "error"]


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )

    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_name, path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except OSError:
            pass


def _empty_like(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _empty_like(v) for k, v in value.items()}
    if isinstance(value, list):
        return []
    if isinstance(value, str):
        return ""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return 0
    if isinstance(value, float):
        return 0.0
    if value is None:
        return None
    return None


def _dump_state_to_compressed_bytes(
    data: dict[str, Any],
    *,
    ensure_ascii: bool = False,
    indent: int = 2,
) -> bytes:
    text = json.dumps(
        data,
        ensure_ascii=ensure_ascii,
        indent=indent,
    )
    return gzip.compress(text.encode("utf-8"))


def _load_state_from_file(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()

    try:
        text = gzip.decompress(raw).decode("utf-8")
        return json.loads(text)
    except Exception:
        text = raw.decode("utf-8")
        return json.loads(text)


@dataclass
class SSRM:
    path: Path = field(default_factory=lambda: DEFAULT_SSRM_STATE_PATH)
    data: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_SSRM_SCHEMA))
    auto_save: bool = True
    ensure_ascii: bool = False
    indent: int = 2

    SEMANTIC_LAYER = "Semantic layer"
    GEOMETRIC_LAYER = "Geometric structure layer"
    VISUAL_LAYER = "Visual representation layer"

    @classmethod
    def load(
        cls,
        path: str | Path | None = DEFAULT_SSRM_STATE_PATH,
        *,
        state: Mapping[str, Any] | None = None,
        auto_save: bool = True,
    ) -> "SSRM":
        if state is not None:
            return cls(
                path=Path(path) if path is not None else DEFAULT_SSRM_STATE_PATH,
                data=copy.deepcopy(dict(state)),
                auto_save=auto_save,
            )

        p = Path(path) if path is not None else DEFAULT_SSRM_STATE_PATH

        if p.exists():
            data = _load_state_from_file(p)
        else:
            data = copy.deepcopy(DEFAULT_SSRM_SCHEMA)

        return cls(
            path=p,
            data=data,
            auto_save=auto_save,
        )

    def save(self, path: str | Path | None = None) -> Path:
        out = Path(path) if path is not None else self.path

        compressed = _dump_state_to_compressed_bytes(
            self.data,
            ensure_ascii=self.ensure_ascii,
            indent=self.indent,
        )

        _atomic_write_bytes(out, compressed)
        return out

    def export_state(self, path: str | Path | None = None) -> Path:
        return self.save(path)

    def import_state(self, path: str | Path) -> "SSRM":
        p = Path(path)
        self.data = _load_state_from_file(p)
        self.path = p
        self._autosave()
        return self

    def _autosave(self) -> None:
        if self.auto_save:
            self.save()

    def initialize(self, topic: str = "", description: str = "") -> "SSRM":
        self.reset()
        self.put(self.SEMANTIC_LAYER, "topic", topic)
        self.put(self.SEMANTIC_LAYER, "description", description)
        return self

    def reset(self) -> "SSRM":
        self.data = copy.deepcopy(DEFAULT_SSRM_SCHEMA)
        self._autosave()
        return self

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    def load_state_dict(self, state: Mapping[str, Any]) -> "SSRM":
        self.data = copy.deepcopy(dict(state))
        self._autosave()
        return self

    def snapshot(self) -> dict[str, Any]:
        return self.state_dict()

    def clear(self) -> "SSRM":
        self.data = _empty_like(self.data)
        self._autosave()
        return self

    def clear_layer(self, layer: str) -> "SSRM":
        layer_dict = self._ensure_layer(layer)
        self.data[layer] = _empty_like(layer_dict)
        self._autosave()
        return self

    def clear_val(self, key: str) -> "SSRM":
        changed = False

        for layer_dict in self.data.values():
            if isinstance(layer_dict, dict) and key in layer_dict:
                layer_dict[key] = _empty_like(layer_dict[key])
                changed = True

        if changed:
            self._autosave()

        return self

    def _ensure_layer(self, layer: str) -> dict[str, Any]:
        if layer not in self.data or not isinstance(self.data[layer], dict):
            self.data[layer] = {}
        return self.data[layer]

    def get(self, layer: str, key: str, default: Any = None) -> Any:
        return self.data.get(layer, {}).get(key, default)

    def get_layer(self, layer: str, default: Any = None) -> dict[str, Any] | Any:
        return self.data.get(layer, default)

    def get_val(self, key: str, default: Any = None) -> Any:
        for layer_data in self.data.values():
            if isinstance(layer_data, dict) and key in layer_data:
                return layer_data[key]
        return default

    def get_semantic(self, key: str, default: Any = None) -> Any:
        return self.get(self.SEMANTIC_LAYER, key, default)

    def get_geometric(self, key: str, default: Any = None) -> Any:
        return self.get(self.GEOMETRIC_LAYER, key, default)

    def get_visual(self, key: str, default: Any = None) -> Any:
        return self.get(self.VISUAL_LAYER, key, default)

    def put(self, layer: str, key: str, value: Any) -> "SSRM":
        self._ensure_layer(layer)[key] = value
        self._autosave()
        return self

    def put_semantic(self, key: str, value: Any) -> "SSRM":
        return self.put(self.SEMANTIC_LAYER, key, value)

    def put_geometric(self, key: str, value: Any) -> "SSRM":
        return self.put(self.GEOMETRIC_LAYER, key, value)

    def put_visual(self, key: str, value: Any) -> "SSRM":
        return self.put(self.VISUAL_LAYER, key, value)

    def add(
        self,
        layer: str,
        key: str,
        value: Any,
        *,
        coerce_empty_str_to_list: bool = True,
    ) -> "SSRM":
        layer_dict = self._ensure_layer(layer)

        if key not in layer_dict:
            layer_dict[key] = value
            self._autosave()
            return self

        cur = layer_dict[key]

        if coerce_empty_str_to_list and cur == "" and not isinstance(value, str):
            layer_dict[key] = [value]
            self._autosave()
            return self

        if isinstance(cur, list):
            cur.append(value)
            self._autosave()
            return self

        if isinstance(cur, dict) and isinstance(value, dict):
            cur.update(value)
            self._autosave()
            return self

        if isinstance(cur, str) and isinstance(value, str):
            layer_dict[key] = cur + value
            self._autosave()
            return self

        if isinstance(cur, (int, float)) and isinstance(value, (int, float)):
            layer_dict[key] = cur + value
            self._autosave()
            return self

        raise TypeError(
            f"Cannot add {type(value).__name__} to existing "
            f"{type(cur).__name__} at [{layer}][{key}]"
        )

    def add_semantic(self, key: str, value: Any) -> "SSRM":
        return self.add(self.SEMANTIC_LAYER, key, value)

    def add_geometric(self, key: str, value: Any) -> "SSRM":
        return self.add(self.GEOMETRIC_LAYER, key, value)

    def add_visual(self, key: str, value: Any) -> "SSRM":
        return self.add(self.VISUAL_LAYER, key, value)

    def put_vars(
        self,
        layer: str,
        var_names: Iterable[str],
        namespace: Mapping[str, Any] | None = None,
        *,
        missing: MissingPolicy = "skip",
    ) -> "SSRM":
        ns = globals() if namespace is None else namespace
        layer_dict = self._ensure_layer(layer)

        changed = False

        for name in var_names:
            if name in ns:
                layer_dict[name] = ns[name]
                changed = True
            else:
                if missing == "skip":
                    continue
                if missing == "none":
                    layer_dict[name] = None
                    changed = True
                else:
                    raise NameError(f"Variable '{name}' not found in namespace")

        if changed:
            self._autosave()

        return self

    def add_vars(
        self,
        layer: str,
        var_names: Iterable[str],
        namespace: Mapping[str, Any] | None = None,
        *,
        missing: MissingPolicy = "skip",
        coerce_empty_str_to_list: bool = True,
    ) -> "SSRM":
        ns = globals() if namespace is None else namespace

        for name in var_names:
            if name in ns:
                self.add(
                    layer,
                    name,
                    ns[name],
                    coerce_empty_str_to_list=coerce_empty_str_to_list,
                )
            else:
                if missing == "skip":
                    continue
                if missing == "none":
                    self.add(
                        layer,
                        name,
                        None,
                        coerce_empty_str_to_list=coerce_empty_str_to_list,
                    )
                else:
                    raise NameError(f"Variable '{name}' not found in namespace")

        return self

    def __call__(self, topic: str = "", description: str = "") -> dict[str, Any]:
        self.initialize(topic, description)
        return self.state_dict()


def load_model(
    path: str | Path | None = DEFAULT_SSRM_STATE_PATH,
    *,
    state: Mapping[str, Any] | None = None,
    auto_save: bool = True,
) -> SSRM:
    return SSRM.load(
        path=path,
        state=state,
        auto_save=auto_save,
    )


ssrm: SSRM = load_model()


if __name__ == "__main__":
    print("Testing SSRM...")

    test_path = Path("test_ssrm_state.dat")

    model = SSRM.load(test_path, auto_save=True)

    model.initialize(
        topic="Pythagorean Theorem",
        description="Generate a two-dimensional digital scene.",
    )

    model.put_semantic("scene_plan", "Scene planning content")
    model.put_semantic("scene_narration", "Scene narration content")
    model.put_semantic("scene_code", "print('hello manim')")

    model.put_geometric(
        "geometric_structure_extraction",
        {
            "objects": ["triangle", "square"],
            "relations": ["right_angle", "side_length"],
        },
    )

    model.put_visual(
        "base64_list",
        ["base64_frame_001", "base64_frame_002"],
    )

    print("State file generated:", test_path.exists())
    print("State file path:", test_path.resolve())

    loaded_model = SSRM.load(test_path, auto_save=False)

    print("Loaded topic:", loaded_model.get_semantic("topic"))
    print("Loaded description:", loaded_model.get_semantic("description"))
    print("Loaded scene_plan:", loaded_model.get_semantic("scene_plan"))
    print("Loaded scene_code:", loaded_model.get_semantic("scene_code"))
    print(
        "Loaded geometric structure:",
        loaded_model.get_geometric("geometric_structure_extraction"),
    )
    print("Loaded frame sequence:", loaded_model.get_visual("base64_list"))

    print("SSRM test finished.")