"""ssr_store.py

Auto-saving store for `ssr.json` with map-like APIs:
  - put(layer, key, value): overwrite and save immediately
  - add(layer, key, value): append/merge/accumulate and save immediately
  - clear(): clear VALUES only, keep structure (layers/keys), then save
  - clear_layer(layer): clear VALUES only in one layer, keep keys, then save
  - put_vars/add_vars: write by variable names from a given namespace (globals/locals)

Exports:
  - SSRJsonStore, load_store, store (default bound to "ssr.json")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Iterable, Literal
import json
import os
import tempfile
from config import cfg

MissingPolicy = Literal["skip", "none", "error"]


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Atomic write to reduce risk of corrupted files on crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
        os.replace(tmp_name, path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except OSError:
            pass


def _empty_like(value: Any) -> Any:
    """Clear VALUES while keeping STRUCTURE (keys/layers)."""
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


@dataclass
class SSRJsonStore:
    """Map-like writer for layered SSR JSON dictionaries (auto-save)."""
    path: Path = field(default_factory=lambda: Path("ssr.json"))
    data: dict[str, Any] = field(default_factory=dict)
    auto_save: bool = True
    ensure_ascii: bool = False
    indent: int = 2

    @classmethod
    def load(cls, path: str | Path = "ssr.json", *, auto_save: bool = True) -> "SSRJsonStore":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        return cls(path=p, data=data, auto_save=auto_save)

    def save(self, path: str | Path | None = None) -> Path:
        out = Path(path) if path is not None else self.path
        text = json.dumps(self.data, ensure_ascii=self.ensure_ascii, indent=self.indent)
        print(f"正在写入文件: {out}")  # 添加这一行
        _atomic_write_text(out, text, encoding="utf-8")
        return out

    def _autosave(self) -> None:
        if self.auto_save:
            self.save()

    def _ensure_layer(self, layer: str) -> dict[str, Any]:
        if layer not in self.data or not isinstance(self.data[layer], dict):
            self.data[layer] = {}
        return self.data[layer]

    # ---------- required clearing APIs (keep structure) ----------
    def clear(self) -> "SSRJsonStore":
        """Clear VALUES only for the whole JSON; keep all layers/keys."""
        self.data = _empty_like(self.data)
        self._autosave()
        return self

    def clear_layer(self, layer: str) -> "SSRJsonStore":
        """Clear VALUES only for one layer; keep that layer's keys."""
        layer_dict = self._ensure_layer(layer)
        self.data[layer] = _empty_like(layer_dict)
        self._autosave()
        return self

    # ---------- read APIs ----------
    def get(self, layer: str, key: str, default: Any = None) -> Any:
        """
        获取指定层中的指定键值。
        如果 layer 不存在或 key 不存在，则返回 default 值。
        """
        return self.data.get(layer, {}).get(key, default)

    def get_layer(self, layer: str, default: Any = None) -> dict[str, Any] | Any:
        """
        获取整个层的数据。
        """
        return self.data.get(layer, default)

    def get_val(self, key: str, default: Any = None) -> Any:
        """
        全量搜索：直接输入属性名获取值。
        遍历所有层级，找到匹配的 key 即返回。
        """
        for layer_data in self.data.values():
            if isinstance(layer_data, dict) and key in layer_data:
                return layer_data[key]
        return default

    # ---------- write APIs (auto-save) ----------
    def put(self, layer: str, key: str, value: Any) -> "SSRJsonStore":
        self._ensure_layer(layer)[key] = value
        self._autosave()
        return self

    def add(self, layer: str, key: str, value: Any, *, coerce_empty_str_to_list: bool = True) -> "SSRJsonStore":
        layer_dict = self._ensure_layer(layer)

        if key not in layer_dict:
            layer_dict[key] = value
            self._autosave()
            return self

        cur = layer_dict[key]

        # Compatibility: if existing value is "" but you want list-style accumulation
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
            f"Cannot add {type(value).__name__} to existing {type(cur).__name__} at [{layer}][{key}]"
        )

    # ---------- bulk write by variable names ----------
    def put_vars(
        self,
        layer: str,
        var_names: Iterable[str],
        namespace: Mapping[str, Any] | None = None,
        *,
        missing: MissingPolicy = "skip",
    ) -> "SSRJsonStore":
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
    ) -> "SSRJsonStore":
        ns = globals() if namespace is None else namespace
        for name in var_names:
            if name in ns:
                self.add(layer, name, ns[name], coerce_empty_str_to_list=coerce_empty_str_to_list)
            else:
                if missing == "skip":
                    continue
                if missing == "none":
                    self.add(layer, name, None, coerce_empty_str_to_list=coerce_empty_str_to_list)
                else:
                    raise NameError(f"Variable '{name}' not found in namespace")
        return self

    def clear_val(self, key: str) -> "SSRJsonStore":
        """
        在所有层中搜索该属性，并将其值重置为初始状态（如 ""、[]、0 等）。
        保留键名，仅清空内容。
        """
        changed = False
        for layer_dict in self.data.values():
            if isinstance(layer_dict, dict) and key in layer_dict:
                # 使用你代码中现有的 _empty_like 函数
                layer_dict[key] = _empty_like(layer_dict[key])
                changed = True

        if changed:
            self._autosave()
        return self


def load_store(path: str | Path = "ssr.json", *, auto_save: bool = True) -> SSRJsonStore:
    return SSRJsonStore.load(path, auto_save=auto_save)


# Default auto-saving store bound to "ssr.json"
ssr_store: SSRJsonStore = load_store(cfg.SSR_STORE_PATH, auto_save=True)


if __name__ == "__main__":
    #ssr_store.put("Semantic layer", "scene_narration", "同学们，……")
    image_list = ["data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD..."]
    image_json_list = [{"filename": "image1.jpg", "path": "/path/to/image1.jpg"}]
    # 将 image_list 和 image_json_list 存入 ssr
    ssr_store.put("Visual representation layer", "base64_list", image_list)
    ssr_store.put("Visual representation layer", "image_json_list", image_json_list)
    base64_list = ssr_store.get_val("base64_list")
    ssr_store.put("Semantic layer","error_message","xx")
    error_message=ssr_store.get("Semantic layer","error_message")
    print(error_message)
    #ssr_store.clear()
