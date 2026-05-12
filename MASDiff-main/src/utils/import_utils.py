from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModuleSpec:
    """
    一个“可插拔模块”的配置描述。

    - class_path: Python 导入路径，例如 "my_pkg.my_mod:MyClass" 或 "my_pkg.my_mod.MyClass"
    - kwargs: 初始化参数
    """

    class_path: str
    kwargs: dict[str, Any]


def import_symbol(class_path: str) -> Any:
    """
    通过导入路径加载符号。

    支持两种写法：
    - "pkg.mod:Cls"
    - "pkg.mod.Cls"
    """
    if ":" in class_path:
        mod_name, sym_name = class_path.split(":", 1)
    else:
        mod_name, sym_name = class_path.rsplit(".", 1)
    module = importlib.import_module(mod_name)
    return getattr(module, sym_name)


def instantiate(spec: ModuleSpec) -> Any:
    """按 ModuleSpec 实例化对象。"""
    cls = import_symbol(spec.class_path)
    return cls(**(spec.kwargs or {}))

