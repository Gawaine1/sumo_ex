from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    """为 json.dumps 提供兜底转换，兼容 Tensor/ndarray/Path 等对象。"""
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _to_text(value: Any) -> str:
    """将任意对象转换为可写入 CSV 的文本。"""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, default=_json_default)


def pickle_to_csv(pickle_path: str | Path, csv_path: str | Path | None = None) -> Path:
    """
    将 pickle 文件转换为 CSV 文件。

    支持的常见输入结构：
    - list[dict]：按字典键展开为表格列
    - list/tuple/set（非 dict 元素）：写为单列 value
    - dict：写为两列 key/value
    - 其他对象：写为单列 value 的一行

    参数：
    - pickle_path: 输入 pickle 文件路径
    - csv_path: 输出 CSV 文件路径；为空时默认与 pickle 同目录同名（后缀 .csv）

    返回：
    - 实际写入的 CSV 路径（Path）
    """
    pkl = Path(pickle_path)
    if not pkl.exists():
        raise FileNotFoundError(f"pickle 文件不存在: {pkl}")

    out = Path(csv_path) if csv_path is not None else pkl.with_suffix(".csv")
    if out.parent != Path("."):
        out.parent.mkdir(parents=True, exist_ok=True)

    with pkl.open("rb") as f:
        data = pickle.load(f)

    with out.open("w", newline="", encoding="utf-8") as f:
        # 情况 1：list[dict]
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            fieldnames: list[str] = []
            for row in data:
                for k in row.keys():
                    if k not in fieldnames:
                        fieldnames.append(str(k))

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow({k: _to_text(row.get(k)) for k in fieldnames})

        # 情况 2：普通序列（非 dict 元素）
        elif isinstance(data, (list, tuple, set)):
            writer = csv.writer(f)
            writer.writerow(["value"])
            for item in data:
                writer.writerow([_to_text(item)])

        # 情况 3：dict
        elif isinstance(data, dict):
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            for k, v in data.items():
                writer.writerow([_to_text(k), _to_text(v)])

        # 情况 4：其他对象
        else:
            writer = csv.writer(f)
            writer.writerow(["value"])
            writer.writerow([_to_text(data)])

    return out

if __name__ == '__main__':
    pickle_to_csv("../../outputs/Q_nov.pickle")