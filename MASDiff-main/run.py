from __future__ import annotations

import argparse

from src.config.loader import load_config
from src.pipeline.runner import run_masdiff


def main() -> None:
    parser = argparse.ArgumentParser(description="MASDiff 框架入口（仅主流程 + 抽象接口）")
    parser.add_argument("--config", required=True, help="配置文件路径（yaml）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_masdiff(cfg)


if __name__ == "__main__":
    main()

