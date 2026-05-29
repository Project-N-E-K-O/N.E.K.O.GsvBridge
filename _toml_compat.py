"""
轻量 TOML 读写兼容层，消除对 tomli / tomli_w 的外部依赖。

读取：
  - Python 3.11+ 使用内置 tomllib
  - Python 3.10  使用内置 configparser 风格的简易解析器
    （仅支持 api_neko 配置文件用到的子集：基本类型 + 一层 [table]）

写入：
  - 手写序列化器，支持 string / int / float / bool / list[str] + 一层 table
"""

import sys

# ─── TOML 读取 ───

if sys.version_info >= (3, 11):
    import tomllib

    def load_toml(path) -> dict:
        with open(path, "rb") as f:
            return tomllib.load(f)
else:
    # Python 3.10 兼容：简易 TOML 解析器
    # 仅支持 api_neko 用到的子集：
    #   - 顶层和一层 [table]
    #   - 值类型：字符串（双引号）、整数、浮点数、布尔值、字符串数组
    import re

    def _parse_value(raw: str):
        """解析 TOML 值"""
        raw = raw.strip()
        if not raw:
            return ""
        # 布尔
        if raw == "true":
            return True
        if raw == "false":
            return False
        # 字符串（双引号或单引号）
        if (raw.startswith('"') and raw.endswith('"')) or \
           (raw.startswith("'") and raw.endswith("'")):
            return raw[1:-1].replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
        # 数组（仅支持字符串数组和空数组）
        if raw.startswith("["):
            inner = raw[1:].rstrip("]").strip()
            if not inner:
                return []
            items = []
            for item in re.split(r',\s*', inner):
                item = item.strip()
                if (item.startswith('"') and item.endswith('"')) or \
                   (item.startswith("'") and item.endswith("'")):
                    items.append(item[1:-1])
                else:
                    items.append(item)
            return items
        # 整数或浮点数
        try:
            if "." in raw:
                return float(raw)
            return int(raw)
        except ValueError:
            return raw

    def load_toml(path) -> dict:
        """简易 TOML 解析器（仅支持 api_neko 配置子集）"""
        result = {}
        current_table = None

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # 跳过空行和注释
                if not line or line.startswith("#"):
                    continue
                # [table] 头
                m = re.match(r'^\[([a-zA-Z_][a-zA-Z0-9_]*)\]$', line)
                if m:
                    current_table = m.group(1)
                    if current_table not in result:
                        result[current_table] = {}
                    continue
                # key = value
                m = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$', line)
                if m:
                    key, raw_val = m.group(1), m.group(2)
                    # 处理行尾注释（不在字符串内的 #）
                    if not raw_val.startswith('"') and not raw_val.startswith("'") and not raw_val.startswith("["):
                        raw_val = raw_val.split("#")[0].strip()
                    val = _parse_value(raw_val)
                    if current_table is not None:
                        result[current_table][key] = val
                    else:
                        result[key] = val

        return result


# ─── TOML 写入 ───

def _format_value(val) -> str:
    """将 Python 值序列化为 TOML 格式字符串"""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        return str(val)
    if isinstance(val, str):
        # 转义双引号和反斜杠
        escaped = val.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    if isinstance(val, list):
        items = ", ".join(_format_value(v) for v in val)
        return f"[{items}]"
    return f'"{val}"'


def dump_toml(data: dict, path) -> None:
    """将 dict 序列化为 TOML 并写入文件

    支持一层 table 嵌套（dict 值作为 [table]）。
    顶层非 dict 值先输出，然后输出各 table。
    """
    lines = []

    # 先输出顶层简单键值
    for key, val in data.items():
        if not isinstance(val, dict):
            lines.append(f"{key} = {_format_value(val)}")

    # 输出 table
    for key, val in data.items():
        if isinstance(val, dict):
            if lines and lines[-1] != "":
                lines.append("")
            lines.append(f"[{key}]")
            for k, v in val.items():
                lines.append(f"{k} = {_format_value(v)}")

    # 末尾换行
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
