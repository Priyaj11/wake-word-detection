"""Stand-alone converter: .tflite → C array.

Usage
-----
$ python tflite_to_c_array.py path/to/model.tflite path/to/model.cpp
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


def emit(in_path: Path, out_path: Path, var_name: str = "g_model") -> None:
    data = in_path.read_bytes()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = out_path.with_suffix(".h")

    lines = []
    for i in range(0, len(data), 12):
        chunk = data[i:i + 12]
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    array_body = "\n".join(lines).rstrip(",")

    stamp = datetime.utcnow().isoformat() + "Z"
    out_path.write_text(
        f"// Auto-generated on {stamp}\n"
        f"#include \"{header.name}\"\n\n"
        f"alignas(16) const unsigned char {var_name}[] = {{\n"
        f"{array_body}\n}};\n\n"
        f"const unsigned int {var_name}_len = {len(data)};\n"
    )
    header.write_text(
        "#pragma once\n#ifdef __cplusplus\nextern \"C\" {\n#endif\n"
        f"extern const unsigned char {var_name}[];\n"
        f"extern const unsigned int {var_name}_len;\n"
        "#ifdef __cplusplus\n}\n#endif\n"
    )
    print(f"[ok] wrote {out_path} ({len(data)} bytes)")
    print(f"[ok] wrote {header}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="path to .tflite")
    p.add_argument("output", type=Path, help="path to .cpp (and .h sibling)")
    p.add_argument("--var", default="g_model")
    args = p.parse_args()
    if not args.input.exists():
        print(f"[error] {args.input} not found", file=sys.stderr)
        return 1
    emit(args.input, args.output, args.var)
    return 0


if __name__ == "__main__":
    sys.exit(main())
