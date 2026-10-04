"""校验 YAML 文件语法（docker-compose.yml 与 CI 配置），并打印关键结构。"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

targets = ["docker-compose.yml", ".github/workflows/ci.yml"]
ok = True

for rel in targets:
    path = ROOT / rel
    if not path.exists():
        print(f"  MISSING {rel}")
        ok = False
        continue
    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL    {rel}: {type(exc).__name__}: {exc}")
        ok = False
        continue
    print(f"  OK      {rel}")
    if isinstance(data, dict):
        if "services" in data:
            print(f"          services = {list(data['services'])}")
            for name, svc in (data["services"] or {}).items():
                img = svc.get("image") or svc.get("build")
                health = "healthcheck" in svc
                print(f"            - {name:9s} build/image={img!r} healthcheck={health}")
            print(f"          volumes  = {list(data.get('volumes') or {})}")
            print(f"          networks = {list(data.get('networks') or {})}")
        if "jobs" in data:
            print(f"          jobs = {list(data['jobs'])}")
            for name, job in (data["jobs"] or {}).items():
                steps = job.get("steps") or []
                print(f"            - {name:9s} runs-on={job.get('runs-on')} steps={len(steps)}")
        if "on" in data or True in data:
            trigger = data.get("on", data.get(True))
            print(f"          triggers = {trigger}")

print()
print("RESULT:", "ALL YAML OK" if ok else "YAML ERROR")
sys.exit(0 if ok else 1)
