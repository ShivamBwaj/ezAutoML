#!/usr/bin/env python3
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "auto_ml_research_agent.main", "predict house prices"],
    capture_output=False,
    text=True,
    timeout=300  # 5 minute timeout
)
print("\n\n=== EXIT CODE:", result.returncode, "===")
