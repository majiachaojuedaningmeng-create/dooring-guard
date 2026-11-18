#!/usr/bin/env python3
"""DooringGuard 命令行入口。

    python main.py --demo
    python main.py --source road.mp4
    python main.py --source 0 --display
"""

import sys

from dooring_guard.cli import main

if __name__ == "__main__":
    sys.exit(main())
