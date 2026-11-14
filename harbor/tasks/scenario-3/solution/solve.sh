#!/bin/bash
# Reference run. Plays the reference policy through the MCP server.
#   SOLUTION=reference   (default) the reference run
#   SOLUTION=stop_early  the reference for 10 weeks, then stop
#   SOLUTION=nothing     place no orders at all
set -euo pipefail
python3 /solution/play.py "${SOLUTION:-reference}"
