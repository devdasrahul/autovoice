"""AutoVoice root entrypoint for Streamlit Community Cloud and Hugging Face Spaces."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to Python path so autovoice package is discoverable
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from autovoice.ui.app import main

# Directly execute the Streamlit application
main()
