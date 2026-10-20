"""AutoVoice root entrypoint for Streamlit Community Cloud and Hugging Face Spaces."""
import sys
from pathlib import Path

# Add src to Python path so autovoice package is discoverable
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Execute the AutoVoice Streamlit application
import autovoice.ui.app  # noqa: F401
