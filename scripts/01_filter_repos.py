"""
Step 1: Filter and clean repository metadata.
 
This script processes raw preprocessed data and extracts clean repository
metadata including names, descriptions, and programming languages.
"""

import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
