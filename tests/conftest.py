import os
import sys

# Ensure project root (parent of this tests dir) is on sys.path for imports like
# `from scrape_collocations import html_line_to_cloze_text`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT) 