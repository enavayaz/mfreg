import sys
import os

# Add the project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print(f"Project root: {project_root}")
print(f"sys.path[0]: {sys.path[0]}")

try:
    from TimeSeries import stats
    print("Successfully imported TimeSeries.stats")
except ImportError as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
