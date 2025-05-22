# main.py

import sys
from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow

try:
    from config import STYLESHEET_PATH
except ImportError:
    # Fallback if config.py is not found or STYLESHEET_PATH is not defined
    print("Warning: config.py not found or STYLESHEET_PATH not defined. Stylesheet will not be loaded globally.", file=sys.stderr)
    STYLESHEET_PATH = "" # Ensure it's an empty string if not found


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Load dark theme stylesheet globally if path is configured and valid
    # MainWindow's __main__ block also tries to load it, but this makes it global sooner.
    if STYLESHEET_PATH: # Check if the path is not empty
        try:
            with open(STYLESHEET_PATH, encoding="utf-8") as f:
                app.setStyleSheet(f.read())
            print(f"Global stylesheet '{STYLESHEET_PATH}' loaded successfully from main.py.")
        except FileNotFoundError:
            # This might be redundant if gui.main_window also tries and fails,
            # but good for clarity if running main.py directly.
            print(f"Warning: Stylesheet not found at '{STYLESHEET_PATH}' when loading from main.py. Application might use default styles or attempt local load.", file=sys.stderr)
        except Exception as e:
            print(f"Could not load stylesheet '{STYLESHEET_PATH}' from main.py: {e}", file=sys.stderr)
    else:
        print("No global stylesheet path configured or found. Application will use default styles or attempt local load in MainWindow.", file=sys.stderr)


    window = MainWindow() # MainWindow itself will also try to load its stylesheet if __name__ == '__main__' in its file
    window.show()
    sys.exit(app.exec())