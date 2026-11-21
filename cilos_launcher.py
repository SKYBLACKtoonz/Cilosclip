import sys

# Runs before any window is created so Windows treats Cilos as its own app
if sys.platform == "win32":
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Cilos.ShortsMaker")

from clipforge.cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("gui")  # double-click launches the GUI, like run.bat does
    raise SystemExit(main())