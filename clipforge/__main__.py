import sys

# Must run before any window is created so Windows treats Cilos as its own app
# instead of grouping it under python.exe (which is where the Python taskbar icon comes from).
if sys.platform == "win32":
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Cilos.ShortsMaker")

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())