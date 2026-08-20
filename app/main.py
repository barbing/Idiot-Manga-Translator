# -*- coding: utf-8 -*-
"""App entry point."""
import os
import sys
from pathlib import Path


def _configure_dll_paths() -> None:
    if not hasattr(os, "add_dll_directory"):
        return
    prefix = Path(sys.prefix)
    candidates = [
        prefix / "Library" / "bin",
        prefix / "DLLs",
        prefix,
        prefix / "Lib" / "site-packages" / "torch" / "lib",
    ]
    for path in candidates:
        if path.exists():
            try:
                os.add_dll_directory(str(path))
            except OSError:
                pass
    torch_lib = prefix / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        os.environ["PATH"] = f"{torch_lib};{os.environ.get('PATH','')}"


def _configure_paddle_env() -> None:
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_enable_mkldnn", "0")
    os.environ.setdefault("FLAGS_enable_onednn", "0")
    os.environ.setdefault("PADDLE_DISABLE_MKLDNN", "1")


def _preload_torch() -> None:
    """Compatibility helper for explicit headless/full-run workers.

    The interactive GUI bootstrap deliberately does not call this helper;
    runtime readiness is handed to the shell only after it has been shown.
    """

    try:
        import torch  # noqa: F401
    except Exception:
        return


def _schedule_runtime_readiness(window: object) -> bool:
    """Hand deferred readiness ownership to a shell that implements it.

    The shell hook must schedule non-modal work after its first paint.  Calling
    the hook does not itself import a model runtime, touch the network, or run a
    readiness scan.  Legacy windows without the hook retain their own migration
    behavior until the GUI-7 cutover.
    """

    schedule = getattr(window, "schedule_runtime_readiness", None)
    if not callable(schedule):
        return False
    schedule()
    return True


def main() -> int:
    # Initialize logging first thing
    from app.utils.logger import setup_logger
    setup_logger()
    
    if sys.platform == "win32":
        try:
            import multiprocessing as mp
            mp.freeze_support()
        except Exception:
            pass
    _configure_dll_paths()
    _configure_paddle_env()
    from PySide6 import QtWidgets, QtGui
    from app.ui.application_coordinator import create_gui_application_window
    app = QtWidgets.QApplication(sys.argv)
    
    # Set global default font to prevent "Point size <= 0" errors
    font = QtGui.QFont("Microsoft YaHei", 10)
    font.setStyleStrategy(QtGui.QFont.PreferAntialias)
    app.setFont(font)
    
    window = create_gui_application_window()
    window.show()
    _schedule_runtime_readiness(window)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
