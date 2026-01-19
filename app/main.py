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
    try:
        import torch  # noqa: F401
    except Exception:
        return


def main() -> int:
    if sys.platform == "win32":
        try:
            import multiprocessing as mp
            mp.freeze_support()
        except Exception:
            pass
    _configure_dll_paths()
    _configure_paddle_env()
    _preload_torch()
    from PySide6 import QtWidgets
    from app.ui.main_window import MainWindow
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
