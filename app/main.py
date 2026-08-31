# -*- coding: utf-8 -*-
"""App entry point."""
import os
import sys
from pathlib import Path


APP_USER_MODEL_ID = "YomiFrame.MangaTranslator"


def _configure_windows_app_identity() -> None:
    """Give Windows one stable identity for taskbar and shortcut grouping."""

    if sys.platform != "win32":
        return
    try:
        import ctypes

        setter = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
        setter.argtypes = [ctypes.c_wchar_p]
        setter.restype = ctypes.c_long
        setter(APP_USER_MODEL_ID)
    except (AttributeError, OSError):
        # Source launches on stripped-down Windows environments still retain
        # the Qt application icon even if shell identity registration fails.
        pass


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
        from app.platform_services.compute import load_torch_runtime

        load_torch_runtime()
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


def configure_application_font(app: object):
    """Apply Qt's platform UI font without naming a Windows-only family."""

    from PySide6 import QtGui

    setter = getattr(app, "setFont", None)
    if not callable(setter):
        raise TypeError("application must provide setFont()")
    if sys.platform == "win32":
        font = QtGui.QFont("Microsoft YaHei", 10)
        font.setStyleStrategy(QtGui.QFont.StyleStrategy.PreferAntialias)
    else:
        font = QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.SystemFont.GeneralFont
        )
    setter(font)
    return font


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
    _configure_windows_app_identity()
    _configure_dll_paths()
    _configure_paddle_env()
    from PySide6 import QtWidgets
    from app.platform_services import build_platform_services
    from app.ui.application_coordinator import create_gui_application_window
    from app.ui.design_system.icons import brand_icon
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("YomiFrame")
    app.setApplicationDisplayName("YomiFrame Manga Translator")
    app.setOrganizationName("YomiFrame")
    app.setWindowIcon(brand_icon())

    configure_application_font(app)

    platform_services = build_platform_services()
    window = create_gui_application_window(
        platform_services=platform_services,
    )
    window.show()
    _schedule_runtime_readiness(window)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
