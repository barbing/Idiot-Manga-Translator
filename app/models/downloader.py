# -*- coding: utf-8 -*-
"""Model downloader logic."""
import contextlib
from email.parser import Parser
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
import time
import uuid
import requests
from PySide6 import QtCore
from app.config.defaults import (
    CLEANUP_INPAINT_MODEL_FILE,
    CLEANUP_INPAINT_REPO_ID,
    COMIC_TEXT_DETECTOR_GPU, 
    COMIC_TEXT_DETECTOR_CPU, 
    KITSUMED_SPEECH_BUBBLE_MODEL_FILE,
    KITSUMED_SPEECH_BUBBLE_REPO_ID,
    MANGA_OCR_BASE_URL,
    MANGA_OCR_FILES,
    NOTO_CJK_SC_FONT_BASE_URL,
    NOTO_CJK_SC_FONT_FILES,
    NOTO_LATIN_FONT_BASE_URL,
    NOTO_LATIN_FONT_FILE,
    NOTO_LATIN_FONT_SHA256,
    OGKALU_TEXT_BUBBLE_CONFIG_FILE,
    OGKALU_TEXT_BUBBLE_MODEL_FILE,
    OGKALU_TEXT_BUBBLE_REPO_ID,
    QWEN_GGUF,
    SAKURA_GGUF,
    SIL_OFL_TEXT_URL,
    YUZUMARKER_FONT_LABELS_FALLBACK_FILE,
    YUZUMARKER_FONT_LABELS_FILE,
    YUZUMARKER_FONT_LABELS_REPO_ID,
    YUZUMARKER_FONT_ONNX_FILE,
    YUZUMARKER_FONT_ONNX_REPO_ID,
)

import hashlib
from dataclasses import dataclass
from typing import Callable, Iterator, List
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import tarfile
import zipfile
from app.models.resolution import (
    has_bubble_detection_runtime,
    has_cleanup_inpaint_model,
    has_font_style_runtime,
    has_paddle_ocr_vl_runtime,
    noto_cjk_sc_font_dir,
    noto_latin_font_dir,
    resolve_manga_ocr_local_dir,
    resolve_manga_ocr_system_ref,
    resolve_ner_local_dir,
    resolve_ner_system_snapshot,
)
from app.platform_services.paths import qt_platform_paths
from app.platform_services.compute import invalidate_compute_capability_cache
from app.platform_services.contracts import OperatingSystem, PlatformIdentity
from app.platform_services.runtime_assets import (
    paddle_targets,
    runtime_asset_spec,
)


PYICU_RUNTIME_PYICU_VERSION = "2.16.2"
PYICU_RUNTIME_ICU_VERSION = "78.3"
PYICU_RUNTIME_WHEEL_NAME = "pyicu-2.16.2-cp310-cp310-win_amd64.whl"
PYICU_RUNTIME_SHA256 = "a20721fe04dcfd8b34c17e2f45ba45beebaf32f1a03bd07efc07a512c2b3f830"
PYICU_RUNTIME_URL = (
    "https://github.com/barbing/YomiFrame-LLM_Manga_Translator/releases/download/"
    "runtime-dependencies-v1/"
    f"{PYICU_RUNTIME_WHEEL_NAME}"
)
PYICU_RUNTIME_ID = (
    f"pyicu-{PYICU_RUNTIME_PYICU_VERSION}-icu-{PYICU_RUNTIME_ICU_VERSION}"
    "-cp310-win_amd64"
)
PYICU_RUNTIME_REQUIRED_FILES = {
    "icu/__init__.py": 1589,
    "icu/_icu_.cp310-win_amd64.pyd": 1239552,
    "icu/icudt78.dll": 33110528,
    "icu/icuin78.dll": 3408896,
    "icu/icuuc78.dll": 2576384,
}

_PYICU_DLL_DIRECTORY_HANDLES: list[object] = []
_PYICU_ACTIVATED_PATHS: set[str] = set()


class PyICURuntimeError(RuntimeError):
    """Raised when the required application-private PyICU runtime is invalid."""

@dataclass
class DownloadTarget:
    url: str
    save_path: str
    label: str
    sha256: str = None  # Optional checksum

class ModelDownloader(QtCore.QObject):
    progress_changed = QtCore.Signal(int)
    status_changed = QtCore.Signal(str)
    finished = QtCore.Signal(bool, str)  # success, message

    def __init__(
        self,
        parent=None,
        *,
        pyicu_runtime_url: str | None = None,
        pyicu_runtime_sha256: str | None = None,
        pyicu_runtime_root: str | os.PathLike[str] | None = None,
        platform_identity: PlatformIdentity | None = None,
    ):
        super().__init__(parent)
        self._cancel_requested = False
        self._session = self._create_session()
        self._pending_targets: List[DownloadTarget] = []
        self._pyicu_runtime_url = str(pyicu_runtime_url or PYICU_RUNTIME_URL)
        self._pyicu_runtime_sha256 = str(
            pyicu_runtime_sha256 or PYICU_RUNTIME_SHA256
        ).lower()
        self._pyicu_runtime_root_override = (
            Path(pyicu_runtime_root).resolve()
            if pyicu_runtime_root is not None
            else None
        )
        self._pyicu_install_requested = False
        self._pyicu_wheel_path: Path | None = None
        self._pyicu_last_error = ""
        self._platform_identity = platform_identity or PlatformIdentity.detect()
        if not isinstance(self._platform_identity, PlatformIdentity):
            raise TypeError("platform_identity must be PlatformIdentity")
        self._paddle_runtime_verification_required = False
        self._paddle_runtime_error = ""

    def _create_session(self) -> requests.Session:
        """Create a robust requests session with retries."""
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({
            "User-Agent": "YomiFrame/1.2.0"
        })
        return session

    @property
    def pyicu_runtime_error(self) -> str:
        return self._pyicu_last_error

    @property
    def paddle_runtime_error(self) -> str:
        return self._paddle_runtime_error

    @staticmethod
    def is_frozen_application() -> bool:
        return bool(getattr(sys, "frozen", False))

    @staticmethod
    def _source_runtime_supported() -> bool:
        return (
            os.name == "nt"
            and sys.version_info[:2] == (3, 10)
            and sys.maxsize > 2**32
        )

    def can_install_pyicu_runtime(self) -> bool:
        return not self.is_frozen_application() and self._source_runtime_supported()

    def _pyicu_runtime_root(self) -> Path:
        if self._pyicu_runtime_root_override is not None:
            return self._pyicu_runtime_root_override
        return qt_platform_paths().runtime_root.resolve()

    def _pyicu_install_dir(self) -> Path:
        return self._pyicu_runtime_root() / PYICU_RUNTIME_ID

    def _pyicu_download_path(self) -> Path:
        return self._pyicu_runtime_root() / "downloads" / PYICU_RUNTIME_WHEEL_NAME

    @staticmethod
    def _path_is_within(path: Path, root: Path) -> bool:
        try:
            return os.path.commonpath([str(path.resolve()), str(root.resolve())]) == str(
                root.resolve()
            )
        except (OSError, ValueError):
            return False

    def _require_runtime_path(self, path: Path) -> Path:
        root = self._pyicu_runtime_root()
        resolved = path.resolve()
        if not self._path_is_within(resolved, root):
            raise PyICURuntimeError(f"Runtime path escapes managed root: {resolved}")
        return resolved

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def request_cancel(self):
        self._cancel_requested = True

    @staticmethod
    def _probe_pyicu_module(icu_module) -> None:
        pyicu_version = str(getattr(icu_module, "VERSION", ""))
        icu_version = str(getattr(icu_module, "ICU_VERSION", ""))
        if pyicu_version != PYICU_RUNTIME_PYICU_VERSION:
            raise PyICURuntimeError(
                f"PyICU {PYICU_RUNTIME_PYICU_VERSION} is required; found {pyicu_version or 'unknown'}"
            )
        if icu_version != PYICU_RUNTIME_ICU_VERSION:
            raise PyICURuntimeError(
                f"ICU {PYICU_RUNTIME_ICU_VERSION} is required; found {icu_version or 'unknown'}"
            )
        try:
            locale = icu_module.Locale("zh@lb=strict")
            iterator = icu_module.BreakIterator.createLineInstance(locale)
            probe_text = "甲～乙"
            iterator.setText(probe_text)
            boundaries = tuple(int(value) for value in iterator)
        except Exception as exc:
            raise PyICURuntimeError(f"PyICU strict line-break probe failed: {exc}") from exc
        if not boundaries or boundaries[-1] != len(probe_text):
            raise PyICURuntimeError(
                f"PyICU strict line-break probe returned invalid boundaries: {boundaries}"
            )

    @staticmethod
    def _loaded_pyicu_module():
        return sys.modules.get("icu")

    @classmethod
    def _validate_loaded_pyicu(cls) -> bool:
        module = cls._loaded_pyicu_module()
        if module is None:
            return False
        cls._probe_pyicu_module(module)
        return True

    def _managed_runtime_layout_valid(self, runtime_dir: Path) -> bool:
        try:
            runtime_dir = self._require_runtime_path(runtime_dir)
            marker_path = runtime_dir / "runtime.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker != {
                "icu_version": PYICU_RUNTIME_ICU_VERSION,
                "pyicu_version": PYICU_RUNTIME_PYICU_VERSION,
                "runtime_id": PYICU_RUNTIME_ID,
                "wheel_name": PYICU_RUNTIME_WHEEL_NAME,
                "wheel_sha256": PYICU_RUNTIME_SHA256,
            }:
                return False
            for relative_path, expected_size in PYICU_RUNTIME_REQUIRED_FILES.items():
                file_path = self._require_runtime_path(runtime_dir / relative_path)
                if not file_path.is_file() or file_path.stat().st_size != expected_size:
                    return False
            return True
        except (OSError, ValueError, TypeError, PyICURuntimeError, json.JSONDecodeError):
            return False

    def _activate_managed_pyicu(self, runtime_dir: Path) -> None:
        runtime_dir = self._require_runtime_path(runtime_dir)
        if not self._managed_runtime_layout_valid(runtime_dir):
            raise PyICURuntimeError(f"Managed PyICU runtime is incomplete: {runtime_dir}")

        loaded = self._loaded_pyicu_module()
        if loaded is not None:
            self._probe_pyicu_module(loaded)
            loaded_path = Path(str(getattr(loaded, "__file__", ""))).resolve()
            if not self._path_is_within(loaded_path, runtime_dir):
                raise PyICURuntimeError(
                    "A different PyICU runtime is already loaded; restart YomiFrame "
                    "to activate the managed runtime"
                )
            return

        runtime_text = str(runtime_dir)
        package_dir = runtime_dir / "icu"
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            normalized_package = str(package_dir.resolve()).lower()
            if normalized_package not in _PYICU_ACTIVATED_PATHS:
                handle = os.add_dll_directory(str(package_dir))
                _PYICU_DLL_DIRECTORY_HANDLES.append(handle)
                _PYICU_ACTIVATED_PATHS.add(normalized_package)
        if runtime_text not in sys.path:
            sys.path.insert(0, runtime_text)
        importlib.invalidate_caches()
        try:
            module = importlib.import_module("icu")
            self._probe_pyicu_module(module)
            loaded_path = Path(str(getattr(module, "__file__", ""))).resolve()
            if not self._path_is_within(loaded_path, runtime_dir):
                raise PyICURuntimeError(
                    f"PyICU resolved outside the managed runtime: {loaded_path}"
                )
        except Exception:
            sys.modules.pop("icu._icu_", None)
            sys.modules.pop("icu", None)
            if runtime_text in sys.path:
                sys.path.remove(runtime_text)
            importlib.invalidate_caches()
            raise

    def _activate_environment_pyicu_if_exact(self) -> bool:
        try:
            if importlib.metadata.version("PyICU") != PYICU_RUNTIME_PYICU_VERSION:
                return False
        except importlib.metadata.PackageNotFoundError:
            return False
        module = importlib.import_module("icu")
        self._probe_pyicu_module(module)
        return True

    def check_pyicu_runtime(self) -> bool:
        """Activate and validate the required PyICU runtime without network I/O."""

        self._pyicu_last_error = ""
        try:
            if self._loaded_pyicu_module() is not None:
                return self._validate_loaded_pyicu()

            if self.is_frozen_application():
                module = importlib.import_module("icu")
                self._probe_pyicu_module(module)
                return True

            if self._activate_environment_pyicu_if_exact():
                return True

            if self._platform_identity.os is OperatingSystem.WINDOWS:
                runtime_dir = self._pyicu_install_dir()
                if self._managed_runtime_layout_valid(runtime_dir):
                    self._activate_managed_pyicu(runtime_dir)
                    return True

            self._pyicu_last_error = runtime_asset_spec(
                "pyicu",
                self._platform_identity,
            ).remediation_for(self._platform_identity)
            return False
        except Exception as exc:
            message = str(exc)
            if self._platform_identity.os is OperatingSystem.MACOS:
                remediation = runtime_asset_spec(
                    "pyicu",
                    self._platform_identity,
                ).remediation_for(self._platform_identity)
                if remediation not in message:
                    message = f"{message}. {remediation}"
            self._pyicu_last_error = message
            return False

    def prepare_pyicu_runtime(self) -> None:
        """Queue the pinned application-private PyICU runtime when required."""

        if self.check_pyicu_runtime():
            return
        self._pyicu_install_requested = True
        if self.is_frozen_application():
            self._pyicu_last_error = (
                "The packaged application is missing its bundled PyICU runtime; "
                "startup download is disabled for frozen builds"
            )
            return
        if not self._source_runtime_supported():
            self._pyicu_last_error = (
                "Managed PyICU installation requires Windows CPython 3.10 x64"
            )
            return
        wheel_path = self._require_runtime_path(self._pyicu_download_path())
        self._pyicu_wheel_path = wheel_path
        self.queue_targets(
            [
                DownloadTarget(
                    url=self._pyicu_runtime_url,
                    save_path=str(wheel_path),
                    label=(
                        f"Downloading PyICU {PYICU_RUNTIME_PYICU_VERSION} / "
                        f"ICU {PYICU_RUNTIME_ICU_VERSION} runtime..."
                    ),
                    sha256=self._pyicu_runtime_sha256,
                )
            ]
        )

    @contextlib.contextmanager
    def _pyicu_install_lock(self) -> Iterator[None]:
        runtime_root = self._require_runtime_path(self._pyicu_runtime_root())
        runtime_root.mkdir(parents=True, exist_ok=True)
        lock_path = self._require_runtime_path(runtime_root / ".pyicu-install.lock")
        deadline = time.monotonic() + 30.0
        acquired = False
        while not acquired:
            if self._cancel_requested:
                raise PyICURuntimeError("Cancelled")
            try:
                descriptor = os.open(
                    str(lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                try:
                    os.write(descriptor, f"{os.getpid()}\n{time.time()}\n".encode("ascii"))
                finally:
                    os.close(descriptor)
                acquired = True
            except FileExistsError:
                try:
                    if time.time() - lock_path.stat().st_mtime > 600:
                        lock_path.unlink()
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() >= deadline:
                    raise PyICURuntimeError(
                        "Timed out waiting for another PyICU installation"
                    )
                time.sleep(0.1)
        try:
            yield
        finally:
            if acquired:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass

    def _extract_pyicu_wheel(self, wheel_path: Path, destination: Path) -> None:
        wheel_path = self._require_runtime_path(wheel_path)
        destination = self._require_runtime_path(destination)
        with zipfile.ZipFile(wheel_path, "r") as archive:
            names = set(archive.namelist())
            missing = sorted(set(PYICU_RUNTIME_REQUIRED_FILES) - names)
            if missing:
                raise PyICURuntimeError(
                    f"PyICU wheel is missing required files: {', '.join(missing)}"
                )
            metadata_name = (
                f"pyicu-{PYICU_RUNTIME_PYICU_VERSION}.dist-info/METADATA"
            )
            if metadata_name not in names:
                raise PyICURuntimeError("PyICU wheel metadata is missing")
            metadata_text = archive.read(metadata_name).decode("utf-8", errors="strict")
            metadata = Parser().parsestr(metadata_text, headersonly=True)
            distribution_names = metadata.get_all("Name", [])
            distribution_versions = metadata.get_all("Version", [])
            canonical_name = (
                re.sub(r"[-_.]+", "-", distribution_names[0]).lower()
                if len(distribution_names) == 1
                else ""
            )
            if (
                canonical_name != "pyicu"
                or distribution_versions != [PYICU_RUNTIME_PYICU_VERSION]
            ):
                raise PyICURuntimeError("PyICU wheel metadata does not match the runtime pin")

            for member in archive.infolist():
                if self._cancel_requested:
                    raise PyICURuntimeError("Cancelled")
                mode = (member.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    raise PyICURuntimeError(
                        f"Unsafe PyICU wheel entry (link): {member.filename}"
                    )
                target = Path(_safe_extract_path(str(destination), member.filename))
                self._require_runtime_path(target)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member, "r") as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)

        for relative_path, expected_size in PYICU_RUNTIME_REQUIRED_FILES.items():
            extracted = self._require_runtime_path(destination / relative_path)
            if not extracted.is_file() or extracted.stat().st_size != expected_size:
                raise PyICURuntimeError(
                    f"Extracted PyICU runtime file is invalid: {relative_path}"
                )

    def _install_pyicu_runtime(self, wheel_path: Path) -> Path:
        wheel_path = self._require_runtime_path(wheel_path)
        if not wheel_path.is_file():
            raise PyICURuntimeError("Downloaded PyICU wheel is missing")
        actual_sha256 = self._hash_file(wheel_path)
        if actual_sha256.lower() != self._pyicu_runtime_sha256:
            raise PyICURuntimeError("Downloaded PyICU wheel checksum mismatch")
        if self._pyicu_runtime_sha256 != PYICU_RUNTIME_SHA256:
            raise PyICURuntimeError("PyICU runtime hash differs from the production pin")

        runtime_root = self._require_runtime_path(self._pyicu_runtime_root())
        runtime_root.mkdir(parents=True, exist_ok=True)
        destination = self._require_runtime_path(self._pyicu_install_dir())
        with self._pyicu_install_lock():
            if self._managed_runtime_layout_valid(destination):
                return destination

            staging = Path(
                tempfile.mkdtemp(
                    prefix=f".{PYICU_RUNTIME_ID}-staging-",
                    dir=str(runtime_root),
                )
            ).resolve()
            staging = self._require_runtime_path(staging)
            backup: Path | None = None
            installed = False
            try:
                self._extract_pyicu_wheel(wheel_path, staging)
                marker = {
                    "icu_version": PYICU_RUNTIME_ICU_VERSION,
                    "pyicu_version": PYICU_RUNTIME_PYICU_VERSION,
                    "runtime_id": PYICU_RUNTIME_ID,
                    "wheel_name": PYICU_RUNTIME_WHEEL_NAME,
                    "wheel_sha256": PYICU_RUNTIME_SHA256,
                }
                (staging / "runtime.json").write_text(
                    json.dumps(marker, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                if not self._managed_runtime_layout_valid(staging):
                    raise PyICURuntimeError("Staged PyICU runtime validation failed")

                if destination.exists():
                    backup = self._require_runtime_path(
                        runtime_root
                        / f".{PYICU_RUNTIME_ID}-replaced-{uuid.uuid4().hex}"
                    )
                    os.replace(destination, backup)
                os.replace(staging, destination)
                installed = True
                if backup is not None:
                    shutil.rmtree(backup, ignore_errors=True)
                return destination
            except Exception:
                if backup is not None and backup.exists() and not destination.exists():
                    os.replace(backup, destination)
                raise
            finally:
                if not installed and staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)

    def _complete_pyicu_runtime_install(self) -> bool:
        if not self._pyicu_install_requested:
            return True
        try:
            if self.check_pyicu_runtime():
                self._pyicu_install_requested = False
                return True
            if self._pyicu_wheel_path is None:
                raise PyICURuntimeError(
                    self._pyicu_last_error or "PyICU runtime cannot be installed"
                )
            self.status_changed.emit("Installing application-private PyICU runtime...")
            runtime_dir = self._install_pyicu_runtime(self._pyicu_wheel_path)
            self._activate_managed_pyicu(runtime_dir)
            if not self.check_pyicu_runtime():
                raise PyICURuntimeError(
                    self._pyicu_last_error or "Installed PyICU runtime validation failed"
                )
            self._pyicu_install_requested = False
            self.status_changed.emit(
                f"PyICU {PYICU_RUNTIME_PYICU_VERSION} / ICU "
                f"{PYICU_RUNTIME_ICU_VERSION} runtime is ready."
            )
            try:
                self._pyicu_wheel_path.unlink()
            except FileNotFoundError:
                pass
            return True
        except Exception as exc:
            self._pyicu_last_error = str(exc)
            self.finished.emit(False, f"PyICU runtime setup failed: {exc}")
            return False

    def _check_hf_cache(self, user: str, repo: str, filename: str = None) -> bool:
        """Check Hugging Face system caches (respects HF_HOME)."""
        try:
            # 1. Check environment variable
            hf_home = os.environ.get("HF_HOME")
            if not hf_home:
                hf_home = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
            
            # Standard HF cache structure: hub/models--user--repo/snapshots/hash/filename
            # We check if *any* snapshot exists and contains the file (if specified)
            model_dir = os.path.join(hf_home, "hub", f"models--{user}--{repo}")
            if not os.path.exists(model_dir):
                return False
                
            snapshots_dir = os.path.join(model_dir, "snapshots")
            if not os.path.exists(snapshots_dir):
                return False
            
            # Check all snapshots
            for snapshot in os.listdir(snapshots_dir):
                snap_path = os.path.join(snapshots_dir, snapshot)
                if not os.path.isdir(snap_path):
                    continue
                
                # If filename specified, check for it
                if filename:
                    if os.path.exists(os.path.join(snap_path, filename)):
                        return True
                else:
                    # If no filename, just existence of snapshot is enough
                    return True
                    
        except Exception:
            pass
        return False

    def check_comic_text_detector(self, models_dir: str) -> bool:
        """Check if ComicTextDetector models exist (Portable ONLY per user request)."""
        target_dir = os.path.join(models_dir, "comic-text-detector")
        path_gpu = os.path.join(target_dir, "comictextdetector.pt")
        path_cpu = os.path.join(target_dir, "comictextdetector.pt.onnx")
        
        return os.path.exists(path_gpu) and os.path.exists(path_cpu)

    def check_manga_ocr(self, models_dir: str) -> bool:
        """Check if MangaOCR models exist (System Priority > Portable)."""
        return bool(resolve_manga_ocr_system_ref() or resolve_manga_ocr_local_dir(models_dir))

    def check_bubble_detection(self, models_dir: str) -> bool:
        """Check if Phase 4 bubble/text-area semantic evidence models exist."""
        return has_bubble_detection_runtime(models_dir)

    def check_big_lama(self, models_dir: str) -> bool:
        """Check if the fixed cleanup-owned iopaint model exists."""
        return has_cleanup_inpaint_model(models_dir)

    def download_targets(self, targets: List[DownloadTarget]):
        """Generic downloader for a list of targets."""
        for target in targets:
            if self._cancel_requested:
                return

            # Check if file exists and verify checksum if provided
            if os.path.exists(target.save_path):
                if target.sha256:
                    self.status_changed.emit(f"Verifying {os.path.basename(target.save_path)}...")
                    if self._verify_checksum(target.save_path, target.sha256):
                        if target.save_path.endswith((".tar", ".zip")):
                            if not self._extract_archive(target.save_path):
                                return
                        continue # File is good
                    else:
                        os.remove(target.save_path) # Corrupt, re-download
                else:
                    if target.save_path.endswith((".tar", ".zip")):
                        if not self._extract_archive(target.save_path):
                            return
                    continue # Assume good if no checksum

            os.makedirs(os.path.dirname(target.save_path), exist_ok=True)
            if not self._download_file(target):
                return

    def queue_targets(self, targets: List[DownloadTarget]):
        """Queue targets for download."""
        self._pending_targets.extend(targets)

    def check_ner(self, models_dir: str) -> bool:
        """Check if NER model exists (System Priority > Portable)."""
        return bool(resolve_ner_system_snapshot() or resolve_ner_local_dir(os.path.join(models_dir, "ner")))

    def check_paddle_ocr_vl(self, models_dir: str = "models") -> bool:
        """Verify model files and a launchable native runtime without loading it."""

        self._paddle_runtime_error = ""
        return has_paddle_ocr_vl_runtime(
            base_dir=models_dir,
            identity=self._platform_identity,
        )

    def check_font_detection(self, models_dir: str = "models") -> bool:
        """Check if YuzuMarker font detection and local CJK fallback fonts exist."""
        return has_font_style_runtime(base_dir=models_dir)

    def check_yuzumarker_font_detection(self, models_dir: str = "models") -> bool:
        """Check only the YuzuMarker ONNX model and label metadata."""
        from app.models.resolution import has_yuzumarker_font_detection_runtime

        return has_yuzumarker_font_detection_runtime(base_dir=models_dir)

    def check_noto_cjk_sc_font_pack(self, models_dir: str = "models") -> bool:
        """Check the platform-neutral CJK and Latin renderer font pack."""
        from app.models.resolution import (
            has_noto_cjk_sc_font_pack,
            has_noto_latin_font_pack,
        )

        return bool(
            has_noto_cjk_sc_font_pack(base_dir=models_dir)
            and has_noto_latin_font_pack(base_dir=models_dir)
        )

    def prepare_ner(self, models_dir: str):
        """Queue NER model download."""
        self._ner_target_dir = os.path.join(models_dir, "ner")
        self._download_ner = True

    def prepare_paddle_ocr_vl(self, models_dir: str):
        """Queue model files and only the runtime archives for this platform."""
        targets = [
            DownloadTarget(
                url=item.url,
                save_path=os.path.join(models_dir, item.relative_path),
                label=item.label,
                sha256=item.sha256,
            )
            for item in paddle_targets(self._platform_identity)
        ]
        self._paddle_models_dir = str(models_dir)
        self._paddle_runtime_verification_required = True
        self.queue_targets(targets)

    def prepare_bubble_detection(self, models_dir: str):
        """Queue root/parent semantic bubble and text-area evidence models."""
        kitsumed_dir = os.path.join(models_dir, "yolov8m_seg-speech-bubble")
        ogkalu_dir = os.path.join(models_dir, "comic-text-and-bubble-detector")
        kitsumed_url = f"https://huggingface.co/{KITSUMED_SPEECH_BUBBLE_REPO_ID}/resolve/main"
        ogkalu_url = f"https://huggingface.co/{OGKALU_TEXT_BUBBLE_REPO_ID}/resolve/main"
        targets = [
            DownloadTarget(
                url=f"{kitsumed_url}/{KITSUMED_SPEECH_BUBBLE_MODEL_FILE}",
                save_path=os.path.join(kitsumed_dir, KITSUMED_SPEECH_BUBBLE_MODEL_FILE),
                label="Downloading speech-bubble segmentation model...",
            ),
            DownloadTarget(
                url=f"{ogkalu_url}/{OGKALU_TEXT_BUBBLE_MODEL_FILE}",
                save_path=os.path.join(ogkalu_dir, OGKALU_TEXT_BUBBLE_MODEL_FILE),
                label="Downloading text/bubble detector model...",
            ),
            DownloadTarget(
                url=f"{ogkalu_url}/{OGKALU_TEXT_BUBBLE_CONFIG_FILE}",
                save_path=os.path.join(ogkalu_dir, OGKALU_TEXT_BUBBLE_CONFIG_FILE),
                label="Downloading text/bubble detector config...",
            ),
        ]
        self.queue_targets(targets)

    def prepare_yuzumarker_font_detection(self, models_dir: str):
        """Queue only the YuzuMarker font-detection model and labels."""
        onnx_dir = os.path.join(models_dir, "YuzuMarker", "onnx")
        labels_dir = os.path.join(models_dir, "YuzuMarker", "safetensors")
        onnx_url = f"https://huggingface.co/{YUZUMARKER_FONT_ONNX_REPO_ID}/resolve/main"
        labels_url = f"https://huggingface.co/{YUZUMARKER_FONT_LABELS_REPO_ID}/resolve/main"
        targets = [
            DownloadTarget(
                url=f"{onnx_url}/{YUZUMARKER_FONT_ONNX_FILE}",
                save_path=os.path.join(onnx_dir, YUZUMARKER_FONT_ONNX_FILE),
                label="Downloading YuzuMarker font detector ONNX model...",
                sha256="99dd351e94f06e31397113602ae000a24c1d38ad76275066e844a0c836f75d4f",
            ),
            DownloadTarget(
                url=f"{labels_url}/{YUZUMARKER_FONT_LABELS_FILE}",
                save_path=os.path.join(labels_dir, YUZUMARKER_FONT_LABELS_FILE),
                label="Downloading YuzuMarker font labels...",
            ),
            DownloadTarget(
                url=f"{labels_url}/{YUZUMARKER_FONT_LABELS_FALLBACK_FILE}",
                save_path=os.path.join(labels_dir, YUZUMARKER_FONT_LABELS_FALLBACK_FILE),
                label="Downloading YuzuMarker fallback font labels...",
            ),
        ]
        self.queue_targets(targets)

    def prepare_font_detection(self, models_dir: str):
        """Queue YuzuMarker detection and local CJK fallback font assets."""
        self.prepare_yuzumarker_font_detection(models_dir)
        self.prepare_noto_cjk_sc_font_pack(models_dir)

    def prepare_noto_cjk_sc_font_pack(self, models_dir: str):
        """Queue the local CJK fallback and condensed Latin renderer fonts."""

        font_dir = noto_cjk_sc_font_dir(models_dir)
        targets = []
        for relative_path in NOTO_CJK_SC_FONT_FILES:
            targets.append(
                DownloadTarget(
                    url=f"{NOTO_CJK_SC_FONT_BASE_URL}/{relative_path}",
                    save_path=os.path.join(font_dir, os.path.basename(relative_path)),
                    label=f"Downloading Noto CJK SC font: {os.path.basename(relative_path)}",
                )
            )
        targets.append(
            DownloadTarget(
                url=SIL_OFL_TEXT_URL,
                save_path=os.path.join(font_dir, "OFL.txt"),
                label="Downloading SIL Open Font License text...",
            )
        )
        latin_dir = noto_latin_font_dir(models_dir)
        targets.extend(
            [
                DownloadTarget(
                    url=f"{NOTO_LATIN_FONT_BASE_URL}/{NOTO_LATIN_FONT_FILE}",
                    save_path=os.path.join(latin_dir, NOTO_LATIN_FONT_FILE),
                    label="Downloading Noto Sans variable Latin font...",
                    sha256=NOTO_LATIN_FONT_SHA256,
                ),
                DownloadTarget(
                    url=SIL_OFL_TEXT_URL,
                    save_path=os.path.join(latin_dir, "OFL.txt"),
                    label="Downloading Noto Sans SIL Open Font License text...",
                ),
            ]
        )
        self.queue_targets(targets)

    def _perform_ner_download(self) -> bool:
        """Execute NER download using transformers."""
        try:
            from app.nlp.ner_extractor import download_ner_model
            
            def progress_adapter(percent):
                self.progress_changed.emit(percent)
                
            self.status_changed.emit("Downloading NER Model (bert-ner-japanese)...")
            return download_ner_model(self._ner_target_dir, progress_callback=progress_adapter)
        except Exception as e:
            self.finished.emit(False, f"NER Download failed: {e}")
            return False

    def process_queue(self):
        """Process queued targets (Slot)."""
        if (
            not self._pending_targets
            and not getattr(self, "_download_ner", False)
            and not self._pyicu_install_requested
        ):
            self.finished.emit(True, "No tasks.")
            return

        if self._pending_targets:
            self.download_targets(self._pending_targets)
            self._pending_targets.clear()
             
        if self._cancel_requested:
            self.finished.emit(False, "Cancelled")
            return

        if self._pyicu_install_requested:
            if not self._complete_pyicu_runtime_install():
                return

        if self._cancel_requested:
            self.finished.emit(False, "Cancelled")
            return

        # Execute NER download if queued
        if getattr(self, "_download_ner", False):
            success = self._perform_ner_download()
            self._download_ner = False
            if not success:
                return

        if self._paddle_runtime_verification_required:
            self._paddle_runtime_verification_required = False
            if not self.check_paddle_ocr_vl(str(getattr(self, "_paddle_models_dir", "models"))):
                remediation = runtime_asset_spec(
                    "paddle_ocr_vl",
                    self._platform_identity,
                ).remediation_for(self._platform_identity)
                self.finished.emit(False, f"PaddleOCR-VL runtime is incomplete. {remediation}")
                return
            invalidate_compute_capability_cache()

        if self._cancel_requested:
            self.finished.emit(False, "Cancelled")
            return

        self.finished.emit(True, "All downloads completed.")


    def prepare_comic_text_detector(self, models_dir: str):
        """Queue ComicTextDetector models."""
        target_dir = os.path.join(models_dir, "comic-text-detector")
        targets = [
            DownloadTarget(
                COMIC_TEXT_DETECTOR_CPU,
                os.path.join(target_dir, "comictextdetector.pt.onnx"),
                "Downloading ComicTextDetector (CPU)..."
            ),
            DownloadTarget(
                COMIC_TEXT_DETECTOR_GPU,
                os.path.join(target_dir, "comictextdetector.pt"),
                "Downloading ComicTextDetector (GPU)..."
            )
        ]
        self.queue_targets(targets)

    def prepare_sakura(self, models_dir: str):
        """Queue Sakura GGUF model."""
        target_dir = os.path.join(models_dir, "sakura")
        targets = [
            DownloadTarget(
                SAKURA_GGUF,
                os.path.join(target_dir, "sakura-14b-qwen3-v1.5-q6k.gguf"),
                "Downloading Sakura 14B Q6k (Subject to network speed)..."
            )
        ]
        self.queue_targets(targets)

    def prepare_qwen(self, models_dir: str):
        """Queue Qwen GGUF model."""
        target_dir = os.path.join(models_dir, "qwen")
        targets = [
            DownloadTarget(
                QWEN_GGUF,
                os.path.join(target_dir, "Qwen3-14B-Q6_K.gguf"),
                "Downloading Qwen 14B Q6k (Subject to network speed)..."
            )
        ]
        self.queue_targets(targets)

    def prepare_manga_ocr(self, models_dir: str):
        """Queue MangaOCR models."""
        target_dir = os.path.join(models_dir, "manga-ocr")
        targets = []
        for filename in MANGA_OCR_FILES:
            targets.append(
                DownloadTarget(
                    url=MANGA_OCR_BASE_URL + filename,
                    save_path=os.path.join(target_dir, filename),
                    label=f"Downloading MangaOCR: {filename}"
                )
            )
        self.queue_targets(targets)

    def prepare_big_lama(self, models_dir: str):
        """Queue fixed cleanup-owned iopaint LaMa model."""
        target_dir = os.path.join(models_dir, "inpaint", "iopaint")
        repo_url = f"https://huggingface.co/{CLEANUP_INPAINT_REPO_ID}/resolve/main"
        targets = [
            DownloadTarget(
                f"{repo_url}/{CLEANUP_INPAINT_MODEL_FILE}",
                os.path.join(target_dir, CLEANUP_INPAINT_MODEL_FILE),
                "Downloading fixed cleanup inpainting model..."
            )
        ]
        self.queue_targets(targets)

    def _verify_checksum(self, path: str, expected_sha256: str) -> bool:
        """Verify file checksum."""
        sha256 = hashlib.sha256()
        try:
            with open(path, 'rb') as f:
                while True:
                    data = f.read(65536)
                    if not data:
                        break
                    sha256.update(data)
            return sha256.hexdigest().lower() == expected_sha256.lower()
        except Exception:
            return False

    def _extract_archive(self, archive_path: str):
        """Extract .tar or .zip archives."""
        directory = os.path.dirname(archive_path)
        try:
            if archive_path.endswith(".tar"):
                with tarfile.open(archive_path, "r") as tar:
                    safe_members = []
                    for member in tar.getmembers():
                        # Block links and path traversal.
                        if member.issym() or member.islnk():
                            raise RuntimeError(f"Unsafe archive entry (link): {member.name}")
                        _safe_extract_path(directory, member.name)
                        safe_members.append(member)
                    tar.extractall(path=directory, members=safe_members)
            elif archive_path.endswith(".zip"):
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    for member in zip_ref.infolist():
                        _safe_extract_path(directory, member.filename)
                        zip_ref.extract(member, directory)
            
            # Remove archive after extraction
            os.remove(archive_path)
            self.status_changed.emit(f"Extracted {os.path.basename(archive_path)}")
            return True
            
        except Exception as e:
            self.status_changed.emit(f"Extraction failed: {e}")
            return False

    def _download_file(self, target: DownloadTarget) -> bool:
        """Helper to download a single file with progress."""
        if self._cancel_requested:
            self.finished.emit(False, "Cancelled")
            return False

        self.status_changed.emit(target.label)
        try:
            # Connect timeout: 10s, Read timeout: 120s (tolerates slow streams)
            with self._session.get(target.url, stream=True, timeout=(10, 120)) as r:
                r.raise_for_status()
                total_header = r.headers.get("content-length")
                total_length = None
                if total_header:
                    try:
                        total_length = int(total_header)
                    except (TypeError, ValueError):
                        total_length = None

                dl = 0
                last_percent = -1
                with open(target.save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if self._cancel_requested:
                            f.close()
                            if os.path.exists(target.save_path):
                                os.remove(target.save_path)
                            self.finished.emit(False, "Cancelled")
                            return False
                        if not chunk:
                            continue
                        dl += len(chunk)
                        f.write(chunk)
                        if total_length and total_length > 0:
                            percent = int(100 * dl / total_length)
                            if percent > last_percent:
                                self.progress_changed.emit(percent)
                                last_percent = percent
                if total_length is None:
                    self.progress_changed.emit(100)

            # Post-download verification
            if not target.sha256:
                self.status_changed.emit(
                    f"Checksum not provided for {os.path.basename(target.save_path)}; integrity not fully verified."
                )
            if target.sha256 and not self._verify_checksum(target.save_path, target.sha256):
                 self.finished.emit(False, "Download failed: Checksum mismatch.")
                 return False

            # Post-download extraction
            if target.save_path.endswith(".tar") or target.save_path.endswith(".zip"):
                self.status_changed.emit("Extracting archive...")
                if not self._extract_archive(target.save_path):
                    self.finished.emit(False, "Download failed: Archive extraction error.")
                    return False

            return True
        except Exception as e:
            self.finished.emit(False, f"Download failed: {str(e)}")
            return False


def _safe_extract_path(base_dir: str, member_name: str) -> str:
    """Return validated extraction path; raise on traversal/absolute paths."""
    if not member_name:
        raise RuntimeError("Unsafe archive entry: empty filename")
    normalized = member_name.replace("\\", "/")
    if normalized.startswith("/") or normalized.startswith("../") or "/../" in normalized:
        raise RuntimeError(f"Unsafe archive entry: {member_name}")
    target_path = os.path.abspath(os.path.join(base_dir, member_name))
    base_abs = os.path.abspath(base_dir)
    try:
        inside = os.path.commonpath([base_abs, target_path]) == base_abs
    except ValueError:
        inside = False
    if not inside:
        raise RuntimeError(f"Unsafe archive entry: {member_name}")
    return target_path
