"""PySide6 desktop app. Optional: the CLI does everything the GUI does."""

from __future__ import annotations

import sys
import threading
import traceback
from pathlib import Path

try:
    from PySide6.QtCore import (
        QEasingCurve,
        QItemSelectionModel,
        QObject,
        QPropertyAnimation,
        QSize,
        Qt,
        QThread,
        QTimer,
        QUrl,
        Signal,
    )
    from PySide6.QtGui import QDesktopServices, QFont, QIcon, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGraphicsOpacityEffect,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListView,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSpinBox,
        QStackedWidget,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError as err:  # pragma: no cover
    raise ImportError(
        "The desktop UI needs PySide6. Install it with:  pip install PySide6"
    ) from err

from .captions import ANIMATIONS, FONTS
from .config import Config
from .pipeline import RunResult, recaption, run_pipeline
from .util import ClipForgeError, config_dir, fmt_ts, show_error_dialog

APP_NAME = "Cilos"
APP_TITLE = "Cilos: Auto-Captioned Shorts Maker"
VIDEO_FILTER = "Video (*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.wmv *.flv);;All files (*)"

THUMB_W, THUMB_H = 150, 267

def _load_app_icon() -> QIcon:
    """Load the official ClipForge/Cilos icon from the project's icon folder."""
    icon_dir = Path(__file__).resolve().parent.parent / "icon"

    # Try the most likely official icon filenames first.
    preferred = [
        "clipforge.ico",
        "cilos.ico",
        "icon.ico",
        "clipforge.png",
        "cilos.png",
        "icon.png",
    ]

    for filename in preferred:
        icon_path = icon_dir / filename
        if icon_path.is_file():
            icon = QIcon(str(icon_path))
            if not icon.isNull():
                return icon

    # If the filename is different, find an ICO or PNG automatically.
    if icon_dir.is_dir():
        for pattern in ("*.ico", "*.png"):
            for icon_path in icon_dir.glob(pattern):
                icon = QIcon(str(icon_path))
                if not icon.isNull():
                    return icon

    return QIcon()

# Display names for the internal keys in captions.py.
ANIM_LABELS = {
    "word_pop": "Word pop",
    "typewriter": "Typewriter",
    "karaoke": "Karaoke sweep",
}
FONT_LABELS = {
    "montserrat": "Montserrat",
    "bebas": "Bebas Neue",
    "bangers": "Bangers",
}

# Item data role holding (animation, font) for each clip in the results list.
STYLE_ROLE = int(Qt.ItemDataRole.UserRole) + 1

STYLE = """
QMainWindow, QWidget#root { background: #101211; }
QWidget { color: #F4F5F0; font-size: 13px; }
QLabel { background: transparent; }
QCheckBox { background: transparent; spacing: 8px; }
QCheckBox::indicator {
    width: 16px; height: 16px; border-radius: 4px;
    background: #191E1B; border: 1px solid #343D36;
}
QCheckBox::indicator:checked { background: #B7F34A; border-color: #B7F34A; }

QFrame#card {
    background: #191E1B; border: 1px solid #29312B; border-radius: 12px;
}
QLabel#cardTitle { font-size: 14px; font-weight: 600; color: #F4F5F0; }
QLabel#title { font-size: 34px; font-weight: 800; letter-spacing: -0.5px; color: #F4F5F0; }
QLabel#subtitle { color: #858D87; font-size: 13px; }
QLabel#hint { color: #858D87; font-size: 12px; }
QLabel#section { color: #B7F34A; font-size: 12px; font-weight: 600; margin-top: 6px; }
QLabel#status { color: #A7B0A8; }
QLabel#emptyTitle { font-size: 18px; font-weight: 700; color: #F4F5F0; }
QLabel#emptyHint { color: #858D87; font-size: 13px; }
QToolButton#advancedToggle {
    color: #B7F34A; background: #151A16; border: 1px solid #29312B;
    border-radius: 8px; padding: 9px 10px; font-weight: 600; text-align: left;
}
QToolButton#advancedToggle:hover { background: #202722; border-color: #3A473C; }
QToolButton#advancedToggle { padding: 6px 10px; color: #A7B0A8; }

QLineEdit, QSpinBox, QComboBox, QPlainTextEdit, QListWidget {
    background: #0C0F0D; border: 1px solid #29312B; border-radius: 8px;
    padding: 7px 10px; selection-background-color: #B7F34A;
    selection-color: #101211;
}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {
    border-color: #B7F34A;
}
QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled {
    color: #8B948D; background: #141916; border-color: #303832;
}
QCheckBox:disabled { color: #8B948D; }
QPlainTextEdit:disabled { color: #8B948D; background: #141916; }
QLabel#hint:disabled, QLabel#subtitle:disabled { color: #747D76; }
QComboBox QAbstractItemView {
    background: #191E1B; border: 1px solid #29312B;
    selection-background-color: #B7F34A; selection-color: #101211;
}

QFrame#segment { background: #0C0F0D; border: 1px solid #29312B; border-radius: 10px; }
QPushButton#seg {
    background: transparent; border: none; border-radius: 7px;
    padding: 7px 16px; color: #858D87; font-weight: 600;
}
QPushButton#seg:hover { color: #F4F5F0; background: #202722; }
QPushButton#seg:checked { background: #B7F34A; color: #101211; }
QPushButton#seg:disabled { color: #747D76; }
QPushButton#seg:checked:disabled { background: #536B2D; color: #D5E8B4; }

QPushButton {
    background: #252C27; border: 1px solid #343D36; border-radius: 8px;
    padding: 9px 16px; font-weight: 600; color: #F4F5F0;
}
QPushButton:hover { background: #303A33; border-color: #48554A; }
QPushButton:disabled { color: #8B948D; background: #171C18; border-color: #303832; }
QPushButton#primary { background: #B7F34A; border-color: #B7F34A; color: #101211; }
QPushButton#primary:hover { background: #C8FF62; border-color: #C8FF62; }
QPushButton#primary:disabled { background: #536B2D; border-color: #536B2D; color: #D5E8B4; }

QProgressBar {
    background: #0C0F0D; border: 1px solid #29312B; border-radius: 6px;
    height: 8px; text-align: center;
}
QProgressBar::chunk { background: #B7F34A; border-radius: 5px; }

QListWidget::item {
    padding: 8px; border: 1px solid #29312B; border-radius: 10px; background: #151A16;
}
QListWidget::item:hover { background: #1D251E; border-color: #3A473C; }
QListWidget::item:selected { border-color: #B7F34A; background: #222D1D; }
"""


def _cuda_device_count() -> int:
    """How many CUDA GPUs ctranslate2 (used by faster-whisper) can see."""
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def _anim_combo(current: str | None) -> QComboBox:
    box = QComboBox()
    for key in ANIMATIONS:
        box.addItem(ANIM_LABELS.get(key, key), key)
    idx = box.findData(current)
    box.setCurrentIndex(idx if idx >= 0 else 0)
    return box


def _font_combo(current: str | None) -> QComboBox:
    box = QComboBox()
    for key in FONTS:
        box.addItem(FONT_LABELS.get(key, key), key)
    idx = box.findData(current)
    box.setCurrentIndex(idx if idx >= 0 else 0)
    return box


def _style_line(animation: str, font: str) -> str:
    if not animation:
        return "No captions"
    return f"{ANIM_LABELS.get(animation, animation)} · {FONT_LABELS.get(font, font)}"


def _thumb_icon(path: Path | None) -> QIcon | None:
    if not path or not path.exists():
        return None
    pixmap = QPixmap(str(path))
    if pixmap.isNull():
        return None
    scaled = pixmap.scaled(
        THUMB_W,
        THUMB_H,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    return QIcon(scaled)


class ClipCard(QFrame):
    """Polished clip result card with independent thumbnail and metadata."""

    def __init__(self, thumbnail: Path | None, title: str, start: float, end: float,
                 duration: float, score: float, style_text: str, selected: bool = False,
                 on_click=None, on_double_click=None):
        super().__init__()
        self.on_click = on_click
        self.on_double_click = on_double_click
        self._selected = selected
        self.setObjectName("clipCard")
        self.setFixedWidth(206)
        self.setMouseTracking(True)
        self.setStyleSheet(self._card_style(selected))
        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity)
        self._press_anim = QPropertyAnimation(self._opacity, b"opacity", self)
        self._press_anim.setDuration(115)
        self._press_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 10)
        outer.setSpacing(8)

        self.preview = QLabel()
        self.preview.setObjectName("clipPreview")
        self.preview.setFixedSize(190, 338)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#0C0F0D; border:0; border-radius:7px; color:#6E786F;")
        self._set_thumbnail(thumbnail)
        outer.addWidget(self.preview)

        title_label = QLabel(title.strip() or "Untitled clip")
        title_label.setObjectName("clipTitle")
        title_label.setWordWrap(True)
        title_label.setMaximumHeight(40)
        title_label.setToolTip(title)
        outer.addWidget(title_label)

        meta = QHBoxLayout()
        meta.setSpacing(4)
        timing = QLabel(f"{fmt_ts(start)}–{fmt_ts(end)}")
        timing.setObjectName("clipMeta")
        meta.addWidget(timing, 1)
        score_badge = QLabel(f"★ {score:.0f}")
        score_badge.setObjectName("clipScore")
        score_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_badge.setFixedWidth(48)
        meta.addWidget(score_badge)
        outer.addLayout(meta)

        second = QHBoxLayout()
        duration_label = QLabel(f"{duration:.0f}s")
        duration_label.setObjectName("clipMeta")
        self.style_label = QLabel(style_text)
        self.style_label.setObjectName("clipStyle")
        self.style_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.style_label.setToolTip(style_text)
        second.addWidget(duration_label)
        second.addStretch(1)
        second.addWidget(self.style_label)
        outer.addLayout(second)

    def _set_thumbnail(self, thumbnail: Path | None) -> None:
        pix = QPixmap(str(thumbnail)) if thumbnail and thumbnail.exists() else QPixmap()
        if pix.isNull():
            self.preview.setPixmap(QPixmap())
            self.preview.setText("VIDEO")
            return
        scaled = pix.scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                            Qt.TransformationMode.SmoothTransformation)
        x = max(0, (scaled.width() - self.preview.width()) // 2)
        y = max(0, (scaled.height() - self.preview.height()) // 2)
        self.preview.setPixmap(scaled.copy(x, y, self.preview.width(), self.preview.height()))

    def update_style(self, style_text: str, thumbnail: Path | None = None) -> None:
        """Refresh the style line and thumbnail after a recaption."""
        self.style_label.setText(style_text)
        self.style_label.setToolTip(style_text)
        if thumbnail is not None:
            self._set_thumbnail(thumbnail)

    @staticmethod
    def _card_style(selected: bool) -> str:
        border = "#B7F34A" if selected else "#29312B"
        bg = "#20271F" if selected else "#171C18"
        return f"""
            QFrame#clipCard {{ background:{bg}; border:1px solid {border}; border-radius:11px; }}
            QLabel#clipTitle {{ color:#F4F5F0; background:transparent; font-size:12px;
                font-weight:600; border:0; padding:0 2px; }}
            QLabel#clipMeta {{ color:#A7B0A8; background:transparent; font-size:10px; border:0; }}
            QLabel#clipScore {{ color:#101211; background:#B7F34A; border:0; border-radius:8px;
                font-size:10px; font-weight:700; padding:3px 5px; }}
            QLabel#clipStyle {{ color:#858D87; background:transparent; font-size:10px; border:0; }}
        """

    def set_selected(self, selected: bool):
        self._selected = selected
        self.setStyleSheet(self._card_style(selected))

    def _animate_press(self):
        # A quick tactile dip, then a smooth return to full opacity.
        self._press_anim.stop()
        self._press_anim.setStartValue(1.0)
        self._press_anim.setEndValue(0.88)
        self._press_anim.start()
        self._press_anim.finished.connect(self._release_press)

    def _release_press(self):
        try:
            self._press_anim.finished.disconnect(self._release_press)
        except RuntimeError:
            pass
        self._press_anim.stop()
        self._press_anim.setStartValue(0.88)
        self._press_anim.setEndValue(1.0)
        self._press_anim.start()

    def mousePressEvent(self, event):
        self._animate_press()
        if self.on_click:
            self.on_click(event)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.on_double_click:
            self.on_double_click()
        super().mouseDoubleClickEvent(event)


class Worker(QObject):
    progress = Signal(float, str)
    log = Signal(str)
    clip_ready = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, video: Path, cfg: Config):
        super().__init__()
        self.video = video
        self.cfg = cfg
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            result = run_pipeline(
                self.video,
                self.cfg,
                progress=lambda p, m: self.progress.emit(float(p), str(m)),
                log=lambda m: self.log.emit(str(m)),
                cancel=self._cancel.is_set,
                on_clip=lambda clip: self.clip_ready.emit(clip),
            )
            self.finished.emit(result)
        except ClipForgeError as err:
            self.failed.emit(str(err))
        except Exception:
            self.failed.emit(traceback.format_exc(limit=3))


class RecaptionWorker(QObject):
    """Re-burns captions on finished clips, one at a time, off the UI thread."""

    log = Signal(str)
    clip_done = Signal(object)   # (file name, animation, font)
    finished = Signal(object)    # list of file names that were done
    failed = Signal(str)

    def __init__(self, output_dir: Path, names: list[str], cfg: Config,
                 animation: str, font: str):
        super().__init__()
        self.output_dir = output_dir
        self.names = names
        self.cfg = cfg
        self.animation = animation
        self.font = font
        self._cancel = threading.Event()

    def cancel(self) -> None:
        # Checked between clips. The ffmpeg call in progress is allowed to finish.
        self._cancel.set()

    def run(self) -> None:
        done: list[str] = []
        try:
            for name in self.names:
                if self._cancel.is_set():
                    break
                recaption(
                    self.output_dir,
                    name,
                    self.cfg,
                    animation=self.animation,
                    font=self.font,
                    log=lambda m: self.log.emit(str(m)),
                )
                done.append(name)
                self.clip_done.emit((name, self.animation, self.font))
            self.finished.emit(done)
        except ClipForgeError as err:
            self.failed.emit(str(err))
        except Exception:
            self.failed.emit(traceback.format_exc(limit=3))


class MainWindow(QMainWindow):
    def __init__(self, cfg: Config, initial: Path | None = None):
        super().__init__()
        self.cfg = cfg
        self.video: Path | None = None
        self.result: RunResult | None = None
        self.output_dir: Path | None = None
        self.thread: QThread | None = None
        self.worker: Worker | RecaptionWorker | None = None
        self._cuda_count = _cuda_device_count()
        self._syncing_durations = False

        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(1080, 760)
        self.setAcceptDrops(True)
        self._build_ui()
        if initial:
            self._set_video(Path(initial))

    # ------------------------------------------------------------------ #
    # Building the UI
    # ------------------------------------------------------------------ #
    def _card(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName("card")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(12)
        if title:
            label = QLabel(title)
            label.setObjectName("cardTitle")
            lay.addWidget(label)
        return frame, lay

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(26, 22, 26, 22)
        layout.setSpacing(16)

        # --- header: title on the left, GPU/CPU switch on the right ----- #
        header = QHBoxLayout()
        header.setSpacing(12)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        title = QLabel(APP_NAME)
        title.setObjectName("title")
        subtitle = QLabel("Long video in. Captioned vertical clips out. Everything stays on this PC.")
        subtitle.setObjectName("subtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles, 1)
        header.addWidget(self._build_device_switch(), 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(header)

        # --- source ------------------------------------------------------ #
        source_row = QHBoxLayout()
        source_row.setSpacing(8)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Drop a video here, or choose one...")
        self.path_edit.setReadOnly(True)
        self.browse_btn = QPushButton("Choose video")
        self.browse_btn.clicked.connect(self._browse)
        source_row.addWidget(self.path_edit, 1)
        source_row.addWidget(self.browse_btn)
        layout.addLayout(source_row)

        # --- body: controls on the left, results and log on the right --- #
        body = QHBoxLayout()
        body.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(16)
        left.addWidget(self._build_settings_card())
        left.addWidget(self._build_run_card())
        left.addStretch(1)
        left_wrap = QWidget()
        left_wrap.setLayout(left)
        left_wrap.setFixedWidth(400)
        body.addWidget(left_wrap, 0)

        right = QVBoxLayout()
        right.setSpacing(16)

        clips_card, clips_lay = self._card("Your clips")
        self.clips_stack = QStackedWidget()

        empty_state = QWidget()
        empty_layout = QVBoxLayout(empty_state)
        empty_layout.setContentsMargins(24, 32, 24, 32)
        empty_layout.setSpacing(10)
        empty_layout.addStretch(1)
        empty_icon = QLabel("✦")
        empty_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_icon.setStyleSheet("color: #B7F34A; font-size: 42px; font-weight: 700;")
        empty_title = QLabel("Your best moments will live here")
        empty_title.setObjectName("emptyTitle")
        empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_hint = QLabel("Choose a video, set your preferences, then generate clips.\nYour results will appear as visual cards here.")
        empty_hint.setObjectName("emptyHint")
        empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_hint.setWordWrap(True)
        empty_layout.addWidget(empty_icon)
        empty_layout.addWidget(empty_title)
        empty_layout.addWidget(empty_hint)
        empty_layout.addStretch(1)

        self.results = QListWidget()
        self.results.setViewMode(QListView.ViewMode.IconMode)
        self.results.setResizeMode(QListView.ResizeMode.Adjust)
        self.results.setMovement(QListView.Movement.Static)
        self.results.setIconSize(QSize(THUMB_W, THUMB_H))
        self.results.setGridSize(QSize(222, 438))
        self.results.setSpacing(14)
        self.results.setWordWrap(False)
        self.results.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.results.setStyleSheet(
            "QListWidget { background:#0C0F0D; border:1px solid #29312B; "
            "border-radius:9px; padding:8px; } "
            "QListWidget::item { background:transparent; border:0; padding:0; } "
            "QListWidget::item:selected { background:transparent; border:0; }"
        )
        self.results.itemDoubleClicked.connect(self._open_clip)
        self.results.itemSelectionChanged.connect(self._on_selection_changed)
        self.clips_stack.addWidget(empty_state)
        self.clips_stack.addWidget(self.results)
        self.clips_stack.setCurrentIndex(0)
        clips_lay.addWidget(self.clips_stack, 1)

        log_card, log_lay = self._card("Activity · Troubleshooting")
        self.log_toggle = QToolButton()
        self.log_toggle.setObjectName("advancedToggle")
        self.log_toggle.setText("▸ Show technical log")
        self.log_toggle.setCheckable(True)
        self.log_toggle.setChecked(False)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(150)
        self.console.setFont(QFont("Consolas" if sys.platform == "win32" else "monospace", 9))
        self.console.setVisible(False)
        self.log_toggle.toggled.connect(self._toggle_technical_log)
        log_lay.addWidget(self.log_toggle)
        log_lay.addWidget(self.console, 1)

        right.addWidget(clips_card, 5)
        self.caption_card = self._build_caption_card()
        self.caption_card.setVisible(False)  # Caption editing is useful after clips exist.
        right.addWidget(self.caption_card, 0)
        right.addWidget(log_card, 0)
        body.addLayout(right, 1)
        layout.addLayout(body, 1)

        self.setCentralWidget(root)
        self.setStyleSheet(STYLE)
        self._sync_durations()
        self._update_device_hint()
        self._sync_caption_controls(busy=False)

    def _update_reframe_hint(self) -> None:
        mode = self.reframe_box.currentData()
        if mode == "face":
            self.reframe_hint.setText(
                "Tracks detected faces to keep the subject framed in vertical video. "
                "Recommended for interviews and talking-head footage."
            )
        else:
            self.reframe_hint.setText(
                "Uses a fixed center crop. Best when the subject stays centered or "
                "when you want stable framing without tracking."
            )

    def _toggle_technical_log(self, expanded: bool) -> None:
        self.console.setVisible(expanded)
        self.log_toggle.setText("▾ Hide technical log" if expanded else "▸ Show technical log")

    def _build_device_switch(self) -> QWidget:
        holder = QWidget()
        col = QVBoxLayout(holder)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)

        seg = QFrame()
        seg.setObjectName("segment")
        row = QHBoxLayout(seg)
        row.setContentsMargins(4, 4, 4, 4)
        row.setSpacing(2)

        self.device_group = QButtonGroup(self)
        self.device_group.setExclusive(True)
        options = [
            ("Auto", "auto", "Use the GPU when CUDA is available, otherwise the CPU"),
            ("GPU", "cuda", "Always use the NVIDIA GPU (fastest)"),
            ("CPU", "cpu", "Always use the CPU (slower, no CUDA setup needed)"),
        ]
        current = (self.cfg.whisper_device or "auto").lower()
        if current not in {v for _, v, _ in options}:
            current = "auto"

        for label, value, tip in options:
            btn = QPushButton(label)
            btn.setObjectName("seg")
            btn.setCheckable(True)
            btn.setToolTip(tip)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setChecked(value == current)
            self.device_group.addButton(btn)
            btn.clicked.connect(lambda _checked=False, v=value: self._set_device(v))
            row.addWidget(btn)

        self.device_hint = QLabel()
        self.device_hint.setObjectName("hint")
        self.device_hint.setAlignment(Qt.AlignmentFlag.AlignRight)

        col.addWidget(seg, 0, Qt.AlignmentFlag.AlignRight)
        col.addWidget(self.device_hint, 0, Qt.AlignmentFlag.AlignRight)
        return holder

    def _build_settings_card(self) -> QFrame:
        card, lay = self._card("Settings")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        lay.addLayout(form)

        self.count = QSpinBox()
        self.count.setRange(1, 30)
        self.count.setValue(self.cfg.clip_count)

        self.min_dur = QSpinBox()
        self.min_dur.setRange(5, 180)
        self.min_dur.setValue(int(self.cfg.min_duration))
        self.min_dur.setSuffix(" s min")
        self.max_dur = QSpinBox()
        self.max_dur.setRange(10, 300)
        self.max_dur.setValue(int(self.cfg.max_duration))
        self.max_dur.setSuffix(" s max")
        self.min_dur.valueChanged.connect(self._sync_durations)
        self.max_dur.valueChanged.connect(self._sync_durations)
        dur_row = QHBoxLayout()
        dur_row.setSpacing(8)
        dur_row.addWidget(self.min_dur)
        dur_row.addWidget(self.max_dur)

        self.layout_box = QComboBox()
        self.layout_box.addItems(["blur", "crop", "pad"])
        self.layout_box.setCurrentText(self.cfg.layout)

        self.reframe_box = QComboBox()
        self.reframe_box.addItem("Smart face tracking", "face")
        self.reframe_box.addItem("Center crop", "center")
        reframe_idx = self.reframe_box.findData(self.cfg.reframe)
        self.reframe_box.setCurrentIndex(reframe_idx if reframe_idx >= 0 else 0)
        self.reframe_box.setToolTip(
            "Smart face tracking keeps a detected face in the vertical frame. "
            "Center crop uses a fixed center position."
        )
        self.reframe_hint = QLabel()
        self.reframe_hint.setObjectName("hint")
        self.reframe_hint.setWordWrap(True)
        self.reframe_box.currentIndexChanged.connect(self._update_reframe_hint)
        self._update_reframe_hint()

        self.captions = QCheckBox("Burn in captions")
        self.captions.setChecked(self.cfg.captions)
        self.captions.toggled.connect(lambda _on: self._sync_caption_controls(self.thread is not None))

        self.job_anim = _anim_combo(getattr(self.cfg, "caption_animation", None))
        self.job_font = _font_combo(getattr(self.cfg, "caption_font", None))

        form.addRow("Clips to make", self.count)
        form.addRow("Clip length", dur_row)
        form.addRow("Vertical fill", self.layout_box)
        form.addRow("Reframe", self.reframe_box)
        form.addRow("", self.reframe_hint)
        form.addRow("", self.captions)
        form.addRow("Caption style", self.job_anim)
        form.addRow("Caption font", self.job_font)

        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setObjectName("advancedToggle")
        self.advanced_toggle.setText("▸  Advanced · AI scoring")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setChecked(False)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        lay.addWidget(self.advanced_toggle)

        self.advanced_panel = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_panel)
        advanced_layout.setContentsMargins(4, 2, 4, 2)
        advanced_layout.setSpacing(8)
        ai_form = QFormLayout()
        ai_form.setHorizontalSpacing(12)
        ai_form.setVerticalSpacing(8)
        self.llm_base = QLineEdit(self.cfg.llm_base_url)
        self.llm_base.setPlaceholderText("Empty = built-in scorer")
        self.llm_model = QLineEdit(self.cfg.llm_model)
        self.llm_model.setPlaceholderText("gpt-4o-mini / llama3.1 / ...")
        self.llm_key = QLineEdit(self.cfg.llm_api_key)
        self.llm_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.llm_key.setPlaceholderText("Only if your endpoint needs one")
        ai_form.addRow("Base URL", self.llm_base)
        ai_form.addRow("Model", self.llm_model)
        ai_form.addRow("API key", self.llm_key)
        advanced_layout.addLayout(ai_form)
        self.advanced_panel.setVisible(False)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        lay.addWidget(self.advanced_panel)
        return card

    def _toggle_advanced(self, expanded: bool) -> None:
        self.advanced_panel.setVisible(expanded)
        self.advanced_toggle.setText("▾  Advanced · AI scoring" if expanded else "▸  Advanced · AI scoring")

    def _build_caption_card(self) -> QFrame:
        card, lay = self._card("Captions")
        hint = QLabel("Select one or more clips, pick a style and font, then apply. "
                      "The original video is kept, so you can change it again.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        form = QFormLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        self.panel_anim = _anim_combo(getattr(self.cfg, "caption_animation", None))
        self.panel_font = _font_combo(getattr(self.cfg, "caption_font", None))
        form.addRow("Style", self.panel_anim)
        form.addRow("Font", self.panel_font)
        lay.addLayout(form)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.apply_btn = QPushButton("Apply to selected")
        self.apply_btn.setObjectName("primary")
        self.apply_btn.clicked.connect(self._recaption_selected)
        self.apply_all_btn = QPushButton("Apply to all")
        self.apply_all_btn.clicked.connect(self._recaption_all)
        row.addWidget(self.apply_btn, 1)
        row.addWidget(self.apply_all_btn)
        lay.addLayout(row)
        return card

    def _build_run_card(self) -> QFrame:
        card, lay = self._card("")
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.start_btn = QPushButton("Make clips")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self._start)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        self.open_btn = QPushButton("Open folder")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_output)
        actions.addWidget(self.start_btn, 1)
        actions.addWidget(self.cancel_btn)
        actions.addWidget(self.open_btn)
        lay.addLayout(actions)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        lay.addWidget(self.progress)

        self.status = QLabel("Ready.")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        return card

    # ------------------------------------------------------------------ #
    # Keeping the settings valid
    # ------------------------------------------------------------------ #
    def _sync_durations(self) -> None:
        """Keep min and max clip length in a valid range, at least 5 s apart."""
        if self._syncing_durations:
            return
        self._syncing_durations = True
        try:
            self.max_dur.setMinimum(self.min_dur.value() + 5)
            # Re-read: setting the minimum may have raised max.
            self.min_dur.setMaximum(self.max_dur.value() - 5)
        finally:
            self._syncing_durations = False

    # ------------------------------------------------------------------ #
    # GPU / CPU switch
    # ------------------------------------------------------------------ #
    def _set_device(self, value: str) -> None:
        self.cfg.whisper_device = value
        self._update_device_hint()

    def _update_device_hint(self) -> None:
        value = (self.cfg.whisper_device or "auto").lower()
        has_gpu = self._cuda_count > 0
        if value == "cpu":
            text = "Using CPU. Slower, but no CUDA setup needed."
        elif value == "cuda":
            text = "Using GPU." if has_gpu else "No CUDA GPU found. Jobs will fail."
        else:
            text = "Auto: GPU detected." if has_gpu else "Auto: no GPU found, using CPU."
        self.device_hint.setText(text)

    # ------------------------------------------------------------------ #
    # Locking the controls while a job runs
    # ------------------------------------------------------------------ #
    def _input_widgets(self) -> list[QWidget]:
        return [
            self.browse_btn,
            *self.device_group.buttons(),
            self.count,
            self.min_dur,
            self.max_dur,
            self.layout_box,
            self.reframe_box,
            self.captions,
            self.advanced_toggle,
            self.llm_base,
            self.llm_model,
            self.llm_key,
        ]

    def _set_busy(self, busy: bool) -> None:
        for widget in self._input_widgets():
            widget.setEnabled(not busy)
        self.start_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        self._sync_caption_controls(busy)

    def _sync_caption_controls(self, busy: bool) -> None:
        """Enable the caption pickers and apply buttons to match the current state."""
        captions_on = self.captions.isChecked()
        self.job_anim.setEnabled(not busy and captions_on)
        self.job_font.setEnabled(not busy and captions_on)
        self.panel_anim.setEnabled(not busy)
        self.panel_font.setEnabled(not busy)
        self.apply_btn.setEnabled(not busy and bool(self.results.selectedItems()))
        self.apply_all_btn.setEnabled(not busy and self.results.count() > 0)

    # ------------------------------------------------------------------ #
    # Drag, drop, and file picking
    # ------------------------------------------------------------------ #
    def dragEnterEvent(self, event) -> None:
        if self.thread is None and event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        if self.thread is not None:
            return
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self._set_video(Path(url.toLocalFile()))
                break

    def _browse(self) -> None:
        if self.thread is not None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Choose a video", self._start_dir(), VIDEO_FILTER)
        if path:
            self._set_video(Path(path))

    def _start_dir(self) -> str:
        """Open the picker somewhere useful, not an empty Linux home folder."""
        if self.video and self.video.parent.exists():
            return str(self.video.parent)
        # Under WSL the user's real videos live on the Windows drive.
        users = Path("/mnt/c/Users")
        if users.is_dir():
            skip = {"public", "default", "default user", "all users"}
            real = [p for p in users.iterdir() if p.is_dir() and p.name.lower() not in skip]
            if len(real) == 1:
                for folder in ("Desktop", "Videos", "Downloads"):
                    candidate = real[0] / folder
                    if candidate.is_dir():
                        return str(candidate)
        return str(Path.home())

    def _set_video(self, path: Path) -> None:
        if not path.exists() or path.is_dir():
            return
        self.video = path
        self.path_edit.setText(str(path))
        self.status.setText(f"{path.name} ready.")

    # ------------------------------------------------------------------ #
    # Running a job
    # ------------------------------------------------------------------ #
    def _collect_config(self) -> Config:
        cfg = self.cfg.copy()
        cfg.clip_count = self.count.value()
        cfg.min_duration = float(self.min_dur.value())
        cfg.max_duration = float(self.max_dur.value())
        if cfg.max_duration <= cfg.min_duration:
            cfg.max_duration = cfg.min_duration + 5.0
        cfg.layout = self.layout_box.currentText()
        cfg.reframe = self.reframe_box.currentData()
        cfg.captions = self.captions.isChecked()
        cfg.caption_animation = self.job_anim.currentData()
        cfg.caption_font = self.job_font.currentData()
        cfg.llm_base_url = self.llm_base.text().strip()
        cfg.llm_model = self.llm_model.text().strip() or cfg.llm_model
        cfg.llm_api_key = self.llm_key.text().strip()
        # whisper_device was already set by the GPU/CPU switch.
        return cfg

    def _start(self) -> None:
        if self.thread is not None:
            return
        if not self.video:
            QMessageBox.information(self, "Pick a video", "Choose a video file first.")
            return
        if (self.cfg.whisper_device or "").lower() == "cuda" and self._cuda_count == 0:
            QMessageBox.warning(
                self,
                "No GPU found",
                "No CUDA GPU was detected. Switch to Auto or CPU, then try again.",
            )
            return

        cfg = self._collect_config()
        self.cfg = cfg
        self.result = None
        self.output_dir = None
        self.results.clear()
        self.clips_stack.setCurrentIndex(0)
        self.caption_card.setVisible(False)
        self.console.clear()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status.setText("Working...")
        self.open_btn.setEnabled(False)
        self._set_busy(True)

        self.thread = QThread(self)
        self.worker = Worker(self.video, cfg)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._on_log)
        self.worker.clip_ready.connect(self._on_clip_ready)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.thread.start()

    def _cancel(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.status.setText("Cancelling...")
            self.cancel_btn.setEnabled(False)

    def _on_progress(self, pct: float, message: str) -> None:
        self.progress.setValue(int(pct))
        self.status.setText(message)

    def _on_log(self, message: str) -> None:
        self.console.appendPlainText(message)

    def _on_clip_ready(self, clip) -> None:
        c = clip.candidate
        self.output_dir = clip.path.parent
        anim = getattr(clip, "caption_animation", "") or ""
        font = getattr(clip, "caption_font", "") or ""
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, str(clip.path))
        item.setData(STYLE_ROLE, (anim, font))
        item.setToolTip(c.hook or c.reason)
        item.setSizeHint(QSize(222, 438))
        self.results.addItem(item)

        card = ClipCard(
            clip.thumbnail, c.title, c.start, c.end, c.duration, c.score,
            _style_line(anim, font), selected=False,
            on_click=lambda event, item=item: self._select_clip_card(item, event),
            on_double_click=lambda path=clip.path: QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(path))
            ),
        )
        self.results.setItemWidget(item, card)
        self.caption_card.setVisible(True)
        self.clips_stack.setCurrentWidget(self.results)
        self.results.scrollToItem(item)
        self.open_btn.setEnabled(True)
        self.status.setText(f"{self.results.count()} clip(s) ready so far...")
        self._sync_caption_controls(busy=self.thread is not None)

    def _on_finished(self, result: RunResult) -> None:
        self._teardown()
        self.result = result
        self.output_dir = result.output_dir
        self.progress.setValue(100)
        self.open_btn.setEnabled(bool(result.clips))
        if result.clips:
            self.status.setText(f"{len(result.clips)} clip(s) in {result.output_dir}")
        else:
            self.status.setText("No clips were produced. Check the activity log.")

    def _on_failed(self, message: str) -> None:
        self._teardown()
        self.progress.setValue(0)
        self.status.setText("Failed.")
        self.console.appendPlainText("\n" + message)
        QMessageBox.critical(self, APP_NAME, message)

    def _teardown(self) -> None:
        if self.thread:
            self.thread.quit()
            self.thread.wait(3000)
            self.thread = None
            self.worker = None
        self._set_busy(False)

    # ------------------------------------------------------------------ #
    # Restyling finished clips
    # ------------------------------------------------------------------ #
    def _selected_names(self) -> list[str]:
        names = []
        for item in self.results.selectedItems():
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                names.append(Path(path).name)
        return names

    def _all_names(self) -> list[str]:
        names = []
        for i in range(self.results.count()):
            path = self.results.item(i).data(Qt.ItemDataRole.UserRole)
            if path:
                names.append(Path(path).name)
        return names

    def _item_for_name(self, name: str) -> QListWidgetItem | None:
        for i in range(self.results.count()):
            item = self.results.item(i)
            if Path(item.data(Qt.ItemDataRole.UserRole)).name == name:
                return item
        return None

    def _select_clip_card(self, item: QListWidgetItem, event) -> None:
        flags = QItemSelectionModel.SelectionFlag.ClearAndSelect
        modifiers = event.modifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            flags = QItemSelectionModel.SelectionFlag.Toggle
        elif modifiers & Qt.KeyboardModifier.ShiftModifier:
            flags = QItemSelectionModel.SelectionFlag.Select
        self.results.setCurrentItem(item, flags)

    def _on_selection_changed(self) -> None:
        selected = set(self.results.selectedItems())
        for i in range(self.results.count()):
            item = self.results.item(i)
            card = self.results.itemWidget(item)
            if isinstance(card, ClipCard):
                card.set_selected(item in selected)
        items = self.results.selectedItems()
        if len(items) == 1:
            # Show the selected clip's current style in the panel, so the
            # pickers start from what the clip actually has.
            data = items[0].data(STYLE_ROLE)
            if data:
                anim, font = data[0], data[1]
                idx = self.panel_anim.findData(anim)
                if idx >= 0:
                    self.panel_anim.setCurrentIndex(idx)
                idx = self.panel_font.findData(font)
                if idx >= 0:
                    self.panel_font.setCurrentIndex(idx)
        self._sync_caption_controls(busy=self.thread is not None)

    def _recaption_selected(self) -> None:
        self._recaption(self._selected_names())

    def _recaption_all(self) -> None:
        self._recaption(self._all_names())

    def _recaption(self, names: list[str]) -> None:
        if self.thread is not None or not names or self.output_dir is None:
            return

        animation = self.panel_anim.currentData()
        font = self.panel_font.currentData()
        cfg = self.cfg.copy()

        self.status.setText(f"Recaptioning {len(names)} clip(s)...")
        self.progress.setRange(0, 0)  # busy indicator: no per-clip progress yet
        self._set_busy(True)

        self.thread = QThread(self)
        self.worker = RecaptionWorker(self.output_dir, names, cfg, animation, font)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self._on_log)
        self.worker.clip_done.connect(self._on_clip_recaptioned)
        self.worker.finished.connect(self._on_recaption_finished)
        self.worker.failed.connect(self._on_recaption_failed)
        self.thread.start()

    def _on_clip_recaptioned(self, data) -> None:
        name, anim, font = data
        item = self._item_for_name(name)
        if item is None:
            return
        item.setData(STYLE_ROLE, (anim, font))
        # The thumbnail is named after the clip's index, e.g. "01-thumb.jpg".
        thumb = self.output_dir / f"{name.split('-', 1)[0]}-thumb.jpg"
        card = self.results.itemWidget(item)
        if isinstance(card, ClipCard):
            card.update_style(_style_line(anim, font), thumb)

    def _on_recaption_finished(self, done) -> None:
        self._teardown()
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.status.setText(f"Captions updated on {len(done)} clip(s).")

    def _on_recaption_failed(self, message: str) -> None:
        self._teardown()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status.setText("Caption update failed.")
        self.console.appendPlainText("\n" + message)
        QMessageBox.critical(self, APP_NAME, message)

    # ------------------------------------------------------------------ #
    def _open_output(self) -> None:
        target = self.output_dir
        if target and Path(target).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _open_clip(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def closeEvent(self, event) -> None:
        if self.thread is None:
            event.accept()
            return

        answer = QMessageBox.question(
            self,
            "Job running",
            "A job is still running. Cancel it and close?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            event.ignore()
            return

        if self.worker:
            self.worker.cancel()
        self.status.setText("Stopping...")

        # Let the worker finish its current step. Keep the event loop running
        # so its finished/failed signals reach _teardown.
        waited_ms = 0
        while self.thread is not None and self.thread.isRunning() and waited_ms < 30000:
            QApplication.processEvents()
            self.thread.wait(200)
            waited_ms += 200

        if self.thread is not None and self.thread.isRunning():
            QMessageBox.warning(
                self,
                "Still stopping",
                "The job hasn't stopped yet. Give it a moment, then close again.",
            )
            event.ignore()
            return

        self._teardown()
        event.accept()


def launch(initial: Path | None, cfg: Config) -> int:
    try:
        app = QApplication.instance() or QApplication(sys.argv)

        # Load the official application icon.
        app_icon = _load_app_icon()
        if not app_icon.isNull():
            app.setWindowIcon(app_icon)

        app.setApplicationName(APP_NAME)
        window = MainWindow(cfg, initial)
        window.showMaximized()
        return app.exec()
    except Exception:
        details = traceback.format_exc()
        try:
            log_path = config_dir() / "error.log"
            log_path.write_text(details, encoding="utf-8")
            where = f"\n\nFull details: {log_path}"
        except OSError:
            where = ""
        show_error_dialog(f"{APP_NAME} could not start", details[-1200:] + where)
        return 1
