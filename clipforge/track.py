"""Face tracking for the vertical crop.

Detection uses MediaPipe Face Detection, a neural network that handles
side-on and smaller faces far better than Haar cascades. Two MediaPipe
flavours are supported:

  * the classic ``mp.solutions`` API (older releases), and
  * the Tasks API (current releases, where ``mp.solutions`` no longer exists).
    It needs a small model file, which is downloaded once into the config
    folder. Set CILOS_FACE_MODEL to use a file you already have.

If MediaPipe cannot be started for any reason, it falls back to the OpenCV
Haar tracker, and says why (see ``backend_name`` and ``backend_note``). If
OpenCV is missing entirely, everything returns None and the renderer uses a
centred crop.

Pipeline:
  1. Sample the clip several times per second and spot hard shot cuts.
  2. Detect faces in each sampled frame and stay on one subject per shot.
  3. Fill detection gaps, drop outliers and smooth. This runs offline on the
     whole clip, so the camera never lags behind the subject.
  4. Dead zone sized to the crop window: the camera sits still while the
     subject stays near the middle of the crop, and only moves when they drift.
  5. Snap on shot cuts instead of panning across them.
  6. Reduce to a few keyframes that the renderer interpolates linearly.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import statistics
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .util import config_dir, run

try:  # pragma: no cover - depends on user install
    import cv2

    HAVE_CV2 = True
except Exception:  # pragma: no cover
    HAVE_CV2 = False

_MP_ERROR = ""
try:  # pragma: no cover - depends on user install
    import mediapipe as mp

    HAVE_MEDIAPIPE = True
except Exception as _err:  # pragma: no cover
    HAVE_MEDIAPIPE = False
    _MP_ERROR = f"{type(_err).__name__}: {_err}"

log_ = logging.getLogger(__name__)

# --- detection -------------------------------------------------------------
SAMPLE_FPS = 6.0          # detections per second of video
PROXY_WIDTH = 640         # detection resolution
MIN_CONFIDENCE = 0.5      # MediaPipe confidence threshold
TARGET_ASPECT = 9 / 16    # output is vertical 9:16
DEFAULT_CROP_FRAC = TARGET_ASPECT / (16 / 9)  # crop width / frame width for a 16:9 source

MODEL_NAME = "blaze_face_short_range.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/" + MODEL_NAME
)

# --- shot cuts ---------------------------------------------------------------
SCENE_CUT_DIFF = 28.0     # mean grey-level change (0..255) between samples that counts as a cut
MIN_SHOT_SAMPLES = 3      # ignore "shots" shorter than this (flashes, glitches)
SNAP_SECONDS = 0.04       # a cut is followed with a ramp this short, effectively instant

# --- camera path -------------------------------------------------------------
HOLD_SECONDS = 2.0        # gaps up to this long are bridged smoothly
GLIDE_SECONDS = 1.2       # after a longer gap, park, then glide to the subject over this time
OUTLIER_WINDOW = 5        # median window (samples) that removes single false detections
SMOOTH_SECONDS = 0.45     # gaussian smoothing of where the subject is
DEAD_ZONE = 0.15          # each side, as a fraction of the crop width
EASE_SECONDS = 0.35       # gaussian easing of the camera's starts and stops
KEYFRAME_TOL = 0.004      # max error (fraction of frame width) when thinning keyframes
MAX_KEYFRAMES = 32        # keeps the ffmpeg expression short

# A detected face is (centre_x as a fraction of frame width, area as a fraction of the frame).
Box = tuple[float, float]


def available() -> bool:
    return HAVE_CV2


def backend_name() -> str:
    """Which detector will really be used: 'mediapipe', 'haar', or 'none'.

    This actually starts the detector once (and may download the model), so it
    never claims MediaPipe when MediaPipe cannot run.
    """
    return _probe()[0]


def backend_note() -> str:
    """Why the fallback detector is in use, or '' when everything is fine."""
    return _probe()[1]


def track_face_x(
    video: str | Path,
    start: float,
    end: float,
    ffmpeg_bin: str,
    workdir: str | Path,
    sample_fps: float = SAMPLE_FPS,
    log: Callable[[str], None] | None = None,
) -> list[tuple[float, float]] | None:
    """Return (seconds_from_clip_start, crop_centre_x_fraction) keyframes, or None."""
    if not HAVE_CV2 or end - start < 0.5:
        return None

    workdir = Path(workdir)
    proxy = workdir / "track_proxy.avi"
    ok = run(
        [
            ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", f"{max(0.0, start):.3f}",
            "-t", f"{max(0.3, end - start):.3f}",
            "-i", str(video),
            "-vf", f"fps={sample_fps},scale={PROXY_WIDTH}:-2",
            "-an", "-c:v", "mjpeg", "-q:v", "5",
            str(proxy),
        ],
        check=False,
    )
    if ok.returncode != 0 or not proxy.exists():
        return None

    try:
        sampled = _sample(proxy, sample_fps)
    finally:
        try:
            proxy.unlink()
        except OSError:
            pass

    if sampled is None:
        return None

    if log:
        seen = sum(v is not None for v in sampled.raw)
        msg = (
            f"  face tracking ({sampled.backend}): face found in "
            f"{seen}/{len(sampled.raw)} samples, {len(sampled.cuts)} shot change(s)"
        )
        if sampled.backend == "haar" and backend_note():
            msg += f"\n  WARNING: using the basic OpenCV detector. {backend_note()}"
        log(msg)

    if sampled.crop_frac >= 0.98:
        return None  # already vertical: nothing to pan across
    return _build_path(sampled.raw, sample_fps, sampled.cuts, sampled.crop_frac)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
class _MediaPipeDetector:
    """Classic mp.solutions API (older MediaPipe releases)."""

    def __init__(self) -> None:
        self._fd = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # 1 = full-range model, covers faces up to about 5 m away
            min_detection_confidence=MIN_CONFIDENCE,
        )

    def detect(self, bgr) -> list[Box]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self._fd.process(rgb)
        boxes: list[Box] = []
        for det in result.detections or []:
            rb = det.location_data.relative_bounding_box
            box = _to_box(rb.xmin, rb.ymin, rb.xmin + rb.width, rb.ymin + rb.height)
            if box:
                boxes.append(box)
        return boxes

    def close(self) -> None:
        self._fd.close()


class _MediaPipeTasksDetector:
    """MediaPipe Tasks API (current releases)."""

    def __init__(self, model_path: Path) -> None:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        options = vision.FaceDetectorOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            min_detection_confidence=MIN_CONFIDENCE,
        )
        self._fd = vision.FaceDetector.create_from_options(options)

    def detect(self, bgr) -> list[Box]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._fd.detect(image)
        boxes: list[Box] = []
        for det in result.detections or []:
            bb = det.bounding_box  # pixels
            box = _to_box(
                bb.origin_x / w, bb.origin_y / h,
                (bb.origin_x + bb.width) / w, (bb.origin_y + bb.height) / h,
            )
            if box:
                boxes.append(box)
        return boxes

    def close(self) -> None:
        self._fd.close()


def _to_box(x0: float, y0: float, x1: float, y1: float) -> Box | None:
    """Clip a relative box to the frame and turn it into (centre_x, area)."""
    x0, x1 = max(0.0, x0), min(1.0, x1)
    y0, y1 = max(0.0, y0), min(1.0, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return ((x0 + x1) / 2, (x1 - x0) * (y1 - y0))


class _HaarDetector:
    """Fallback when MediaPipe cannot run."""

    def __init__(self) -> None:
        self._frontal = _load_cascade("haarcascade_frontalface_default.xml")
        self._profile = _load_cascade("haarcascade_profileface.xml")

    def detect(self, bgr) -> list[Box]:
        if self._frontal is None:
            return []
        gray = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        h, w = gray.shape[:2]
        side = max(24, int(h * 0.10))

        found = self._run(self._frontal, gray, side)
        if not found and self._profile is not None:
            found = self._run(self._profile, gray, side)
            flipped = cv2.flip(gray, 1)
            for x, y, bw, bh in self._run(self._profile, flipped, side):
                found.append((w - x - bw, y, bw, bh))  # map back to the original frame

        return [((x + bw / 2) / w, (bw * bh) / (w * h)) for x, _, bw, bh in found]

    @staticmethod
    def _run(cascade, img, side):
        found = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(side, side))
        return [tuple(map(float, b)) for b in found]

    def close(self) -> None:
        pass


def _load_cascade(name: str):
    path = Path(cv2.data.haarcascades) / name
    if not path.exists():
        return None
    cascade = cv2.CascadeClassifier(str(path))
    return None if cascade.empty() else cascade


def _model_file() -> tuple[Path | None, str]:
    """Find (or download once) the Tasks face model. Returns (path, reason_if_missing)."""
    env = os.environ.get("CILOS_FACE_MODEL")
    if env:
        path = Path(env)
        return (path, "") if path.exists() else (None, f"CILOS_FACE_MODEL points to a missing file: {env}")

    try:
        target = config_dir() / "models" / MODEL_NAME
    except Exception as err:
        return None, f"no place to keep the face model ({err})"
    if target.exists():
        return target, ""

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(".part")
        with urllib.request.urlopen(MODEL_URL, timeout=15) as resp, open(part, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        if part.stat().st_size < 50_000:
            raise ValueError("downloaded file is too small to be the model")
        part.replace(target)
        return target, ""
    except Exception as err:
        return None, (
            f"Could not download the face model ({type(err).__name__}: {err}). "
            f"Download {MODEL_URL} and save it as {target}, or "
            f"install an older MediaPipe that still has mp.solutions, e.g. pip install mediapipe==0.10.14"
        )


@lru_cache(maxsize=1)
def _probe():
    """Work out, once, which detector really works. Returns (name, note, factory)."""
    if not HAVE_CV2:
        return "none", "OpenCV (cv2) is not installed.", None

    notes: list[str] = []
    if not HAVE_MEDIAPIPE:
        notes.append(f"MediaPipe is not installed ({_MP_ERROR}).")
    else:
        factory = None
        if hasattr(mp, "solutions"):
            factory = _MediaPipeDetector
        else:
            path, why = _model_file()
            if path is not None:
                factory = lambda p=path: _MediaPipeTasksDetector(p)  # noqa: E731
            else:
                notes.append(why)
        if factory is not None:
            try:
                factory().close()
                return "mediapipe", "", factory
            except Exception as err:
                notes.append(f"MediaPipe failed to start ({type(err).__name__}: {err}).")

    note = " ".join(notes)
    log_.warning("Face tracking is using the basic OpenCV detector. %s", note)
    return "haar", note, _HaarDetector


def _make_detector():
    name, _, factory = _probe()
    return factory(), name


# ---------------------------------------------------------------------------
# Sampling and subject selection
# ---------------------------------------------------------------------------
@dataclass
class _Sampled:
    raw: list[float | None]   # per sample: chosen subject centre (0..1), or None
    cuts: list[int]           # sample indexes where a new shot begins
    crop_frac: float          # width of the vertical crop as a fraction of frame width
    backend: str


def _sample(proxy: Path, sample_fps: float) -> _Sampled | None:
    detector, backend = _make_detector()
    cap = cv2.VideoCapture(str(proxy))
    if not cap.isOpened():
        detector.close()
        return None

    raw: list[float | None] = []
    cuts: list[int] = []
    prev: float | None = None
    prev_small = None
    crop_frac = DEFAULT_CROP_FRAC
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not raw:
                h, w = frame.shape[:2]
                crop_frac = (h * TARGET_ASPECT) / w

            # A big change in the whole picture means a cut to another camera.
            small = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA
            )
            if prev_small is not None and float(cv2.absdiff(small, prev_small).mean()) >= SCENE_CUT_DIFF:
                cuts.append(len(raw))
                prev = None  # the old subject means nothing in the new shot
            prev_small = small

            x = pick_subject(detector.detect(frame), prev)
            if x is not None:
                prev = x
            raw.append(x)
    finally:
        cap.release()
        detector.close()

    if not raw:
        return None
    return _Sampled(raw, cuts, crop_frac, backend)


def pick_subject(boxes: list[Box], prev: float | None) -> float | None:
    """Choose the subject: the largest face, weighted toward the current subject."""
    best_x: float | None = None
    best_score = -1.0
    for cx, area in boxes:
        # A face far from the current subject must be much bigger to win.
        score = area if prev is None else area / (1.0 + 24.0 * abs(cx - prev))
        if score > best_score:
            best_x, best_score = cx, score
    return best_x


# ---------------------------------------------------------------------------
# Turning detections into a smooth camera path
# ---------------------------------------------------------------------------
def _clean_cuts(cuts, n: int) -> list[int]:
    """Drop cuts that would leave a tiny shot (camera flashes, one-frame glitches)."""
    kept: list[int] = []
    last = 0
    for c in sorted(set(cuts)):
        if c - last >= MIN_SHOT_SAMPLES and n - c >= MIN_SHOT_SAMPLES:
            kept.append(c)
            last = c
    return kept


def _fill_gaps(vals: list[float | None], fps: float) -> list[float] | None:
    """Replace missing detections. We know the whole clip, so bridge gaps smoothly."""
    idx = [i for i, v in enumerate(vals) if v is not None]
    if not idx:
        return None
    n = len(vals)
    out = [0.0] * n
    first, last = idx[0], idx[-1]
    for i in range(first):
        out[i] = vals[first]          # start on the subject, not on the centre
    for i in range(last + 1, n):
        out[i] = vals[last]           # stay parked after the subject is gone

    hold = max(1, int(HOLD_SECONDS * fps))
    glide = max(1, int(GLIDE_SECONDS * fps))
    for a, b in zip(idx, idx[1:]):
        out[a] = vals[a]
        gap = b - a - 1
        if gap <= 0:
            continue
        xa, xb = vals[a], vals[b]
        if gap <= hold:
            for i in range(a + 1, b):
                out[i] = xa + (xb - xa) * (i - a) / (b - a)
        else:
            g0 = max(a, b - glide)    # park on the last known position, then glide over
            for i in range(a + 1, b):
                out[i] = xa if i <= g0 else xa + (xb - xa) * (i - g0) / (b - g0)
    out[last] = vals[last]
    return out


def _median_filter(values: list[float], window: int) -> list[float]:
    half = window // 2
    return [
        statistics.median(values[max(0, i - half): i + half + 1])
        for i in range(len(values))
    ]


def _gauss(values: list[float], sigma: float) -> list[float]:
    """Zero-lag gaussian smoothing (edges are held, not pulled toward zero)."""
    n = len(values)
    if sigma < 0.3 or n < 2:
        return list(values)
    radius = max(1, int(math.ceil(3 * sigma)))
    offsets = range(-radius, radius + 1)
    weights = [math.exp(-0.5 * (k / sigma) ** 2) for k in offsets]
    total = sum(weights)
    out = []
    for i in range(n):
        acc = 0.0
        for k, w in zip(offsets, weights):
            acc += w * values[min(n - 1, max(0, i + k))]
        out.append(acc / total)
    return out


def _dead_zone(values: list[float], dz: float) -> list[float]:
    """The camera stays put until the subject leaves +-dz, then keeps them at that edge."""
    cam = values[0]
    out = []
    for v in values:
        if v - cam > dz:
            cam = v - dz
        elif cam - v > dz:
            cam = v + dz
        out.append(cam)
    return out


def _shot_path(vals: list[float | None], fps: float, crop_frac: float) -> list[float]:
    """Camera centre for every sample of one continuous shot."""
    half = crop_frac / 2
    filled = _fill_gaps(vals, fps)
    if filled is None:
        return [0.5] * len(vals)      # no face in this shot (slides, b-roll): stay centred

    filled = _median_filter(filled, OUTLIER_WINDOW)
    # The crop cannot go past the frame edge, so do not chase a target beyond it.
    filled = [min(1.0 - half, max(half, v)) for v in filled]
    smooth = _gauss(filled, SMOOTH_SECONDS * fps)
    cam = _dead_zone(smooth, DEAD_ZONE * crop_frac)
    cam = _gauss(cam, EASE_SECONDS * fps)
    return [min(1.0 - half, max(half, v)) for v in cam]


def _simplify(points: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker: fewest keyframes whose straight lines stay within tol."""
    n = len(points)
    if n <= 2:
        return list(points)
    keep = [False] * n
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        t0, x0 = points[i]
        t1, x1 = points[j]
        worst, worst_d = -1, 0.0
        for k in range(i + 1, j):
            t, x = points[k]
            d = abs(x - (x0 + (x1 - x0) * (t - t0) / (t1 - t0)))
            if d > worst_d:
                worst, worst_d = k, d
        if worst != -1 and worst_d > tol:
            keep[worst] = True
            stack.append((i, worst))
            stack.append((worst, j))
    return [p for p, k in zip(points, keep) if k]


def _build_path(
    raw: list[float | None],
    sample_fps: float,
    cuts=(),
    crop_frac: float = DEFAULT_CROP_FRAC,
) -> list[tuple[float, float]] | None:
    n = len(raw)
    if n == 0 or not any(v is not None for v in raw):
        return None

    dt = 1.0 / sample_fps
    snap = min(SNAP_SECONDS, dt / 2)
    bounds = [0, *_clean_cuts(cuts, n), n]
    spans = list(zip(bounds, bounds[1:]))
    paths = [_shot_path(raw[a:b], sample_fps, crop_frac) for a, b in spans]

    if n == 1:
        return [(0.0, paths[0][0]), (dt, paths[0][0])]

    def assemble(tol: float) -> list[tuple[float, float]]:
        kfs: list[tuple[float, float]] = []
        last = len(spans) - 1
        for s, ((a, b), path) in enumerate(zip(spans, paths)):
            pts = _simplify([((a + i) * dt, x) for i, x in enumerate(path)], tol)
            # A cut falls somewhere between two samples; put the jump in the middle.
            if s > 0:
                pts[0] = (a * dt - dt / 2, pts[0][1])
            if s < last:
                pts[-1] = (b * dt - dt / 2 - snap, pts[-1][1])
            kfs.extend(pts)
        return kfs

    tol = KEYFRAME_TOL
    keyframes = assemble(tol)
    while len(keyframes) > MAX_KEYFRAMES and tol < 0.02:
        tol *= 1.6
        keyframes = assemble(tol)
    if len(keyframes) > MAX_KEYFRAMES:
        keyframes = _downsample(keyframes, MAX_KEYFRAMES)
    return keyframes


def _downsample(points: list[tuple[float, float]], max_points: int) -> list[tuple[float, float]]:
    if len(points) <= max_points:
        return points
    stride = len(points) / max_points
    picked = [points[min(len(points) - 1, int(i * stride))] for i in range(max_points)]
    picked[-1] = points[-1]
    return picked
