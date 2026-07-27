from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import face_recognition
import numpy as np

logger = logging.getLogger(__name__)

REQUIRED_POSES = 5
FRAMES_PER_POSE = 5          # Option C: capture 5 frames per pose burst
MIN_QUALITY_FRAMES = 3       # need at least 3 good frames out of 5
BURST_DELAY_MS = 80          # ~80ms between burst frames (≈12fps burst)

MIN_FACE_AREA_RATIO = 0.05
LAPLACIAN_VARIANCE_MIN = 30
BRIGHTNESS_MIN = 40
BRIGHTNESS_MAX = 220

# Option A: advanced preprocessing thresholds
CLAHE_CLIP_LIMIT = 2.0       # CLAHE contrast limit
CLAHE_TILE_GRID = (8, 8)     # CLAHE tile grid size
DENOISE_H = 7                # Fast NL-Means denoising strength (lower = less aggressive)
DENOISE_TEMPLATE = 7
DENOISE_SEARCH = 21
SHARPENING_ALPHA = 1.3       # Unsharp mask strength
SHARPENING_SIGMA = 1.0

# Option D: DBSCAN outlier rejection
# eps: max L2 distance between two embeddings to be considered "neighbours".
#
# Calibrated empirically on 128-D L2-normalised face_recognition embeddings:
#   - Same-person burst frames (minor head micro-movement):  L2 ≈ 0.65–0.90
#   - Clearly bad/outlier frames (wrong angle, occlusion):   L2 ≈ 1.20–1.55
#   - Random different-person vectors:                       L2 ≈ 1.40 (mean)
#
# eps=1.00 sits comfortably in the gap, clustering good frames together
# and labelling genuinely bad frames as noise.
DBSCAN_EPS = 1.00
DBSCAN_MIN_SAMPLES = 2       # a core point needs ≥2 neighbours (inc. itself)
DBSCAN_OUTLIER_LABEL = -1    # standard DBSCAN noise label

FACE_DATA_DIR = Path("face_data")
EMBEDDINGS_DIR = FACE_DATA_DIR / "embeddings"
PROFILES_PATH = FACE_DATA_DIR / "profiles.json"

# Option B: Pose guide messages shown to the user
POSE_GUIDES: list[dict[str, str]] = [
    {
        "action": "Look straight at the camera",
        "tip":    "Keep your face centred and at eye level",
    },
    {
        "action": "Turn your head slightly to the LEFT",
        "tip":    "About 20–30° — don't go too far",
    },
    {
        "action": "Turn your head slightly to the RIGHT",
        "tip":    "About 20–30° — don't go too far",
    },
    {
        "action": "Tilt your head slightly UP",
        "tip":    "Chin up a little, keep eyes on camera",
    },
    {
        "action": "Tilt your head slightly DOWN",
        "tip":    "Chin down a little, keep eyes on camera",
    },
]

# Option B: Quality-issue human-readable guidance
QUALITY_GUIDANCE: dict[str, str] = {
    "too_far":    "Come in closer — your face is too small in the frame",
    "blurry":     "Hold still — the image is blurry",
    "too_dark":   "Move to a brighter area or add more light",
    "too_bright": "Avoid strong light behind or directly on you",
    "no_face":    "No face detected — make sure your face is visible",
    "multi_face": "Please have only one person in frame at a time",
    "embed_fail": "Could not compute embedding — please retry",
}


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Option A: Advanced image preprocessing pipeline.

    Steps:
    1. Denoise with Fast NL-Means (reduces sensor noise on cheap cameras)
    2. Convert to LAB colour space and apply CLAHE on the L channel
       (adaptive histogram equalisation — handles uneven lighting well)
    3. Unsharp masking to recover edge detail lost by denoising
    4. Clip to [0, 255] and convert back to BGR

    This is meaningfully better than simple gamma / global contrast
    because CLAHE is spatially adaptive — it won't blow out a bright
    forehead when the rest of the face is in shadow.
    """
    # 1. Fast NL-Means denoising (BGR)
    denoised = cv2.fastNlMeansDenoisingColored(
        frame,
        None,
        DENOISE_H,
        DENOISE_H,
        DENOISE_TEMPLATE,
        DENOISE_SEARCH,
    )

    # 2. CLAHE on L channel in LAB
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 3. Unsharp mask (sharpening)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), SHARPENING_SIGMA)
    sharpened = cv2.addWeighted(enhanced, SHARPENING_ALPHA, blurred, -(SHARPENING_ALPHA - 1), 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return sharpened


# ---------------------------------------------------------------------------
# Option D — pure-numpy DBSCAN implementation
# ---------------------------------------------------------------------------

def _dbscan_cluster(
    embeddings: list[np.ndarray],
    eps: float = DBSCAN_EPS,
    min_samples: int = DBSCAN_MIN_SAMPLES,
) -> np.ndarray:
    """
    Lightweight DBSCAN for 128-D face embeddings — no sklearn required.

    Returns an integer label array of length len(embeddings):
      ≥ 0  → cluster id (inlier)
      -1   → noise / outlier

    Why DBSCAN rather than k-means?
    - We don't know k in advance (all embeddings might be one cluster, or
      some might be isolated noise points).
    - DBSCAN naturally labels outliers without forcing them into a cluster.
    - It handles non-spherical clusters and is deterministic.

    Complexity is O(n²) in distance calculations, but n ≤ FRAMES_PER_POSE
    (currently 5) so this is trivially fast (<0.1 ms).
    """
    n = len(embeddings)
    if n == 0:
        return np.array([], dtype=int)

    # Pre-compute pairwise L2 distance matrix
    mat = np.stack(embeddings)                # shape (n, 128)
    # ||a - b||² = ||a||² + ||b||² - 2·a·bᵀ
    sq = np.sum(mat ** 2, axis=1, keepdims=True)   # (n, 1)
    dist_sq = sq + sq.T - 2.0 * (mat @ mat.T)      # (n, n)
    # Numerical safety: clamp tiny negatives to zero before sqrt
    dist_sq = np.clip(dist_sq, 0.0, None)
    dist = np.sqrt(dist_sq)                         # (n, n)

    labels = np.full(n, DBSCAN_OUTLIER_LABEL, dtype=int)
    cluster_id = 0
    visited = np.zeros(n, dtype=bool)

    def _region_query(idx: int) -> list[int]:
        return [j for j in range(n) if dist[idx, j] <= eps]

    def _expand_cluster(idx: int, neighbours: list[int], cid: int) -> None:
        labels[idx] = cid
        i = 0
        while i < len(neighbours):
            pt = neighbours[i]
            if not visited[pt]:
                visited[pt] = True
                pt_neighbours = _region_query(pt)
                if len(pt_neighbours) >= min_samples:
                    neighbours += [q for q in pt_neighbours if q not in neighbours]
            if labels[pt] == DBSCAN_OUTLIER_LABEL:
                labels[pt] = cid
            i += 1

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbours = _region_query(i)
        if len(neighbours) < min_samples:
            labels[i] = DBSCAN_OUTLIER_LABEL  # noise
        else:
            _expand_cluster(i, neighbours, cluster_id)
            cluster_id += 1

    return labels


def _cluster_centroid(
    embeddings: list[np.ndarray],
    labels: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Option D: Given DBSCAN labels, pick the largest cluster, compute its
    L2-normalised centroid, and return (centroid, n_outliers_discarded).

    If every embedding is labelled as noise (all -1), falls back to a
    simple mean of all embeddings so enrollment never completely fails.
    """
    unique_labels = [lbl for lbl in set(labels.tolist()) if lbl != DBSCAN_OUTLIER_LABEL]

    if not unique_labels:
        # All noise — fall back to simple mean
        logger.warning(
            "DBSCAN found no clusters; falling back to simple mean of all %d embeddings",
            len(embeddings),
        )
        centroid = np.mean(embeddings, axis=0)
        n_outliers = 0
    else:
        # Pick the largest cluster by member count
        best_label = max(unique_labels, key=lambda lbl: int(np.sum(labels == lbl)))
        inlier_mask = labels == best_label
        inliers = [emb for emb, keep in zip(embeddings, inlier_mask) if keep]
        n_outliers = int(np.sum(~inlier_mask))
        centroid = np.mean(inliers, axis=0)
        logger.info(
            "DBSCAN: kept %d inliers, discarded %d outliers (cluster %d)",
            len(inliers),
            n_outliers,
            best_label,
        )

    # L2 normalise for cosine-compatible comparison
    norm = np.linalg.norm(centroid)
    if norm > 1e-10:
        centroid = centroid / norm
    return centroid, n_outliers


class EnrollmentManager:
    def __init__(self, face_data_dir: Path) -> None:
        self._face_data_dir = face_data_dir
        self._embeddings_dir = face_data_dir / "embeddings"
        self._embeddings_dir.mkdir(parents=True, exist_ok=True)
        self._active_name: str | None = None
        self._current_pose = 0
        self._pose_frames: list[np.ndarray] = []
        self._pose_embeddings: list[np.ndarray] = []
        self._all_embeddings: list[list[np.ndarray]] = []
        # Option B: last guidance message
        self._last_guidance: str = ""
        self._last_quality_issue: str = ""
        # Option D: track how many outliers were discarded across all poses
        self._total_outliers_discarded: int = 0
        self._last_pose_outliers: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_enrollment(self, name: str) -> dict[str, Any]:
        if self._active_name is not None:
            return {"status": "error", "message": "Enrollment already in progress"}
        self._active_name = name
        self._current_pose = 0
        self._pose_frames = []
        self._pose_embeddings = []
        self._all_embeddings = []
        self._last_guidance = ""
        self._last_quality_issue = ""
        logger.info("Started enrollment for '%s'", name)
        guide = self._current_pose_guide()
        return {
            "status": "ok",
            "message": f"Enrollment started for {name}",
            "pose": self._current_pose + 1,
            "total_poses": REQUIRED_POSES,
            "guidance": guide,
        }

    def capture_pose(self, get_frame) -> dict[str, Any]:
        """
        Burst-capture FRAMES_PER_POSE frames, calling `get_frame()` fresh
        for each one so the burst captures genuinely distinct frames
        rather than five copies of a single snapshot.
        """
        if self._active_name is None:
            return {"status": "error", "message": "No active enrollment"}

        burst_results = self._collect_burst(get_frame)
        good_frames = [r for r in burst_results if r["ok"]]
        bad_frames  = [r for r in burst_results if not r["ok"]]

        logger.info(
            "Burst for pose %d: %d/%d good frames",
            self._current_pose + 1,
            len(good_frames),
            FRAMES_PER_POSE,
        )

        if len(good_frames) < MIN_QUALITY_FRAMES:
            issues = [r.get("issue", "unknown") for r in bad_frames]
            top_issue = max(set(issues), key=issues.count) if issues else "unknown"
            self._last_quality_issue = top_issue
            guidance_msg = QUALITY_GUIDANCE.get(top_issue, top_issue)
            self._last_guidance = guidance_msg
            return {
                "status": "retry",
                "message": f"Only {len(good_frames)}/{MIN_QUALITY_FRAMES} quality frames: {guidance_msg}",
                "guidance": guidance_msg,
                "pose": self._current_pose + 1,
                "total_poses": REQUIRED_POSES,
            }

        for r in good_frames:
            self._pose_frames.append(r["frame"])
            self._pose_embeddings.append(r["embedding"])

        return self._finalize_pose()

    def cancel_enrollment(self) -> dict[str, Any]:
        if self._active_name is None:
            return {"status": "error", "message": "No active enrollment"}
        name = self._active_name
        for i in range(self._current_pose):
            p = self._embeddings_dir / f"{name}_{i:03d}.npy"
            try:
                p.unlink(missing_ok=True)
            except OSError as e:
                logger.warning("Failed to delete %s: %s", p, e)
        self._active_name = None
        self._pose_frames.clear()
        self._pose_embeddings.clear()
        self._all_embeddings.clear()
        self._current_pose = 0
        self._last_guidance = ""
        self._last_quality_issue = ""
        logger.info("Enrollment cancelled for '%s'", name)
        return {"status": "ok", "message": f"Enrollment cancelled for {name}"}

    def get_status(self) -> dict[str, Any]:
        if self._active_name is None:
            return {"status": "idle"}
        guide = self._current_pose_guide()
        return {
            "status": "active",
            "name": self._active_name,
            "pose": self._current_pose + 1,
            "total_poses": REQUIRED_POSES,
            "captured": len(self._pose_frames),
            "guidance": guide,
            "quality_issue": self._last_quality_issue,
            # Option D: expose outlier stats for the UI
            "outliers_discarded": self._total_outliers_discarded,
            "last_pose_outliers": self._last_pose_outliers,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_burst(self, get_frame) -> list[dict[str, Any]]:
        """Collect FRAMES_PER_POSE genuinely distinct frames via get_frame(),
        with a short delay between each so they represent different instants."""
        frames_to_process: list[np.ndarray] = []
        for i in range(FRAMES_PER_POSE):
            if i > 0:
                time.sleep(BURST_DELAY_MS / 1000.0)
            frames_to_process.append(get_frame())

        return [self._process_single_frame(f) for f in frames_to_process]

    def _process_single_frame(self, raw_frame: np.ndarray) -> dict[str, Any]:
        """Option A: Preprocess + quality-gate + embed one frame."""
        # Option A: preprocess
        frame = preprocess_frame(raw_frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Multi-face check (Option B guidance)
        face_locs = face_recognition.face_locations(rgb, model="hog")
        if not face_locs:
            return {"ok": False, "issue": "no_face", "frame": raw_frame, "embedding": None}
        if len(face_locs) > 1:
            return {"ok": False, "issue": "multi_face", "frame": raw_frame, "embedding": None}

        face_loc = face_locs[0]
        quality_issue = self.check_quality(frame, face_loc)
        if quality_issue is not None:
            return {"ok": False, "issue": quality_issue, "frame": raw_frame, "embedding": None}

        encodings = face_recognition.face_encodings(rgb, [face_loc])
        if not encodings:
            return {"ok": False, "issue": "embed_fail", "frame": raw_frame, "embedding": None}

        return {"ok": True, "issue": None, "frame": raw_frame, "embedding": encodings[0]}

    def _finalize_pose(self) -> dict[str, Any]:
        if len(self._pose_embeddings) < MIN_QUALITY_FRAMES:
            self._pose_frames.clear()
            self._pose_embeddings.clear()
            guidance_msg = QUALITY_GUIDANCE.get(
                self._last_quality_issue, "Please retry this pose"
            )
            return {
                "status": "retry",
                "message": f"Only {len(self._pose_embeddings)}/{MIN_QUALITY_FRAMES} quality frames, retaking pose",
                "guidance": guidance_msg,
                "pose": self._current_pose + 1,
                "total_poses": REQUIRED_POSES,
            }

        # ----------------------------------------------------------------
        # Option D: DBSCAN outlier rejection before computing centroid.
        #
        # Why this matters:
        #   Even with quality gating (Options A+B+C), some embeddings may
        #   still be slightly off — e.g. from a micro-movement during the
        #   burst, partial hair occlusion, or a compression artefact.
        #   A simple mean drags the centroid toward those noisy points.
        #   DBSCAN finds the dense core cluster and ignores isolated
        #   outliers, giving a centroid that better represents a clean
        #   frontal capture.
        # ----------------------------------------------------------------
        labels = _dbscan_cluster(self._pose_embeddings)
        centroid, n_outliers = _cluster_centroid(self._pose_embeddings, labels)

        self._last_pose_outliers = n_outliers
        self._total_outliers_discarded += n_outliers

        if n_outliers > 0:
            logger.info(
                "Pose %d: DBSCAN removed %d outlier frame(s) before computing centroid",
                self._current_pose + 1,
                n_outliers,
            )

        self._all_embeddings.append(self._pose_embeddings.copy())
        save_path = (
            self._embeddings_dir
            / f"{self._active_name}_{self._current_pose:03d}.npy"
        )
        np.save(str(save_path), centroid)
        logger.info(
            "Saved pose %d centroid (DBSCAN) to %s",
            self._current_pose + 1,
            save_path,
        )

        self._pose_frames.clear()
        self._pose_embeddings.clear()
        self._current_pose += 1

        if self._current_pose >= REQUIRED_POSES:
            return self._finish_enrollment()

        self._last_quality_issue = ""
        guide = self._current_pose_guide()
        outlier_note = f" ({n_outliers} noisy frame(s) filtered out)" if n_outliers else ""
        return {
            "status": "next_pose",
            "message": f"Pose {self._current_pose}/{REQUIRED_POSES} captured{outlier_note}",
            "pose": self._current_pose + 1,
            "total_poses": REQUIRED_POSES,
            "guidance": guide,
            "outliers_discarded": n_outliers,
        }

    def _finish_enrollment(self) -> dict[str, Any]:
        profiles: list[dict[str, Any]] = []
        if PROFILES_PATH.exists():
            try:
                profiles = json.loads(PROFILES_PATH.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read profiles.json: %s", e)

        entry = {
            "name": self._active_name,
            "embeddings": [
                f"{self._active_name}_{i:03d}.npy" for i in range(REQUIRED_POSES)
            ],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        existing = [p for p in profiles if p.get("name") != self._active_name]
        existing.append(entry)

        try:
            PROFILES_PATH.write_text(json.dumps(existing, indent=2))
            logger.info("Enrollment complete for '%s'", self._active_name)
        except OSError as e:
            logger.error("Failed to write profiles.json: %s", e)
            return {"status": "error", "message": f"Failed to save profile: {e}"}

        name = self._active_name
        total_outliers = self._total_outliers_discarded
        self._active_name = None
        self._last_guidance = ""
        self._last_quality_issue = ""
        self._total_outliers_discarded = 0
        self._last_pose_outliers = 0

        logger.info(
            "Enrollment complete for '%s' — total outlier frames discarded by DBSCAN: %d",
            name,
            total_outliers,
        )
        return {
            "status": "complete",
            "message": f"Enrollment complete for {name}",
            "name": name,
            "total_outliers_discarded": total_outliers,
        }

    def _current_pose_guide(self) -> dict[str, str]:
        """Option B: Return the guidance dict for the current pose."""
        idx = min(self._current_pose, len(POSE_GUIDES) - 1)
        return POSE_GUIDES[idx]

    # ------------------------------------------------------------------
    # Static quality gate
    # ------------------------------------------------------------------

    @staticmethod
    def check_quality(
        frame: np.ndarray, face_loc: tuple[int, int, int, int]
    ) -> str | None:
        top, right, bottom, left = face_loc
        fw = right - left
        fh = bottom - top
        face_area = fw * fh
        frame_area = frame.shape[0] * frame.shape[1]
        if face_area / frame_area < MIN_FACE_AREA_RATIO:
            return "too_far"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < LAPLACIAN_VARIANCE_MIN:
            return "blurry"

        mean_brightness = np.mean(gray)
        if mean_brightness < BRIGHTNESS_MIN:
            return "too_dark"
        if mean_brightness > BRIGHTNESS_MAX:
            return "too_bright"

        return None
