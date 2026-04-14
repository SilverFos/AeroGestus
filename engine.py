import cv2
import torch
import clip
from PIL import Image
import numpy as np
import time
import os
import sys
import pickle
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

# Resolve base dir: inside a PyInstaller bundle use _MEIPASS, else script dir
_BASE = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

def _asset(name):
    return os.path.join(_BASE, name)

# Persistent data lives beside the exe/script so saves survive between runs
_DATA_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))


class MotionDetector:
    HISTORY_LEN  = 20
    MIN_FRAMES   = 8
    SWIPE_THRESH = 0.18
    FLIP_THRESH  = 0.45
    COOLDOWN     = 1.2

    def __init__(self):
        self.history: deque[dict]      = deque(maxlen=self.HISTORY_LEN)
        self.last_fire: float          = 0.0
        self.motion_gestures: dict     = {}

    def register(self, name: str, action: str, gesture_type: str,
                 direction: str | None = None):
        self.motion_gestures[name] = {
            "action": action,
            "type":   gesture_type,
            "dir":    direction,
        }

    def push(self, landmarks, active_names: set | None = None):
        snap = self._extract(landmarks)
        self.history.append(snap)
        if len(self.history) < self.MIN_FRAMES:
            return None, None
        if time.time() - self.last_fire < self.COOLDOWN:
            return None, None
        return self._evaluate(active_names)

    def clear(self):
        self.history.clear()

    def _extract(self, lm) -> dict:
        v1 = np.array([lm[5].x  - lm[0].x, lm[5].y  - lm[0].y, lm[5].z  - lm[0].z])
        v2 = np.array([lm[17].x - lm[0].x, lm[17].y - lm[0].y, lm[17].z - lm[0].z])
        normal = np.cross(v1, v2)
        norm   = np.linalg.norm(normal)
        if norm > 1e-6:
            normal /= norm
        return {"wx": lm[0].x, "wy": lm[0].y, "nz": float(normal[2]), "t": time.time()}

    def _evaluate(self, active_names):
        snap = list(self.history)
        old, new = snap[0], snap[-1]
        dx, dy, dnz = new["wx"] - old["wx"], new["wy"] - old["wy"], new["nz"] - old["nz"]

        detected = set()
        if abs(dx)  > self.SWIPE_THRESH: detected.add("right" if dx  > 0 else "left")
        if abs(dy)  > self.SWIPE_THRESH: detected.add("down"  if dy  > 0 else "up")
        if abs(dnz) > self.FLIP_THRESH:  detected.add("front" if dnz > 0 else "back")

        if not detected:
            return None, None

        for name, info in self.motion_gestures.items():
            if active_names and name not in active_names:
                continue
            if info["dir"] in detected:
                self.last_fire = time.time()
                self.history.clear()
                print(f"[MOTION] '{name}' dx={dx:.3f} dy={dy:.3f} dnz={dnz:.3f}")
                return name, info["action"]

        return None, None


class AeroEngine:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=_asset('gesture_recognizer.task'))
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        self.motion_detector = MotionDetector()

        self.db_path      = os.path.join(_DATA_DIR, "gesture_memory.pkl")
        self.gesture_db   = {}
        self.active_module = "Default"
        self.load_database()

        self.scanning_enabled     = False
        self.last_toggle_time     = 0.0
        self.last_triggered_action: str | None = None

        self.prev_landmarks       = None
        self.movement_energy      = 0.0
        self.active_phases:  dict = {}
        self.last_phase_match_time: float = 0.0

    # ── DB ───────────────────────────────────────────────────────
    def load_database(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "rb") as f:
                    self.gesture_db = pickle.load(f)
                # Back-fill missing 'module' key for old records
                for data in self.gesture_db.values():
                    if "module" not in data:
                        data["module"] = "Default"
            except Exception:
                self.gesture_db = {}

    def _save(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self.gesture_db, f)

    def register_motion_gesture(self, name: str, action: str,
                                 gesture_type: str, direction: str,
                                 module: str = "Default"):
        self.motion_detector.register(name, action, gesture_type, direction)
        self.gesture_db[name.lower().strip()] = {
            "phases":      [],
            "action":      action,
            "is_temporal": False,
            "motion":      True,
            "motion_type": gesture_type,
            "motion_dir":  direction,
            "module":      module,
        }
        self._save()

    def record_gesture(self, frames: list, name: str, action: str,
                       module: str = "Default"):
        embeddings = []
        indices = [0, len(frames) // 2, len(frames) - 1] if len(frames) > 5 else [0]
        for idx in indices:
            img = self.clip_preprocess(
                Image.fromarray(cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB))
            ).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.clip_model.encode_image(img)
                feat /= feat.norm(dim=-1, keepdim=True)
                embeddings.append(feat)
        self.gesture_db[name.lower().strip()] = {
            "phases":      embeddings,
            "action":      action,
            "is_temporal": len(embeddings) > 1,
            "motion":      False,
            "module":      module,
        }
        self._save()

    # ── Helpers ──────────────────────────────────────────────────
    def _active_names(self) -> set:
        """Names of gestures belonging to the active module."""
        return {k for k, v in self.gesture_db.items()
                if v.get("module", "Default") == self.active_module}

    def get_finger_count(self, landmarks) -> int:
        tips, knuckles = [8, 12, 16, 20], [6, 10, 14, 18]
        return sum(1 for t, k in zip(tips, knuckles)
                   if landmarks[t].y < landmarks[k].y)

    def calculate_movement(self, landmarks) -> float:
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            return 0.0
        dist = sum(
            np.sqrt((self.prev_landmarks[i].x - landmarks[i].x) ** 2 +
                    (self.prev_landmarks[i].y - landmarks[i].y) ** 2)
            for i in [0, 8, 12]
        )
        self.prev_landmarks = landmarks
        return dist

    # ── Main processing ──────────────────────────────────────────
    def process_frame(self, frame):
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res    = self.recognizer.recognize_for_video(mp_img, int(time.time() * 1000))

        triggered_action = None
        status           = "IDLE" if not self.scanning_enabled else "SCANNING"

        if not res.hand_landmarks:
            self.motion_detector.clear()
            self.prev_landmarks = None
            return frame, status, None

        landmarks            = res.hand_landmarks[0]
        curr_fingers         = self.get_finger_count(landmarks)
        self.movement_energy = self.calculate_movement(landmarks)
        now                  = time.time()
        active               = self._active_names()

        # Gate
        gate = ("OPEN" if curr_fingers >= 4 else
                "CLOSED" if curr_fingers == 0 else "NEUTRAL")

        if self.movement_energy < 0.02:
            if gate == "OPEN"   and not self.scanning_enabled and now - self.last_toggle_time > 1.5:
                self.scanning_enabled, self.last_toggle_time = True,  now
                self.motion_detector.clear()
                print(">>> SCANNING ENABLED")
            elif gate == "CLOSED" and self.scanning_enabled  and now - self.last_toggle_time > 1.5:
                self.scanning_enabled, self.last_toggle_time = False, now
                self.motion_detector.clear()
                print(">>> SCANNING DISABLED")

        if not self.scanning_enabled:
            return frame, status, None

        # Layer 1 — motion (pass active module filter)
        motion_name, motion_act = self.motion_detector.push(landmarks, active)
        if motion_act:
            return frame, f"MOTION: {motion_name}", motion_act

        # Layer 2 — CLIP static / temporal
        if gate != "NEUTRAL":
            return frame, status, None

        img = self.clip_preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            live_feat = self.clip_model.encode_image(img)
            live_feat /= live_feat.norm(dim=-1, keepdim=True)

        clip_db = {k: v for k, v in self.gesture_db.items()
                   if k in active and v.get("phases") and not v.get("motion", False)}

        candidates = []
        for name, data in clip_db.items():
            if "palm"  in name and curr_fingers < 3: continue
            if ("finger" in name or "point" in name) and curr_fingers > 2: continue
            curr_idx = self.active_phases.get(name, 0)
            sim      = (live_feat @ data["phases"][curr_idx].T).item()
            candidates.append({"name": name, "sim": sim, "data": data, "idx": curr_idx})

        candidates.sort(key=lambda x: x["sim"], reverse=True)

        if candidates:
            best = candidates[0]
            if best["sim"] > 0.82:
                print(f"[CLIP] {best['name']}  sim={best['sim']:.3f}  "
                      f"fingers={curr_fingers}  energy={self.movement_energy:.4f}")

            if best["sim"] > 0.88:
                if best["data"]["is_temporal"]:
                    if self.movement_energy > 0.005:
                        if best["idx"] < len(best["data"]["phases"]) - 1:
                            self.active_phases[best["name"]] = best["idx"] + 1
                            self.last_phase_match_time = now
                        else:
                            triggered_action = best["data"]["action"]
                            self.active_phases = {}
                else:
                    if best["sim"] > 0.94 and self.movement_energy < 0.02:
                        if best["data"]["action"] != self.last_triggered_action:
                            triggered_action = best["data"]["action"]
                            self.last_triggered_action = triggered_action

        if now - self.last_phase_match_time > 1.5:
            self.active_phases = {}

        if triggered_action:
            status = f"ACTION: {triggered_action}"

        return frame, status, triggered_action