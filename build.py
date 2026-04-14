"""
AeroGestus — build script
Run with:  python build.py

Produces:  dist/AeroGestus.exe  (Windows)
           dist/AeroGestus      (Linux/Mac)

Requirements:
    pip install pyinstaller
"""

import os
import sys
import subprocess
import shutil
import importlib.util


# ── Locate packages ──────────────────

def pkg_path(name):
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise RuntimeError(f"Cannot find package '{name}'. Is it installed?")
    # for namespace packages, submodule_search_locations is a list
    loc = spec.submodule_search_locations
    if loc:
        return list(loc)[0]
    return os.path.dirname(spec.origin)


# ── Data files to bundle ─────────────────────────────────────────
# Format: (source_path, dest_folder_inside_bundle)
sep = ";" if sys.platform == "windows" else ":"

data_files = []

# 1. MediaPipe .task model
if os.path.exists("gesture_recognizer.task"):
    data_files.append(f"gesture_recognizer.task{sep}.")
else:
    print("WARNING: gesture_recognizer.task not found in current directory.")

# 2. gesture_memory.pkl (optional — may not exist yet)
if os.path.exists("gesture_memory.pkl"):
    data_files.append(f"gesture_memory.pkl{sep}.")

# 3. App icon if/when made
for icon_name in ("icon.ico", "icon.png"):
    if os.path.exists(icon_name):
        data_files.append(f"{icon_name}{sep}.")
        break

# 4. MediaPipe data files - important
try:
    mp_path = pkg_path("mediapipe")
    data_files.append(f"{mp_path}{sep}mediapipe")
    print(f"  mediapipe: {mp_path}")
except RuntimeError as e:
    print(f"WARNING: {e}")

# 5. CustomTkinter
try:
    ctk_path = pkg_path("customtkinter")
    data_files.append(f"{ctk_path}{sep}customtkinter")
    print(f"  customtkinter: {ctk_path}")
except RuntimeError as e:
    print(f"WARNING: {e}")

# 6. CLIP model
try:
    clip_path = pkg_path("clip")
    data_files.append(f"{clip_path}{sep}clip")
    print(f"  clip: {clip_path}")
except RuntimeError as e:
    print(f"WARNING: {e}")

hidden_imports = [
    "clip",
    "mediapipe",
    "mediapipe.tasks",
    "mediapipe.tasks.python",
    "mediapipe.tasks.python.vision",
    "customtkinter",
    "PIL",
    "PIL.Image",
    "cv2",
    "torch",
    "torchvision",
    "ftfy",
    "regex",
    "tqdm",
]


icon_arg = []
for icon_name in ("icon.ico", "icon.png"):
    if os.path.exists(icon_name):
        icon_arg = [f"--icon={icon_name}"]
        break


# ── Build command ───
cmd = [
    sys.executable, "-m", "PyInstaller",
    "--noconfirm",
    "--onefile",
    "--windowed",          # no console window — remove this line while debugging!
    "--name=AeroGestus",
    *icon_arg,
    *[f"--add-data={d}" for d in data_files],
    *[f"--hidden-import={h}" for h in hidden_imports],
    # Exclude things that bloat size without helping
    "--exclude-module=matplotlib",
    "--exclude-module=notebook",
    "--exclude-module=IPython",
    "--exclude-module=scipy",
    "app.py",
]

print("\n── Running PyInstaller ──────────────────────────────────────")
print(" ".join(cmd))
print()

result = subprocess.run(cmd)

if result.returncode == 0:
    exe = "dist/AeroGestus.exe" if sys.platform == "win32" else "dist/AeroGestus"
    print(f"\n✔  Build succeeded → {exe}")
    print(   "   Note: first launch will be slow (~10-20s) as PyTorch initialises.")
    print(   "   The file will be large (1-2 GB) — this is normal with PyTorch + CLIP.")
else:
    print("\n✖  Build failed. Check output above for errors.")
    print(   "   Tip: remove --windowed temporarily to see console errors on launch.")
