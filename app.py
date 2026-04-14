import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import threading
import time
from engine import AeroEngine
from executor import OSExecutor, ACTIONS

MOTION_TYPES = {
    "Swipe Right":   ("swipe_h", "right"),
    "Swipe Left":    ("swipe_h", "left"),
    "Swipe Up":      ("swipe_v", "up"),
    "Swipe Down":    ("swipe_v", "down"),
    "Flip to Front": ("flip",    "front"),
    "Flip to Back":  ("flip",    "back"),
}

BG      = "#F4F4F5"
PANEL   = "#FFFFFF"
BORDER  = "#D4D4D8"
ACCENT  = "#18181B"
TEXT    = "#18181B"
TEXT2   = "#52525B"
SILVER  = "#A1A1AA"

FONT_HEAD  = ("Georgia", 22, "bold")
FONT_BODY  = ("Helvetica Neue", 14)
FONT_MONO  = ("Courier New", 13)
FONT_LABEL = ("Helvetica Neue", 12)


class AeroGestusApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AeroGestus")
        self.geometry("1120x640")
        self.configure(fg_color=BG)
        ctk.set_appearance_mode("light")

        self.engine         = AeroEngine()
        self.executor       = OSExecutor()
        self.ui_state       = "MAIN"
        self.last_raw_frame = None
        self.running        = True
        self.camera_on      = False
        self.cap            = None

        self._reload_motion_gestures()
        self._setup_layout()
        threading.Thread(target=self.video_loop, daemon=True).start()

    def _reload_motion_gestures(self):
        for name, data in self.engine.gesture_db.items():
            if data.get("motion"):
                self.engine.motion_detector.register(
                    name, data["action"], data["motion_type"], data["motion_dir"])

    def _setup_layout(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        vf = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=14,
                          border_width=1, border_color=BORDER)
        vf.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.video_label = ctk.CTkLabel(vf, text="Camera is off", text_color=SILVER, font=FONT_BODY)
        self.video_label.pack(expand=True, fill="both", padx=8, pady=8)

        self.sidebar = ctk.CTkFrame(self, fg_color=PANEL, width=310,
                                    corner_radius=14, border_width=1,
                                    border_color=BORDER)
        self.sidebar.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self._refresh()

    def _refresh(self):
        for w in self.sidebar.winfo_children():
            w.destroy()
        {"MAIN": self._ui_main, "RECORDING": self._ui_recorder,
         "MODULE": self._ui_module}.get(self.ui_state, self._ui_main)()

    def _set(self, state):
        self.ui_state = state
        self._refresh()

    # ── widget helpers ───────────────────────────────────────────
    def _divider(self, parent):
        ctk.CTkFrame(parent, height=1, fg_color=BORDER).pack(
            fill="x", padx=16, pady=6)

    def _entry(self, parent, placeholder):
        e = ctk.CTkEntry(parent, placeholder_text=placeholder,
                         fg_color="#FAFAFA", border_color=BORDER,
                         text_color=TEXT, placeholder_text_color=SILVER,
                         font=FONT_BODY, height=36, corner_radius=8)
        e.pack(pady=4, padx=16, fill="x")
        return e

    def _dropdown(self, parent, values, variable=None, default=None):
        om = ctk.CTkOptionMenu(
            parent, values=values, variable=variable,
            fg_color="#FAFAFA", button_color=BORDER,
            button_hover_color="#E4E4E7", text_color=TEXT,
            dropdown_fg_color=PANEL, dropdown_text_color=TEXT,
            font=FONT_BODY, corner_radius=8)
        if default:
            om.set(default)
        om.pack(pady=4, padx=16, fill="x")
        return om

    def _btn(self, parent, text, cmd, primary=True):
        ctk.CTkButton(
            parent, text=text, command=cmd,
            fg_color=ACCENT if primary else "transparent",
            hover_color="#3F3F46" if primary else "#F4F4F5",
            text_color="white" if primary else TEXT2,
            border_width=0,
            font=FONT_BODY, corner_radius=8, height=36
        ).pack(pady=3, padx=16, fill="x")

    def _small_label(self, parent, text):
        ctk.CTkLabel(parent, text=text, font=FONT_LABEL,
                     text_color=SILVER).pack(anchor="w", padx=18, pady=(8, 2))

    # ── MAIN ─────────────────────────────────────────────────────
    def _ui_main(self):
        sb = self.sidebar

        hdr = ctk.CTkFrame(sb, fg_color="transparent")
        hdr.pack(fill="x", padx=16, pady=(20, 2))

        # Try to show the window icon beside the name
        try:
            raw = Image.open("icon.ico").convert("RGBA").resize((24, 24), Image.Resampling.LANCZOS)
            self._logo_img = ctk.CTkImage(light_image=raw, size=(24, 24))
            ctk.CTkLabel(hdr, image=self._logo_img, text="").pack(side="left", padx=(0, 8))
        except Exception:
            pass  # no icon file — just skip silently

        ctk.CTkLabel(hdr, text="AeroGestus", font=FONT_HEAD,
                     text_color=TEXT).pack(side="left")
        ctk.CTkLabel(hdr, text="beta", font=FONT_LABEL,
                     text_color=SILVER).pack(side="left", padx=(6, 0), pady=(4, 0))

        self.status_label = ctk.CTkLabel(sb, text="● IDLE",
                                         font=FONT_MONO, text_color=SILVER)
        self.status_label.pack(anchor="w", padx=18, pady=(0, 8))

        self._divider(sb)

        # Module row
        mod_row = ctk.CTkFrame(sb, fg_color="transparent")
        mod_row.pack(fill="x", padx=16, pady=(4, 0))
        ctk.CTkLabel(mod_row, text="MODULE", font=FONT_LABEL,
                     text_color=SILVER).pack(side="left")
        ctk.CTkButton(mod_row, text="+ new", fg_color="transparent",
                      text_color=SILVER, hover_color="#F4F4F5",
                      font=FONT_LABEL, height=20, width=48,
                      command=lambda: self._set("MODULE")).pack(side="right")

        modules = self._module_names()
        self.mod_var = ctk.StringVar(value=getattr(self.engine, "active_module", modules[0]))
        self._dropdown(sb, modules, variable=self.mod_var)
        self.mod_var.trace_add("write", lambda *_: self._on_mod_change())

        self._divider(sb)

        ctk.CTkLabel(sb, text="GESTURES", font=FONT_LABEL,
                     text_color=SILVER).pack(anchor="w", padx=18, pady=(4, 2))

        self.listbox = ctk.CTkTextbox(
            sb, fg_color="#FAFAFA", border_color=BORDER, border_width=1,
            height=200, text_color=TEXT, font=FONT_MONO, corner_radius=8)
        self.listbox.pack(pady=4, padx=16, fill="x")
        self._update_list()

        self._divider(sb)
        self._btn(sb, "+ Add gesture", lambda: self._set("RECORDING"))

        cam_text = "Turn off camera" if self.camera_on else "Turn on camera"
        self.cam_btn = ctk.CTkButton(
            sb, text=cam_text, command=self._toggle_camera,
            fg_color="transparent", hover_color="#F4F4F5",
            text_color=TEXT2, border_width=0,
            font=FONT_BODY, corner_radius=8, height=36)
        self.cam_btn.pack(pady=3, padx=16, fill="x")

    def _module_names(self):
        names = sorted({d.get("module", "Default")
                        for d in self.engine.gesture_db.values()})
        if not names:
            names = ["Default"]
        if "Default" not in names:
            names.insert(0, "Default")
        return names

    def _on_mod_change(self):
        self.engine.active_module = self.mod_var.get()
        self._update_list()

    def _update_list(self):
        self.listbox.delete("0.0", "end")
        active = getattr(self.engine, "active_module", "Default")
        found  = False
        for name, data in self.engine.gesture_db.items():
            if data.get("module", "Default") != active:
                continue
            found = True
            icon = "↔" if data.get("motion") else ("▶" if data.get("is_temporal") else "◆")
            self.listbox.insert("end", f" {icon}  {name}  →  {data['action']}\n")
        if not found:
            self.listbox.insert("end", "  no gestures in this module yet\n")

    # ── RECORDER ─────────────────────────────────────────────────
    def _ui_recorder(self):
        sb = self.sidebar

        ctk.CTkLabel(sb, text="Add gesture", font=FONT_HEAD,
                     text_color=TEXT).pack(anchor="w", padx=16, pady=(20, 2))
        ctk.CTkLabel(sb, text="Static, temporal, or motion-based.",
                     font=FONT_LABEL, text_color=SILVER).pack(
                         anchor="w", padx=18, pady=(0, 6))

        self._divider(sb)

        # Type toggle
        self._small_label(sb, "TYPE")
        self.rec_type = ctk.StringVar(value="Static")
        ctk.CTkSegmentedButton(
            sb, values=["Static", "Motion"],
            variable=self.rec_type,
            selected_color=ACCENT, selected_hover_color="#3F3F46",
            unselected_color="#E8E4DF", unselected_hover_color="#DDD8D2",
            text_color="white", text_color_disabled="#3F3F46",
            font=FONT_BODY,
            command=self._on_type_change
        ).pack(pady=4, padx=16, fill="x")

        # Name
        self._small_label(sb, "NAME")
        self.name_in = self._entry(sb, "e.g. peace_sign, swipe_next …")

        # Module tag
        self._small_label(sb, "MODULE")
        mods = self._module_names()
        current_mod = getattr(self.engine, "active_module", mods[0])
        self.rec_mod_var = ctk.StringVar(value=current_mod)
        self._dropdown(sb, mods, variable=self.rec_mod_var)

        # Action
        self._small_label(sb, "ACTION")
        self.act_var = ctk.StringVar(value="Play/Pause")
        self._dropdown(sb, list(ACTIONS.keys()), variable=self.act_var)

        # Dynamic section
        self.dyn = ctk.CTkFrame(sb, fg_color="transparent")
        self.dyn.pack(fill="x")
        self._dyn_static()

        self._divider(sb)

        self.rec_btn = ctk.CTkButton(
            sb, text="Capture", command=self._do_capture,
            fg_color=ACCENT, hover_color="#3F3F46", text_color="white",
            font=FONT_BODY, corner_radius=8, height=36)
        self.rec_btn.pack(pady=3, padx=16, fill="x")
        self._btn(sb, "Cancel", lambda: self._set("MAIN"), primary=False)

    def _on_type_change(self, val):
        for w in self.dyn.winfo_children():
            w.destroy()
        if val == "Static":
            self._dyn_static()
        else:
            self._dyn_motion()

    def _dyn_static(self):
        ctk.CTkLabel(self.dyn, text="CAPTURE MODE", font=FONT_LABEL,
                     text_color=SILVER).pack(anchor="w", padx=18, pady=(8, 2))
        self.cap_mode = ctk.StringVar(value="video")
        ctk.CTkSegmentedButton(
            self.dyn, values=["image", "video"],
            variable=self.cap_mode,
            selected_color=ACCENT, selected_hover_color="#3F3F46",
            unselected_color="#E8E4DF", unselected_hover_color="#DDD8D2",
            text_color="white", text_color_disabled="#3F3F46",
            font=FONT_BODY
        ).pack(pady=4, padx=16, fill="x")
        self.prog = ctk.CTkProgressBar(
            self.dyn, progress_color=ACCENT,
            fg_color=BORDER, corner_radius=4, height=5)
        self.prog.set(0)
        self.prog.pack(pady=(6, 2), padx=16, fill="x")

    def _dyn_motion(self):
        ctk.CTkLabel(self.dyn, text="MOTION TEMPLATE", font=FONT_LABEL,
                     text_color=SILVER).pack(anchor="w", padx=18, pady=(8, 2))
        self.mot_type_var = ctk.StringVar(value=list(MOTION_TYPES.keys())[0])
        self._dropdown(self.dyn, list(MOTION_TYPES.keys()),
                       variable=self.mot_type_var)
        ctk.CTkLabel(self.dyn,
                     text="No capture needed — direction is detected\nfrom landmark deltas at runtime.",
                     font=FONT_LABEL, text_color=SILVER,
                     wraplength=265, justify="left").pack(
                         anchor="w", padx=18, pady=(4, 0))

    def _do_capture(self):
        if self.rec_type.get() == "Motion":
            self._register_motion()
        else:
            threading.Thread(target=self._capture_worker, daemon=True).start()

    def _register_motion(self):
        name   = self.name_in.get().strip()
        action = self.act_var.get()
        module = self.rec_mod_var.get()
        if not name:
            return
        gtype, direction = MOTION_TYPES[self.mot_type_var.get()]
        self.engine.register_motion_gesture(name, action, gtype, direction,
                                            module=module)
        self._set("MAIN")

    def _capture_worker(self):
        name   = self.name_in.get().strip()
        action = self.act_var.get()
        module = self.rec_mod_var.get()
        mode   = self.cap_mode.get()
        if not name:
            return
        self.rec_btn.configure(state="disabled", text="Capturing…")
        frames, n = [], 20 if mode == "video" else 1
        for i in range(n):
            if self.last_raw_frame is not None:
                frames.append(self.last_raw_frame)
            self.prog.set((i + 1) / n)
            time.sleep(0.1)
        self.engine.record_gesture(frames, name, action, module=module)
        self._set("MAIN")

    # ── MODULE creator ───────────────────────────────────────────
    def _ui_module(self):
        sb = self.sidebar
        ctk.CTkLabel(sb, text="New module", font=FONT_HEAD,
                     text_color=TEXT).pack(anchor="w", padx=16, pady=(20, 4))
        ctk.CTkLabel(sb,
                     text="Group gestures by context — e.g. Media, Browser, Gaming.\n"
                          "Assign gestures to a module when recording.",
                     font=FONT_LABEL, text_color=SILVER,
                     wraplength=265, justify="left").pack(
                         anchor="w", padx=16, pady=(0, 10))
        self._divider(sb)
        self.mod_in = self._entry(sb, "Module name…")
        self._btn(sb, "Create", self._create_module)
        self._btn(sb, "Back", lambda: self._set("MAIN"), primary=False)

    def _create_module(self):
        name = self.mod_in.get().strip()
        if name:
            self.engine.active_module = name
            self._set("MAIN")

    # ── Camera toggle ─────────────────────────────────────────────
    def _toggle_camera(self):
        if not self.camera_on:
            self.cap = cv2.VideoCapture(0)
            self.camera_on = True
            self.engine.scanning_enabled = False
            try:
                self.cam_btn.configure(text="Turn off camera")
            except Exception:
                pass
        else:
            self.camera_on = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.engine.scanning_enabled = False
            self.engine.motion_detector.clear()
            self._show_placeholder()
            try:
                self.cam_btn.configure(text="Turn on camera")
                self._safe_status("IDLE")
            except Exception:
                pass

    def _show_placeholder(self):
        try:
            self.video_label.imgtk = None
            self.video_label.configure(image="", text="Camera is off",
                                       text_color=SILVER, font=FONT_BODY)
        except Exception:
            pass

    # ── Video loop ───────────────────────────────────────────────
    def video_loop(self):
        while self.running:
            if not self.camera_on or self.cap is None:
                time.sleep(0.05)
                continue
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            display = cv2.flip(frame, 1)
            self.last_raw_frame = display.copy()

            if self.ui_state == "MAIN":
                res, status, act = self.engine.process_frame(display)
                if act:
                    threading.Thread(
                        target=self.executor.press_key,
                        args=(ACTIONS.get(act),),
                        daemon=True
                    ).start()
                self._safe_status(status)
            else:
                res = display

            img = Image.fromarray(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
            img = img.resize((760, 520), Image.Resampling.LANCZOS)
            tk  = ImageTk.PhotoImage(img)
            try:
                self.video_label.configure(image=tk, text="")
                self.video_label.imgtk = tk
            except Exception:
                break
            time.sleep(0.01)

    def _safe_status(self, txt):
        try:
            col = "#16A34A" if ("ACTION" in txt or "MOTION" in txt) else SILVER
            self.status_label.configure(text=f"● {txt}", text_color=col)
        except Exception:
            pass


if __name__ == "__main__":
    app = AeroGestusApp()
    app.mainloop()