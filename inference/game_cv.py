#!/usr/bin/env python3
"""
X11-based replacement for the pygame game loop.
  • Creates a 640 × 360 window (or user-supplied size)
  • Collects keyboard / mouse events            → builds 11-button vector
  • Computes scaled mouse deltas                → 2-vector
  • Feeds both to CausvidPipeline               → frame tensor [3,360,640]
  • Converts tensor → 24-bit RGB byte-buffer    → XPutImage
Latency & FPS are printed to stdout every second.
"""
from __future__ import annotations

import Xlib.Xatom as Xatom
import Xlib.display
import Xlib.X as X
import Xlib.XK as XK
import numpy as np
import torch
import time
import torch.cuda

from .causvid_pipeline import CausvidPipeline
from .tekken_pipeline import TekkenPipeline
#from .game import DummyPipeline as CausvidPipeline

class GameCV:
    # Mapping from (keysym OR mouse-button) → position in the 11-button vector
    KEYMAP: dict[int, int] = {
        XK.XK_w: 0,
        XK.XK_a: 1,
        XK.XK_s: 2,
        XK.XK_d: 3,
        XK.XK_u: 4,
        XK.XK_i: 5,
        XK.XK_j: 6,
        XK.XK_k: 7,
        # mouse buttons handled separately (1 & 3)
    }

    def __init__(self, width: int = 736, height: int = 448,
                 mouse_scale: float = 0.01, fps: int = 60):
        self.width, self.height = width, height
        self.mouse_scale = mouse_scale
        self.target_frame_time = 1.0 / fps

        # X11 setup ----------------------------------------------------------
        self.disp = Xlib.display.Display()
        self.screen = self.disp.screen()
        self.win = self.screen.root.create_window(
            0, 0, width, height, 0,
            self.screen.root_depth,
            X.InputOutput,
            X.CopyFromParent,
            background_pixel=self.screen.black_pixel,
            event_mask=(X.ExposureMask | X.KeyPressMask | X.KeyReleaseMask |
                        X.ButtonPressMask | X.ButtonReleaseMask |
                        X.PointerMotionMask | X.StructureNotifyMask)
        )
        self.win.set_wm_name("Causvid Game - X11")
        self.gc = self.win.create_gc()
        self.win.map()

        # Handle graceful close via WM_DELETE_WINDOW
        self.WM_DELETE = self.disp.intern_atom('WM_DELETE_WINDOW')
        self.win.change_property(self.disp.intern_atom('WM_PROTOCOLS'),
                                 Xatom.ATOM, 32, [self.WM_DELETE])

        # Game state ---------------------------------------------------------
        print("Initializing pipeline...")
        self.pipeline = TekkenPipeline()
        print("Pipeline initialized successfully")
        
        self.button_state = [False] * 11
        self.last_mouse_pos: tuple[int, int] | None = None
        self.running = True
        
        print("GameCV initialization complete, starting main loop...")

        # Stats
        self.pipe_fps_sum  = 0.0     # pipeline-only
        self.total_fps_sum = 0.0     # pipeline + draw
        self.frame_counter = 0
        self.stats_t0      = time.time()

    # --------------------------------------------------------------------- #
    # Input Handling
    # --------------------------------------------------------------------- #
    def _handle_key(self, keysym: int, pressed: bool):
        # Quit on Escape / q
        if pressed and keysym in (XK.XK_Escape, XK.XK_q):
            self.running = False
            return

        # Pipeline control keys
        if pressed:
            if keysym == XK.XK_y:
                self.pipeline.init_buffers()
            elif keysym == XK.XK_r and hasattr(self.pipeline, "restart_from_buffer"):
                self.pipeline.restart_from_buffer()
            # Use up and down arrows to change mouse scaler
            if keysym == XK.XK_Up:
                if hasattr(self.pipeline, 'mouse_scaler'):
                    self.pipeline.mouse_scaler += 0.01
                    print(f"Mouse scaler: {self.pipeline.mouse_scaler}")
            elif keysym == XK.XK_Down:
                if hasattr(self.pipeline, 'mouse_scaler'):
                    self.pipeline.mouse_scaler -= 0.01
                    print(f"Mouse scaler: {self.pipeline.mouse_scaler}")

        # Regular movement / action keys
        if keysym in self.KEYMAP:
            self.button_state[self.KEYMAP[keysym]] = pressed

    def _handle_button(self, button: int, pressed: bool):
        if button == 1:          # Left mouse
            self.button_state[9] = pressed
        elif button == 3:        # Right mouse
            self.button_state[10] = pressed

    def _mouse_delta(self) -> list[float]:
        ptr = self.win.query_pointer()
        pos = (ptr.win_x, ptr.win_y)
        if self.last_mouse_pos is None:
            self.last_mouse_pos = pos
            return [0.0, 0.0]

        dx = (pos[0] - self.last_mouse_pos[0]) * self.mouse_scale
        dy = (pos[1] - self.last_mouse_pos[1]) * self.mouse_scale
        self.last_mouse_pos = pos
        # Clamp to [-1,1]
        dx = max(-1.0, min(1.0, dx))
        dy = max(-1.0, min(1.0, dy))
        return [dx, dy]

    # --------------------------------------------------------------------- #
    # Rendering helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _tensor_to_ximage_bytes(frame: torch.Tensor, width: int, height: int) -> bytes:
        """
        Converts [H,W,3] uint8 tensor → little-endian 32-bit RGBX byte string for X11.
        """
        # Ensure frame is the right shape and type
        if frame.dim() == 3:
            # frame is [H, W, 3] - this is correct
            pass
        else:
            raise ValueError(f"Expected [H,W,3] tensor, got shape {frame.shape}")
        
        # Ensure it's uint8
        if frame.dtype != torch.uint8:
            frame = frame.to(torch.uint8)
        
        # Convert to numpy for easier manipulation
        np_frame = frame.cpu().numpy()
        
        # Verify dimensions match expected window size
        if np_frame.shape[0] != height or np_frame.shape[1] != width:
            print(f"Warning: Frame size {np_frame.shape[:2]} doesn't match window size {(height, width)}")
        
        # Create 32-bit RGBX buffer (X11 expects 32-bit alignment)
        rgbx_buffer = np.zeros((height, width, 4), dtype=np.uint8)
        rgbx_buffer[:, :, 0] = np_frame[:, :, 0]  # B
        rgbx_buffer[:, :, 1] = np_frame[:, :, 1]  # G  
        rgbx_buffer[:, :, 2] = np_frame[:, :, 2]  # R
        rgbx_buffer[:, :, 3] = 255                # X (padding)
        
        return rgbx_buffer.tobytes()

    def _draw_frame(self, frame: torch.Tensor):
        try:
            # Convert frame to the format X11 expects
            data = self._tensor_to_ximage_bytes(frame, self.width, self.height)
            BPP = 4  # bytes per pixel (RGBX)
            stride = self.width * BPP

            # Use smaller chunks to avoid BadLength errors
            CHUNK_ROWS = 32  # Reduced from 64 to be more conservative
            for y in range(0, self.height, CHUNK_ROWS):
                h = min(CHUNK_ROWS, self.height - y)
                offset = y * stride
                chunk_data = data[offset:offset + h * stride]
                
                # Ensure chunk size is reasonable (X11 has limits around 256KB per request)
                if len(chunk_data) > 200000:  # 200KB safety margin
                    # Split into even smaller chunks if needed
                    mini_chunk_rows = max(1, min(h, 200000 // stride))
                    for mini_y in range(0, h, mini_chunk_rows):
                        mini_h = min(mini_chunk_rows, h - mini_y)
                        mini_offset = offset + mini_y * stride
                        mini_chunk_data = data[mini_offset:mini_offset + mini_h * stride]
                        
                        self.win.put_image(
                            self.gc,
                            0, y + mini_y,           # dest x,y
                            self.width, mini_h,
                            X.ZPixmap,
                            24,                      # depth
                            0,                       # left pad
                            mini_chunk_data
                        )
                else:
                    self.win.put_image(
                        self.gc,
                        0, y,                    # dest x,y
                        self.width, h,
                        X.ZPixmap,
                        24,                      # depth
                        0,                       # left pad
                        chunk_data
                    )

            self.disp.flush()
        except Exception as e:
            print(f"Error drawing frame: {e}")
            # Continue without crashing

    def buttons_to_actionid(self) -> int:
        """
        Converts the current button state to a single integer action ID.
        This is a placeholder and should be adapted based on the specific game.
        """
        # Example mapping for demonstration purposes
        action_id = 0
        if self.button_state[0]:  # W
            action_id += 128
        if self.button_state[1]:  # A
            action_id += 32
        if self.button_state[2]:  # S
            action_id += 64
        if self.button_state[3]:  # D
            action_id += 16
        if self.button_state[4]:  # U left punch
            action_id += 8
        if self.button_state[5]:  # I right punch
            action_id += 4
        if self.button_state[6]:  # J left kick
            action_id += 2
        if self.button_state[7]:  # K right kick
            action_id += 1
        # Add more mappings as needed
        return action_id

    # --------------------------------------------------------------------- #
    # Main loop
    # --------------------------------------------------------------------- #
    def run(self):
        while self.running:
            # ---------------- Event processing --------------------------- #
            while self.disp.pending_events():
                ev = self.disp.next_event()
                if ev.type == X.ClientMessage and ev.data[0] == self.WM_DELETE:
                    self.running = False
                elif ev.type in (X.KeyPress, X.KeyRelease):
                    keysym = self.disp.keycode_to_keysym(ev.detail, 0)
                    self._handle_key(keysym, ev.type == X.KeyPress)
                elif ev.type in (X.ButtonPress, X.ButtonRelease):
                    self._handle_button(ev.detail, ev.type == X.ButtonPress)
                # Ignore MotionNotify; we poll pointer each frame

            # ---------------- Inference & Render ------------------------- #
            mouse_delta  = self._mouse_delta()
            mouse_tensor = torch.tensor(mouse_delta, dtype=torch.bfloat16, device='cuda')
            btn_tensor   = torch.tensor(self.button_state, dtype=torch.bool,  device='cuda')

            t_frame_start = time.time()

            # --- pipeline ------------------------------------------------ #
            # action_id = 0  # Dummy action ID for now
            action_id = self.buttons_to_actionid()
            frame, pipe_time = self.pipeline(action_id)
            
            # # The pipeline returns [H,W,3] uint8 BGR, we need to ensure correct format
            # if frame.dim() == 3 and frame.shape[-1] == 3:
            #     # Frame is already [H,W,3], which is what we expect
            #     display_frame = frame
            # else:
            #     print(f"Unexpected frame shape: {frame.shape}")
            #     continue
            display_frame = frame[:, :, 1:4]  # Ensure it's [H,W,3]
            # --- draw ---------------------------------------------------- #
            t1 = time.time()
            self._draw_frame(display_frame)
            draw_time = time.time() - t1                       # seconds

            # --- accumulate stats --------------------------------------- #
            total_time = time.time() - t_frame_start           # = pipe_time+draw_time (+ε)
            self.pipe_fps_sum  += 1.0 / max(pipe_time,  1e-6)
            self.total_fps_sum += 1.0 / max(total_time, 1e-6)
            self.frame_counter += 1

            # ---------------- Statistics -------------------------------- #
            now = time.time()
            if now - self.stats_t0 >= 1.0:
                avg_pipe_fps  = self.pipe_fps_sum  / max(self.frame_counter, 1)
                avg_total_fps = self.total_fps_sum / max(self.frame_counter, 1)
                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"FPS (total): {avg_total_fps:5.1f} | "
                      f"FPS (pipeline): {avg_pipe_fps:5.1f} | "
                      f"Latency pipeline: {pipe_time*1000:6.1f} ms | "
                      f"Latency draw: {draw_time*1000:6.1f} ms")

                self.stats_t0      = now
                self.pipe_fps_sum  = 0.0
                self.total_fps_sum = 0.0
                self.frame_counter = 0

        self.disp.close()


# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    game = GameCV()     
    game.run()