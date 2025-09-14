#!/usr/bin/env python3
"""
X11-based game loop with a non-blocking, action-queued pipeline for smooth
remote gameplay.

This script decouples input handling from model inference to mitigate network
latency. The main thread handles X11 events and captures user actions at a high 
frequency, placing them in a queue. The rendering loop consumes actions from 
this queue, sends them to the inference pipeline, and displays the resulting 
frame without waiting for real-time user input.
"""

import time
import threading
from collections import deque

import Xlib.Xatom as Xatom
import Xlib.display
import Xlib.X as X
import Xlib.XK as XK
import numpy as np
import torch
import torch.cuda

# Ensure the project root is in the Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.tekken_pipeline import TekkenPipeline


class GameCV:
    """
    Manages the X11 window, input handling, and rendering loop for the game.
    """
    # Mapping from (keysym OR mouse-button) → position in the button vector
    KEYMAP: dict[int, int] = {
        XK.XK_w: 0, # Mapped to 'Up' in buttons_to_actionid
        XK.XK_a: 1, # Mapped to 'Left' in buttons_to_actionid
        XK.XK_s: 2, # Mapped to 'Down' in buttons_to_actionid
        XK.XK_d: 3, # Mapped to 'Right' in buttons_to_actionid
        XK.XK_u: 4, # Mapped to Left Punch
        XK.XK_i: 5, # Mapped to Right Punch
        XK.XK_j: 6, # Mapped to Left Kick
        XK.XK_k: 7, # Mapped to Right Kick
    }

    def __init__(self, width: int = 736, height: int = 448, fps: int = 60):
        self.width, self.height = width, height
        self.target_frame_time = 1.0 / fps

        # --- X11 Setup ---
        try:
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
            self.win.set_wm_name("Tekken Streaming CV")
            self.gc = self.win.create_gc()
            self.win.map()

            # Handle graceful window close
            self.WM_DELETE = self.disp.intern_atom('WM_DELETE_WINDOW')
            self.win.change_property(self.disp.intern_atom('WM_PROTOCOLS'),
                                     Xatom.ATOM, 32, [self.WM_DELETE])
        except Exception as e:
            print(f"Failed to initialize X11 display: {e}")
            raise

        # --- Game State and Pipeline ---
        print("Initializing pipeline...")
        try:
            self.pipeline = TekkenPipeline()
            print("Pipeline initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize pipeline: {e}")
            self.disp.close()
            raise
        
        # State for button presses - protected by lock for thread safety
        self.button_state = [False] * 8 
        self.button_lock = threading.Lock()

        # --- Action Queue and Threading ---
        self.action_queue = deque(maxlen=10)  # Reasonable queue size
        self.last_action_id = 0  # Default to a neutral action
        self.running = True
        
        # Use a separate lock for the action queue
        self.queue_lock = threading.Lock()

        # --- Performance and Profiling ---
        try:
            self.cuda_device = torch.cuda.current_device()
            self.frame_start_event = torch.cuda.Event(enable_timing=True)
            self.pipeline_start_event = torch.cuda.Event(enable_timing=True)
            self.pipeline_end_event = torch.cuda.Event(enable_timing=True)
            self.frame_end_event = torch.cuda.Event(enable_timing=True)
            
            # Pinned memory for faster GPU-CPU transfers
            self.pinned_buffer = torch.zeros((height, width, 3), dtype=torch.uint8, pin_memory=True)
            self.cpu_buffer = np.zeros((height, width, 4), dtype=np.uint8)
            self.cpu_buffer[:, :, 3] = 255  # Set alpha channel once
            
            # Stats tracking
            self.pipe_time_sum = 0.0
            self.total_time_sum = 0.0
            self.frame_counter = 0
            self.stats_start_event = torch.cuda.Event(enable_timing=True)
            self.stats_end_event = torch.cuda.Event(enable_timing=True)
        except Exception as e:
            print(f"Failed to initialize CUDA components: {e}")
            self.disp.close()
            raise
        
        print("GameCV initialization complete. Starting main loop...")

    def _handle_key(self, keysym: int, pressed: bool):
        """Handle key press/release events with thread safety."""
        if pressed and keysym in (XK.XK_Escape, XK.XK_q):
            self.running = False
            return

        if pressed:
            try:
                if keysym == XK.XK_y:
                    print("Re-initializing pipeline buffers...")
                    self.pipeline.init_buffers()
                elif keysym == XK.XK_r and hasattr(self.pipeline, "restart_from_buffer"):
                    print("Restarting from initial buffer...")
                    self.pipeline.restart_from_buffer()
            except Exception as e:
                print(f"Error handling special key {keysym}: {e}")

        if keysym in self.KEYMAP:
            with self.button_lock:
                self.button_state[self.KEYMAP[keysym]] = pressed
            
            # Add the new action to the queue
            action_id = self.buttons_to_actionid()
            with self.queue_lock:
                self.action_queue.append(action_id)

    def _handle_button(self, button: int, pressed: bool):
        """Handle mouse button events. Currently not mapped to any Tekken action."""
        # This is currently not mapped to any Tekken action but is kept for completeness.
        # You can map mouse buttons if needed.
        pass

    def buttons_to_actionid(self) -> int:
        """
        Converts the current button state list into a single integer action ID.
        This mapping matches the logic in the Tekken model files.
        Thread-safe version that acquires the button lock.
        """
        with self.button_lock:
            action_id = 0
            if self.button_state[0]: action_id += 128  # W (Up)
            if self.button_state[2]: action_id += 64   # S (Down)
            if self.button_state[1]: action_id += 32   # A (Left)
            if self.button_state[3]: action_id += 16   # D (Right)
            if self.button_state[4]: action_id += 8    # U (Left Punch)
            if self.button_state[5]: action_id += 4    # I (Right Punch)
            if self.button_state[6]: action_id += 2    # J (Left Kick)
            if self.button_state[7]: action_id += 1    # K (Right Kick)
            return action_id

    def _process_x11_events(self):
        """Process all pending X11 events. Must be called from main thread only."""
        try:
            while self.disp.pending_events():
                ev = self.disp.next_event()
                if ev.type == X.ClientMessage and ev.data[0] == self.WM_DELETE:
                    self.running = False
                elif ev.type in (X.KeyPress, X.KeyRelease):
                    keysym = self.disp.keycode_to_keysym(ev.detail, 0)
                    self._handle_key(keysym, ev.type == X.KeyPress)
                elif ev.type in (X.ButtonPress, X.ButtonRelease):
                    self._handle_button(ev.detail, ev.type == X.ButtonPress)
        except Exception as e:
            print(f"Error processing X11 events: {e}")
            self.running = False

    def _tensor_to_ximage_bytes_optimized(self, frame: torch.Tensor) -> bytes:
        """
        Optimized conversion using pre-allocated pinned memory.
        Converts a [H,W,3] uint8 tensor to a 32-bit RGBX byte string for X11.
        """
        try:
            # Ensure frame is on CPU and the right shape
            if frame.device.type != 'cpu':
                self.pinned_buffer.copy_(frame, non_blocking=True)
                torch.cuda.synchronize()  # Ensure copy is complete
                np_frame = self.pinned_buffer.numpy()
            else:
                np_frame = frame.numpy()
            
            # Validate frame shape
            if len(np_frame.shape) != 3 or np_frame.shape[2] != 3:
                raise ValueError(f"Expected frame shape [H,W,3], got {np_frame.shape}")
            
            self.cpu_buffer[:, :, 0] = np_frame[:, :, 0]  # R <- R
            self.cpu_buffer[:, :, 1] = np_frame[:, :, 1]  # G <- G
            self.cpu_buffer[:, :, 2] = np_frame[:, :, 2]  # B <- B
            return self.cpu_buffer.tobytes()
        except Exception as e:
            print(f"Error converting tensor to X11 format: {e}")
            # Return a black frame as fallback
            self.cpu_buffer.fill(0)
            self.cpu_buffer[:, :, 3] = 255
            return self.cpu_buffer.tobytes()

    def _draw_frame(self, frame: torch.Tensor):
        """Draw frame to X11 window with error handling."""
        try:
            data = self._tensor_to_ximage_bytes_optimized(frame)
            stride = self.width * 4

            # Draw in chunks to avoid X11 request size limits
            CHUNK_ROWS = 32
            for y in range(0, self.height, CHUNK_ROWS):
                h = min(CHUNK_ROWS, self.height - y)
                offset = y * stride
                chunk_data = data[offset:offset + h * stride]
                self.win.put_image(
                    self.gc, 0, y, self.width, h,
                    X.ZPixmap, 24, 0, chunk_data
                )

            self.disp.flush()
        except Exception as e:
            print(f"Error drawing frame: {e}")

    def run(self):
        """Main game loop - handles events and rendering in the main thread."""
        try:
            self.stats_start_event.record()

            while self.running:
                # Process X11 events (input handling)
                self._process_x11_events()
                
                if not self.running:
                    break

                # --- Consume Action from Queue ---
                current_action = self.last_action_id  # Default fallback
                
                with self.queue_lock:
                    if self.action_queue:
                        # Get the most recent action (LIFO for responsiveness)
                        current_action = self.action_queue.pop()
                        # Clear older actions to prevent input lag
                        self.action_queue.clear()
                        self.last_action_id = current_action

                # --- Inference & Render ---
                try:
                    self.frame_start_event.record()
                    
                    self.pipeline_start_event.record()
                    frame, _ = self.pipeline(current_action)
                    self.pipeline_end_event.record()
                    
                    # Validate and process frame
                    if frame is None:
                        print("Warning: Pipeline returned None frame")
                        continue
                        
                    # Handle different frame formats efficiently
                    if len(frame.shape) == 4 and frame.shape[0] == 1:
                        # Remove batch dimension if present
                        frame = frame.squeeze(0)
                    
                    
                    if frame.shape[2] == 4:
                        # If RGBA, take the last 3 channels (assuming first is alpha or unused)
                        frame = frame[:, :, 1:4]
                    
                    # Draw frame without waiting for CUDA sync in the critical path
                    self._draw_frame(frame)
                    
                    self.frame_end_event.record()
                    
                    # Only sync when we need timing measurements
                    if self.frame_counter % 60 == 0:  # Sync every 60 frames for stats
                        torch.cuda.synchronize()
                    
                    # --- Accumulate and Report Stats (less frequently) ---
                    if self.frame_counter % 60 == 0:  # Update stats every 60 frames
                        torch.cuda.synchronize()  # Ensure all events are recorded
                        pipeline_time_ms = self.pipeline_start_event.elapsed_time(self.pipeline_end_event)
                        total_time_ms = self.frame_start_event.elapsed_time(self.frame_end_event)
                        
                        self.pipe_time_sum += pipeline_time_ms
                        self.total_time_sum += total_time_ms
                    
                    self.frame_counter += 1

                    # Report stats less frequently
                    if self.frame_counter % 60 == 0:
                        self.stats_end_event.record()
                        torch.cuda.synchronize()
                        stats_elapsed_ms = self.stats_start_event.elapsed_time(self.stats_end_event)

                        if stats_elapsed_ms >= 1000.0:
                            avg_pipe_time_ms = self.pipe_time_sum / (self.frame_counter // 60)
                            avg_total_time_ms = self.total_time_sum / (self.frame_counter // 60)
                            avg_total_fps = 1000.0 / avg_total_time_ms if avg_total_time_ms > 0 else 0
                            print(
                                f"[CUDA] FPS: {avg_total_fps:5.1f} | "
                                f"GPU Pipeline: {avg_pipe_time_ms:6.1f} ms | "
                                f"Total Frame Time: {avg_total_time_ms:6.1f} ms"
                            )
                            self.stats_start_event.record()
                            self.pipe_time_sum = 0.0
                            self.total_time_sum = 0.0

                except Exception as e:
                    print(f"Error in pipeline processing: {e}")
                    # Continue running even if one frame fails
                    continue

        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt, shutting down...")
        except Exception as e:
            print(f"Fatal error in main loop: {e}")
        finally:
            # --- Cleanup ---
            print("Cleaning up...")
            self.running = False
            try:
                self.disp.close()
            except:
                pass
            print("Cleanup complete.")


if __name__ == "__main__":
    try:
        game = GameCV()
        game.run()
    except Exception as e:
        print(f"Failed to start game: {e}")
        sys.exit(1)