import numpy as np
import cv2
import os
import argparse
import time
import imageio

# Button mappings - order: [Up, Down, Left, Right, Square, Triangle, Cross, Circle]
BUTTON_NAMES = ['Up', 'Down', 'Left', 'Right', 'Square', 'Triangle', 'Cross', 'Circle']
BUTTON_SYMBOLS = ['↑', '↓', '←', '→', '□', '△', '✕', '○']

def save_video_imageio(video_np, output_path, fps=30):
    """
    Save video frames using the imageio library.

    Args:
        video_np: numpy array of shape [t, h, w, c] with uint8 values 0-255
        output_path: path to save the video (e.g., 'video.mp4')
        fps: frames per second
    """
    try:
        with imageio.get_writer(output_path, fps=fps, macro_block_size=1) as writer:
            for frame in video_np:
                writer.append_data(frame)
        print(f"Video saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving video with imageio: {e}")

def action_id_to_buttons(action_id):
    """Convert action ID to 8-bit button representation."""
    if isinstance(action_id, (list, np.ndarray)) and len(action_id) > 1:
        # Already in button format
        return action_id
    
    # Convert single ID to 8-bit representation
    action_id = int(action_id)
    buttons = []
    for i in range(8):
        buttons.append((action_id >> i) & 1)
    return buttons

def get_active_buttons(action_id):
    """Get list of active button names from action ID."""
    buttons = action_id_to_buttons(action_id)
    active = []
    for i, pressed in enumerate(buttons):
        if pressed:
            active.append(BUTTON_NAMES[i])
    return active if active else ['Neutral']

def draw_button_overlay(frame, action_id, frame_idx, total_frames, show_legend=False):
    """Draw button overlay on the frame."""
    height, width = frame.shape[:2]
    
    # Create extended frame with button overlay area
    overlay_height = 100
    extended_frame = np.zeros((height + overlay_height, width, 3), dtype=np.uint8)
    extended_frame[:height] = frame
    
    # Get button states
    buttons = action_id_to_buttons(action_id)
    
    # Button layout positions (relative to bottom area)
    button_size = 60
    button_spacing = 10
    
    # Calculate starting position to center buttons
    total_width = len(BUTTON_SYMBOLS) * button_size + (len(BUTTON_SYMBOLS) - 1) * button_spacing
    start_x = (width - total_width) // 2
    button_y = height + 20
    
    # Draw each button
    for i, (symbol, name, pressed) in enumerate(zip(BUTTON_SYMBOLS, BUTTON_NAMES, buttons)):
        x = start_x + i * (button_size + button_spacing)
        
        # Button colors
        if pressed:
            button_color = (0, 255, 0)  # Green when pressed
            text_color = (0, 0, 0)      # Black text
        else:
            button_color = (64, 64, 64)  # Dark gray when not pressed
            text_color = (255, 255, 255) # White text
        
        # Draw button background
        cv2.rectangle(extended_frame, (x, button_y), (x + button_size, button_y + button_size), button_color, -1)
        cv2.rectangle(extended_frame, (x, button_y), (x + button_size, button_y + button_size), (255, 255, 255), 2)
        
        # Draw button symbol
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        # Get text size to center it
        text_size = cv2.getTextSize(symbol, font, font_scale, thickness)[0]
        text_x = x + (button_size - text_size[0]) // 2
        text_y = button_y + (button_size + text_size[1]) // 2
        
        cv2.putText(extended_frame, symbol, (text_x, text_y), font, font_scale, text_color, thickness)
        
        # Draw button name below
        name_font_scale = 0.4
        name_thickness = 1
        name_size = cv2.getTextSize(name, font, name_font_scale, name_thickness)[0]
        name_x = x + (button_size - name_size[0]) // 2
        name_y = button_y + button_size + 15
        
        cv2.putText(extended_frame, name, (name_x, name_y), font, name_font_scale, (255, 255, 255), name_thickness)
    
    # Add action info text
    active_buttons = get_active_buttons(action_id)
    if len(active_buttons) == 1 and active_buttons[0] == 'Neutral':
        action_text = f"Action ID: {action_id} (Neutral)"
    else:
        action_text = f"Action ID: {action_id} ({', '.join(active_buttons)})"
    
    frame_text = f"Frame {frame_idx+1}/{total_frames}"
    
    # Draw action text
    info_y = height + overlay_height - 25
    cv2.putText(extended_frame, action_text, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(extended_frame, frame_text, (width - 150, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw legend if requested
    if show_legend:
        legend_x = 10
        legend_y = 30
        cv2.putText(extended_frame, "Legend: Green=Pressed, Gray=Not Pressed", 
                   (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for i, (symbol, name) in enumerate(zip(BUTTON_SYMBOLS, BUTTON_NAMES)):
            text_y = legend_y + 20 + i * 15
            legend_text = f"{symbol} = {name}"
            cv2.putText(extended_frame, legend_text, 
                       (legend_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return extended_frame

def play_npz_video(npz_path, fps=30, loop=False):
    """
    Play video from NPZ file with action IDs overlaid at the bottom.
    
    Args:
        npz_path (str): Path to the NPZ file
        fps (int): Frames per second for playback
        loop (bool): Whether to loop the video
    """
    print(f"Loading NPZ file: {npz_path}")
    
    # Load NPZ file
    data = np.load(npz_path)
    
    # Check available keys
    print(f"Available keys in NPZ: {list(data.keys())}")
    
    # Get video frames
    if 'images' in data:
        images = data['images']
    else:
        print("No 'images' key found in NPZ file")
        return
    
    # Get action data
    actions = None
    if 'actions_p1' in data:
        actions = data['actions_p1']
        print(f"Found actions_p1 with shape: {actions.shape}")
    elif 'actions' in data:
        actions = data['actions']
        print(f"Found actions with shape: {actions.shape}")
    
    # Get attention mask to find valid frames
    if 'attention_mask' in data:
        mask = data['attention_mask']
        end_idx = int(np.where(mask == 1)[0][-1]) + 1
        print(f"Using attention_mask, valid frames: {end_idx}")
    elif 'valid_frames' in data:
        mask = data['valid_frames']
        end_idx = int(np.where(mask == 1)[0][-1]) + 1
        print(f"Using valid_frames, valid frames: {end_idx}")
    else:
        end_idx = len(images)
        print(f"No mask found, using all frames: {end_idx}")
    
    # Trim to valid frames
    images = images[:end_idx]
    if actions is not None:
        actions = actions[:end_idx]
    
    print(f"Video shape: {images.shape}")
    if actions is not None:
        print(f"Actions shape: {actions.shape}")
    
    # Calculate frame delay for desired FPS
    frame_delay = 1.0 / fps
    
    # Create window
    cv2.namedWindow('Tekken Video with Actions', cv2.WINDOW_AUTOSIZE)
    
    frame_idx = 0
    paused = False
    
    print(f"Playing video with {len(images)} frames at {fps} FPS")
    print("Controls:")
    print("  SPACE: Pause/Resume")
    print("  LEFT/RIGHT: Previous/Next frame (when paused)")
    print("  ESC/Q: Quit")
    print("  R: Restart")
    print("  L: Toggle button legend")
    print("\nButton Legend:")
    for symbol, name in zip(BUTTON_SYMBOLS, BUTTON_NAMES):
        print(f"  {symbol} = {name}")
    print("  Green = Pressed, Gray = Not Pressed")
    
    show_legend = False
    
    while True:
        start_time = time.time()
        
        # Get current frame
        frame = images[frame_idx].copy()
        
        # Convert to uint8 if necessary
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Add button overlay
        if actions is not None and frame_idx < len(actions):
            action_val = actions[frame_idx]
            display_frame = draw_button_overlay(frame, action_val, frame_idx, len(images), show_legend)
        else:
            # No action data, just add frame counter
            height, width = frame.shape[:2]
            bar_height = 30
            display_frame = np.zeros((height + bar_height, width, 3), dtype=np.uint8)
            display_frame[:height] = frame
            
            frame_text = f"Frame {frame_idx+1}/{len(images)} (No action data)"
            cv2.putText(display_frame, frame_text, (10, height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Tekken Video with Actions', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # ESC or Q
            break
        elif key == ord(' '):  # SPACE
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('r'):  # R
            frame_idx = 0
            print("Restarted")
        elif key == ord('l'):  # L
            show_legend = not show_legend
            print("Legend:", "ON" if show_legend else "OFF")
        elif paused:
            if key == 81 or key == 2:  # LEFT arrow
                frame_idx = max(0, frame_idx - 1)
            elif key == 83 or key == 3:  # RIGHT arrow
                frame_idx = min(len(images) - 1, frame_idx + 1)
        
        # Advance frame if not paused
        if not paused:
            frame_idx += 1
            
            # Loop or stop at end
            if frame_idx >= len(images):
                if loop:
                    frame_idx = 0
                    print("Looping video...")
                else:
                    print("Video finished")
                    break
        
        # Maintain FPS
        elapsed = time.time() - start_time
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)
    
    cv2.destroyAllWindows()
    data.close()

def list_npz_files(directory):
    """List all NPZ files in a directory structure."""
    npz_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.npz'):
                npz_files.append(os.path.join(root, file))
    return sorted(npz_files)

def main():
    parser = argparse.ArgumentParser(description='Visualize Tekken NPZ videos with action overlays')
    parser.add_argument('input', help='Path to NPZ file or directory containing NPZ files')
    parser.add_argument('--fps', type=int, default=30, help='Playback FPS (default: 30)')
    parser.add_argument('--loop', action='store_true', help='Loop the video')
    parser.add_argument('--list', action='store_true', help='List available NPZ files and exit')
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input) and args.input.endswith('.npz'):
        # Single NPZ file
        if args.list:
            print(f"Single file: {args.input}")
            return
        play_npz_video(args.input, fps=args.fps, loop=args.loop)
        
    elif os.path.isdir(args.input):
        # Directory with NPZ files
        npz_files = list_npz_files(args.input)
        
        if not npz_files:
            print(f"No NPZ files found in {args.input}")
            return
        
        if args.list:
            print(f"Found {len(npz_files)} NPZ files:")
            for i, file in enumerate(npz_files):
                print(f"  {i+1}: {file}")
            return
        
        print(f"Found {len(npz_files)} NPZ files")
        
        # Play each file
        for i, npz_file in enumerate(npz_files):
            print(f"\n--- Playing file {i+1}/{len(npz_files)}: {os.path.basename(npz_file)} ---")
            play_npz_video(npz_file, fps=args.fps, loop=False)
            
            # Ask to continue if not the last file
            if i < len(npz_files) - 1:
                response = input("Continue to next video? (y/n/q): ").lower()
                if response == 'q':
                    break
                elif response == 'n':
                    continue
    else:
        print(f"Invalid input: {args.input}")
        print("Please provide a valid NPZ file or directory containing NPZ files")

if __name__ == "__main__":
    main()