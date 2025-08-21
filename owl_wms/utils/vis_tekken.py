import cv2
import torch
import torch.nn.functional as F
import numpy as np

# Button mappings - order: [Up, Down, Left, Right, Square, Triangle, Cross, Circle]
BUTTON_NAMES = ['Up', 'Down', 'Left', 'Right', 'Square', 'Triangle', 'Cross', 'Circle']
BUTTON_SYMBOLS = ['^', 'v', '<', '>', '[]', '/\\', 'X', 'O']

def action_id_to_buttons(action_id):
    """Convert action ID to 8-bit button representation."""
    if isinstance(action_id, torch.Tensor):
        action_id = action_id.item()
    
    if isinstance(action_id, (list, np.ndarray)) and len(action_id) > 1:
        # Already in button format
        return action_id
    
    # Convert single ID to 8-bit representation
    action_id = int(action_id)
    buttons = []
    for i in range(7, -1, -1):  # Extract bits 7,6,5,4,3,2,1,0
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

def draw_tekken_frame(frame, action_id, frame_idx=None, total_frames=None, show_legend=False):
    """
    Draw Tekken action overlay on a single frame.
    
    Args:
        frame: torch tensor of shape [3, h, w] with values in range [-1, 1] or [0, 1]
        action_id: int or tensor representing the action ID
        frame_idx: optional frame index for display
        total_frames: optional total frame count for display
        show_legend: whether to show button legend
    
    Returns:
        numpy array of shape [3, h, w] with overlay drawn
    """
    # Handle tensor input
    if isinstance(action_id, torch.Tensor):
        action_id = action_id.item()
    
    frame = frame[:3]  # Only ever take 3 channels
    frame = frame.squeeze(0) if frame.dim() == 4 else frame
    
    # Convert from CHW to HWC
    frame = frame.permute(1, 2, 0)
    
    # Normalize to [0, 255]
    if frame.max() <= 1.0:
        if frame.min() >= -1.0:
            # Range [-1, 1] -> [0, 255]
            frame = (frame + 1) * 127.5
        else:
            # Range [0, 1] -> [0, 255]
            frame = frame * 255
    
    frame = frame.float().cpu().numpy()
    frame = frame.astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
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
    
    # Add frame info if provided
    if frame_idx is not None and total_frames is not None:
        frame_text = f"Frame {frame_idx+1}/{total_frames}"
    else:
        frame_text = ""
    
    # Draw action text
    info_y = height + overlay_height - 25
    cv2.putText(extended_frame, action_text, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if frame_text:
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
    
    # Convert back to RGB and CHW format
    extended_frame = cv2.cvtColor(extended_frame, cv2.COLOR_BGR2RGB)
    extended_frame = np.transpose(extended_frame, (2, 0, 1))  # HWC -> CHW
    return extended_frame

def draw_tekken_frames(frames, action_inputs, show_legend=False):
    """
    Draw Tekken action overlays on batch of frames.
    
    Args:
        frames: torch tensor of shape [b, n, c, h, w] or [b, n, c, h, w]
        action_inputs: torch tensor of shape [b, n] containing action IDs
        show_legend: whether to show button legend on frames
    
    Returns:
        numpy array of shape [b, n, 3, h+overlay_height, w] with overlays drawn
    """
    if frames.dim() == 4:
        # Single batch dimension, add time dimension
        frames = frames.unsqueeze(1)
        action_inputs = action_inputs.unsqueeze(1)
        squeeze_time = True
    else:
        squeeze_time = False
    
    b, n = frames.shape[:2]
    out_frames = []
    
    for i in range(b):
        batch_frames = []
        for j in range(n):
            frame = frames[i, j]
            action = action_inputs[i, j]
            drawn = draw_tekken_frame(frame, action, frame_idx=j, total_frames=n, show_legend=show_legend)
            batch_frames.append(drawn)
        out_frames.append(np.stack(batch_frames))
    
    result = np.stack(out_frames)
    
    if squeeze_time:
        result = result.squeeze(1)
    
    return result

# Example usage and testing function
def test_tekken_overlay():
    """Test function to verify the overlay works correctly"""
    # Create dummy data
    batch_size, seq_len, channels, height, width = 2, 4, 3, 224, 224
    
    # Random frames in range [-1, 1]
    frames = torch.randn(batch_size, seq_len, channels, height, width)
    
    # Test with specific action IDs
    test_actions = torch.tensor([
        [0, 1, 4, 255],  # Neutral, Circle, Triangle, All buttons
        [16, 32, 128, 64]  # Right, Left, Up, Down
    ])
    
    print("Testing with action IDs:")
    for i, actions in enumerate(test_actions):
        for j, action in enumerate(actions):
            buttons = get_active_buttons(action.item())
            print(f"Batch {i}, Frame {j}: Action {action.item()} -> {buttons}")
    
    # Draw overlays
    result = draw_tekken_frames(frames, test_actions, show_legend=True)
    
    print(f"Input shape: {frames.shape}")
    print(f"Output shape: {result.shape}")
    print("Test completed successfully!")

if __name__ == "__main__":
    test_tekken_overlay()