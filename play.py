# play_fixed.py
# Fixed Rock-Paper-Scissors webcam demo:
# - loads computer icons from common locations (/mnt/data, ./images, current dir)
# - handles PNGs with alpha (transparency)
# - scales model input to [0,1]
# - adapts computer rectangle to actual frame width
# - safer fallback behavior when icons or camera not available

import os
import cv2
import numpy as np
from random import choice
from keras.models import load_model

REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none"
}

def mapper(val):
    return REV_CLASS_MAP.get(val, "none")

def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"
    if move1 == "rock":
        return "User" if move2 == "scissors" else "Computer"
    if move1 == "paper":
        return "User" if move2 == "rock" else "Computer"
    if move1 == "scissors":
        return "User" if move2 == "paper" else "Computer"
    return "Waiting..."

# Try to locate the model in the current directory
MODEL_PATH = "rock-paper-scissors-model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Place model in script directory.")

model = load_model("rock-paper-scissors-model.h5")

# locations to search for images (order: user-provided container path, ./images, current dir)
IMAGE_DIRS = ["/mnt/data", "images", "."]

def find_icon_path(name):
    for d in IMAGE_DIRS:
        p = os.path.join(d, f"{name}.png")
        if os.path.exists(p):
            return p
    return None

# Preload icons (or None if missing)
ICON_PATHS = {m: find_icon_path(m) for m in ["rock", "paper", "scissors"]}
print("Icon paths found:", ICON_PATHS)

def load_icon(path, size):
    """Load icon, handle alpha channel, resize to size (w,h) and return 3-channel BGR image plus mask if alpha present."""
    icon = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if icon is None:
        return None, None
    # If icon has alpha channel (4 channels), split alpha as mask
    if icon.shape[2] == 4:
        bgr = icon[:, :, :3]
        alpha = icon[:, :, 3]  # 0-255
        alpha_mask = alpha.astype(float) / 255.0
        bgr = cv2.resize(bgr, size)
        alpha_mask = cv2.resize(alpha_mask, size)
        return bgr, alpha_mask
    else:
        bgr = cv2.resize(icon, size)
        return bgr, None

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera index and permissions.")

prev_move = None
computer_move_name = "none"
winner = "Waiting..."

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    H, W = frame.shape[:2]

    # User rectangle (left)
    ux1, uy1 = int(W * 0.05), int(H * 0.10)
    ux2, uy2 = int(W * 0.45), int(H * 0.60)

    # Computer rectangle (right)
    cx1, cy1 = int(W * 0.55), int(H * 0.10)
    cx2, cy2 = int(W * 0.95), int(H * 0.60)

    cv2.rectangle(frame, (ux1, uy1), (ux2, uy2), (255, 255, 255), 2)
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 255, 255), 2)

    # extract ROI for user's move
    roi = frame[uy1:uy2, ux1:ux2]
    # prepare image for model: convert BGR->RGB, resize, scale to [0,1]
    img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (227, 227))
    img_input = img_resized.astype("float32") / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    pred = model.predict(img_input)
    move_code = int(np.argmax(pred[0]))
    user_move_name = mapper(move_code)

    # decide computer move only when user_move changes and it's not "none"
    if prev_move != user_move_name:
        if user_move_name != "none":
            computer_move_name = choice(['rock', 'paper', 'scissors'])
            winner = calculate_winner(user_move_name, computer_move_name)
        else:
            computer_move_name = "none"
            winner = "Waiting..."
    prev_move = user_move_name

    # Display info
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Your Move: {user_move_name}", (ux1, max(30, uy1 - 10)), font, 0.9, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Computer's Move: {computer_move_name}", (cx1, max(30, cy1 - 10)), font, 0.9, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Winner: {winner}", (int(W*0.35), int(H*0.80)), font, 1.5, (0,0,255), 3, cv2.LINE_AA)

    # draw computer icon (if available)
    if computer_move_name != "none":
        icon_path = ICON_PATHS.get(computer_move_name)
        if icon_path:
            box_w = cx2 - cx1
            box_h = cy2 - cy1
            icon_bgr, alpha_mask = load_icon(icon_path, (box_w, box_h))
            if icon_bgr is not None:
                # place icon onto frame with alpha blending if mask present
                if alpha_mask is not None:
                    for c in range(3):
                        frame[cy1:cy2, cx1:cx2, c] = (alpha_mask * icon_bgr[:, :, c] +
                                                     (1 - alpha_mask) * frame[cy1:cy2, cx1:cx2, c]).astype(np.uint8)
                else:
                    frame[cy1:cy2, cx1:cx2] = icon_bgr
        else:
            # fallback: just write text if icon missing
            cv2.putText(frame, f"[No icon for {computer_move_name}]", (cx1+5, cy1+40), font, 0.7, (200,200,200), 2)

    cv2.imshow("Rock Paper Scissors", frame)

    k = cv2.waitKey(10) & 0xFF
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
