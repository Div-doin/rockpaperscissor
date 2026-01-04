import random
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import time

# Path to the background image
bg_image_path = "/mnt/data/BG.png"

# Load the background image
imgBG = cv2.imread(bg_image_path)

# Check if the background image was loaded successfully
if imgBG is None:
    print(f"Failed to load background image from {bg_image_path}")
else:
    print("Background image loaded successfully")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = HandDetector(maxHands=1)

timer = 0
stateResult = False
startGame = False
scores = [0, 0]  # [AI, Player]

while True:
    success, img = cap.read()
    
    if not success:
        print("Failed to capture image from camera.")
        continue
    
    imgScaled = cv2.resize(img, (0, 0), None, 0.875, 0.875)
    imgScaled = imgScaled[:, 80:480]
    
    if imgScaled is None:
        print("Failed to resize the image.")
        continue

    # Find Hands
    hands, img = detector.findHands(imgScaled)  # with draw

    if startGame:
        if stateResult is False:
            timer = time.time() - initialTime
            cv2.putText(imgBG, str(int(timer)), (605, 435), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 255), 4)

            if timer > 3:
                stateResult = True
                timer = 0

                if hands:
                    playerMove = None
                    hand = hands[0]
                    fingers = detector.fingersUp(hand)
                    if fingers == [0, 0, 0, 0, 0]:
                        playerMove = 1
                    if fingers == [1, 1, 1, 1, 1]:
                        playerMove = 2
                    if fingers == [0, 1, 1, 0, 0]:
                        playerMove = 3

                    randomNumber = secure_random.randint(1, 3)
                    imgAI = cv2.imread(f'RESOURCES/{randomNumber}.png', cv2.IMREAD_UNCHANGED)
                    if imgAI is not None:
                        imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))
                    else:
                        print(f"Failed to load AI image RESOURCES/{randomNumber}.png")

                    # Player Wins
                    if (playerMove == 1 and randomNumber == 3) or \
                            (playerMove == 2 and randomNumber == 1) or \
                            (playerMove == 3 and randomNumber == 2):
                        scores[1] += 1

                    # AI Wins
                    if (playerMove == 3 and randomNumber == 1) or \
                            (playerMove == 1 and randomNumber == 2) or \
                            (playerMove == 2 and randomNumber == 3):
                        scores[0] += 1

    if stateResult:
        if imgAI is not None:
            imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))

    cv2.putText(imgBG, str(scores[0]), (410, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)
    cv2.putText(imgBG, str(scores[1]), (1112, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)

    # Check if imgBG is loaded properly before displaying
    if imgBG is not None:
        cv2.imshow("BG", imgBG)
    else:
        print("imgBG is None, cannot display the image.")

    key = cv2.waitKey(1)
    if key == ord('s'):
        startGame = True
        initialTime = time.time()
        stateResult = False

cv2.destroyAllWindows()
cap.release()
