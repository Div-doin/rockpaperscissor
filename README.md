# rock-paper-scissors

An AI to play the Rock Paper Scissors game

## Requirements
- Python 3
- Keras
- Tensorflow
- OpenCV

## Set up instructions
1. Clone the repo.
```sh
$ git clone https://github.com/SouravJohar/rock-paper-scissors.git
$ cd rock-paper-scissors
```

2. Install the dependencies
```sh
$ pip install -r requirements.txt
```

3. Gather Images for each gesture (rock, paper and scissors and None):
In this example, we gather 200 images for the "rock" gesture
```sh
$ python3 gather_images.py rock 200
```

4. Train the model
```sh
$ python3 train.py
```

5. Test the model on some images
```sh
$ python3 test.py <path_to_test_image>
```

6. Play the game with your computer!
```sh
$ python3 play.py
```
7.## Dataset and Trained Model

Due to GitHub file size limitations, the dataset and trained model are hosted on Google Drive.

📁 Google Drive Folder (Dataset + Model):  
https://drive.google.com/drive/folders/1CS_2a2eQadQrxzJR2KJcL3_OXyTbWK2e?usp=drive_link

### How to use the dataset
1. Download the dataset ZIP file from the Drive link
2. Extract it inside the project root
3. Ensure folder structure:

