import random
import cv2
from keras.models import load_model
import numpy as np
import time

list = ["rock", "paper", "scissors"]


class RPS:

    def __init__(self):
        self.model = load_model('keras_model.h5')
        self.cap = cv2.VideoCapture(0)
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        self.computer_wins = 0
        self.user_wins = 0


    def get_computer_choice(self): 
        computer_choice = random.choice(list)
        return computer_choice


    def create_countdown(self):
        countdown = 3
        print("\nGet ready..")
        while countdown > 0:
            print(f'{countdown}')
            cv2.waitKey(1000)
            countdown -= 1
        print('\nShow your object now')


    def get_predictions(self):
        end = time.time() + 1
        while time.time() < end:
            ret, frame = self.cap.read()
            resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
            image_np = np.array(resized_frame)
            normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
            self.data[0] = normalized_image
            prediction = self.model.predict(self.data)
            self.predicted_choice = list[prediction.argmax()]
            cv2.imshow('frame', frame)
            # Press q to close the window
            print(prediction)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        return self.predicted_choice
          
    def get_winner(self, computer_choice, predicted_choice):
        #user_choice = input("What is your choice? Rock, Paper or Scissors? ")

        if computer_choice == predicted_choice:
            print('It is a tie!')
            #break
        elif (
            (computer_choice == 'rock' and predicted_choice == 'scissors') or
            (computer_choice == 'paper' and predicted_choice== 'rock') or
            computer_choice == 'scissors' and predicted_choice == 'paper'
            ):
            print('You lost!')
            self.computer_wins +=1
        else:
            print('You won!')
            self.user_wins += 1

def play(list):
    game = RPS()
    while True:
        
        game.create_countdown()
        computer_choice = game.get_computer_choice()  
        user_choice = game.get_predictions()
        game.get_winner(computer_choice, user_choice) 
        if game.user_wins == 3 or game.computer_wins == 3:
            print(f"The game is over, Computer won {game.computer_wins} and you won {game.user_wins}")
            break

# After the loop release the cap object
    game.cap.release()
# Destroy all the windows
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    list = ['rock', 'paper', 'scissors']
    play(list)
