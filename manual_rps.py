import random
list = ["rock", "paper", "scissors"]

def get_computer_choice (): 
    computer_choice = random.choice(list)
    return computer_choice
c = get_computer_choice()
print(c)
    
def get_user_choice():
    user_choice = input("What is your choice? Rock, Paper or Scissors? ")
    user_choice = user_choice.lower()
    if user_choice not in list:
       print("Invalid input, try again!")
    return user_choice
u = get_user_choice()
print(u)

def get_winner(computer_choice, user_choice):
    #user_choice = input("What is your choice? Rock, Paper or Scissors? ")

   while True:    
        
        if computer_choice == user_choice:
            print('It is a tie!')
            break
        elif (
            (computer_choice == 'rock' and user_choice == 'scissors') or
            (computer_choice == 'paper' and user_choice == 'rock') or
            computer_choice == 'scissors' and user_choice == 'paper'
            ):
            print('You lost!')
            break
        else:
            print('You won!')
            break
            
def play():
    list = ["Rock", "Paper", "Scissors"]
    computer_choice = get_computer_choice()
    user_choice = get_user_choice()
    get_winner(computer_choice, user_choice) 
play()     

