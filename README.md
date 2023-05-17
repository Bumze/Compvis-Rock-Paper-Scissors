# Computer Vision RPS
# Overview of the project
Rock-Paper-Scissors is a game in which each player simultaneously shows one of three images representing rock, paper, or scissors. Rock beats scissors. Scissors beats paper. Paper beats rock. The player who shows the first option that beats the other player's option wins. This is an implementation of an interactive "Rock Paper Scissors" game in which the user can play with the computer using the camera.

# Milestone 1 - 3
For this milestone, I used Teachable Machines, a web-based tool for machine learning models to train a model with images. The model can recognise four different image categories: Rock, Paper, Scissors or Nothing. The trained modelwas downloaded to my local machine for further processing. To implement this, a virtual environment was created in which relevant dependencies were installed.  Important syntaxes used include: 
conda install 
conda activate env
conda install pip
pip install <libary>
Finally, I created a requirements.txt file by running the following command:
pip list > requirements.txt This file allows other users that want to use my computer to install the exact dependencies by running
pip install requirements.txt. 

# Milestone 4 
Creating the Rock, Paper & Scissors game manually 
A script that plays the game without the use of a camera was created.
  
# Milestone 5
A new file called camera_rps.py was created to create a code that will allow the computer to make predictions which is used to replace the user guess in the manual_rps file. The comnputer plays against predictions.
  
Methods created include:
get_prediction to return the output of the model.
create_countdown - to create a countdown process that signifies a readiness to start the game.
get_winner- a method that has attributes computer choice and the newly created predictions, affixed a variable called user choice 
play- to play the game.
  