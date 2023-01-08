# halo
This is a reinforcement agent for the video game, Halo Combat Evolved (2001). 

This is a Python program that uses PyAutoGUI and OpenCV to train an agent to play the game Halo using reinforcement learning. The agent is a neural network that takes in game states (screenshots of the game) as input and outputs actions to take in the game. The agent is trained using the Q-learning algorithm, and the game environment is simulated using the HaloEnv class. The program allows the user to control the game using WASD keys and the mouse, and it also allows the user to reset the game and to quit the program.

To use this program, you will need to have the following libraries installed:

PyAutoGUI
OpenCV
NumPy
PyTorch
To run the program, clone this repository and navigate to the root directory in your terminal. Then, run the following command:

Copy code
python halo.py
Features
Trains an agent to play the game using reinforcement learning
Allows the user to control the game using WASD keys and the mouse
Allows the user to reset the game and to quit the program
Uses PyAutoGUI and OpenCV to simulate the game environment and to capture game states
Uses a neural network to model the agent and the Q-learning algorithm to train the agent
Example
To see an example of the agent in action, run the following command:

Copy code
python halo.py --demo
This will run the agent in demo mode, where it will use the trained model to play the game. The agent will take actions in the game based on its predictions of the optimal actions to take given the current game state. You can watch the agent play the game and see how it performs.

Training
To train the agent yourself, run the following command:

Copy code
python halo.py --train
This will run the agent in training mode, where it will learn to play the game by interacting with the game environment and learning from its mistakes. The training process can take a while, depending on the complexity of the game and the size of the neural network. You can stop the training at any time by pressing the 'q' key.

Configuration
You can configure the behavior of the agent by modifying the hyperparameters in the halo.py file. Some of the key hyperparameters include:

learning rate
discount factor
exploration rate
number of epochs
batch size
By adjusting these hyperparameters, you can fine-tune the behavior of the agent and improve its performance.

Limitations
This program is intended for educational purposes only, and it is not a commercial product. As such, it may have some limitations and may not work perfectly in all cases. Some potential limitations include:

Compatibility with different versions of the game or with different operating systems
Performance issues due to the complexity of the game or the size of the neural network
Unforeseen bugs or issues due to the nature of the program
If you encounter any issues while using this program, please open an issue on GitHub and we will do our best to help.
