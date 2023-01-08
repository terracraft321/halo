import torch
import torch.nn as nn
import torch.optim as optim
import pyautogui
import cv2
import numpy as np

# define the keys that will be used to control the game
WASD_KEYS = ['w', 'a', 's', 'd']

# define the neural network for the agent
class Agent(nn.Module):
    def __init__(self, input_size, output_size):
        super(Agent, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_action(self, state):
    # reshape the state array to have a shape of (1, self.fc1.in_features)
        state = np.array(state)  # convert state to a numpy array
        state = state.flatten()  # flatten the state array
        state = np.resize(state, 1049088)  # resize the state array to have a size of 1049088
        state = np.reshape(state, (1, 480000))  # reshape the state array to have a shape of (1, 480000)
        action = agent.get_action(state)

# define the Halo environment
class HaloEnv:
    def __init__(self):
        # set the size of the game window
        self.width = 800
        self.height = 600

        # set the action space
        self.action_space = ['forward', 'left', 'right', 'shoot']
    
    def reset(self):
        # bring the game window to the front
        pyautogui.click(self.width // 2, self.height // 2)

        # get the initial game state
        game_state = self.get_game_state()

        return game_state
    
    def step(self, action):
        # perform the given action
        if action == 'forward':
            pyautogui.keyDown('w')
            pyautogui.keyUp('w')
        elif action == 'left':
            pyautogui.keyDown('a')
            pyautogui.keyUp('a')
        elif action == 'right':
            pyautogui.keyDown('d')
            pyautogui.keyUp('d')
        elif action == 'shoot':
            pyautogui.mouseDown()
            pyautogui.mouseUp()
        
        # get the new game state
        next_state = self.get_game_state()

        # determine the reward for the action
        reward = self.get_reward(state, next_state)

        # check if the game is over
        done = self.is_game_over(next_state)

        return next_state, reward, done, {}
    
    def get_game_state(self):
        # get a screenshot of the game
        img = pyautogui.screenshot()
        # convert the screenshot to a numpy array
        img_np = np.array(img)
        # convert the image to grayscale
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        return img_np
    
    def get_reward(self, state, next_state):
        # define the reward as the difference in score between the two states
        reward = self.get_score(next_state) - self.get_score

def get_score(self, state):
        # define the score as the sum of the values of the white pixels in the image
        score = np.sum(state == 255)
        return score
    
def is_game_over(self, state):
    # define the game as over if there are no more white pixels in the image
    return np.sum(state == 255) == 0

# define the Halo environment
env = HaloEnv()

# define the agent
agent = Agent(input_size=env.width*env.height, output_size=len(env.action_space))

# define the agent
agent = Agent(input_size=env.width*env.height, output_size=len(env.action_space))

# define the training loop for the agent
def train(agent, env, num_episodes=10000, learning_rate=0.01, discount_factor=0.99):
    # define the optimizer
    optimizer = optim.Adam(agent.parameters(), learning_rate)

    # set the starting episode
    episode = 1

    # loop through the number of episodes
    while episode <= num_episodes:
        # reset the environment
        state = env.reset()

        # set the cumulative reward for the episode
        episode_reward = 0

        # set the loop variable to True
        done = False

        # loop until the game is over
        while not done:
            # get the action from the agent
            action = agent.get_action(state)

            # take a step in the environment
            next_state, reward, done, _ = env.step(action)

            # update the cumulative reward for the episode
            episode_reward += reward

            # update the agent
            agent.update(state, action, next_state, reward, done, optimizer, discount_factor)

            # set the new state
            state = next_state

        # print the reward for the episode
        print(f'Episode {episode}: {episode_reward}')

        # increment the episode
        episode += 1

# run the training loop
train(agent, env)
