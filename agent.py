import argparse
import time
import os
import random
from collections import deque
import torch
import numpy as np
from game import SnakeGameAI, Direction, Point
from model import LinearQNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

def get_args():
    parser = argparse.ArgumentParser(description='Snake AI Agent')
    parser.add_argument('--load', type=str,
                        help='Path to model file to load (e.g., model/model.pth)')
    parser.add_argument('--new', action='store_true',
                        help='Start training from scratch (default)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (games without improvement)')
    parser.add_argument('--speed', type=int, default=40,
                        help='Game speed (default: 40)')
    return parser.parse_args()

class Agent:

    def __init__(self, model_path=None):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = LinearQNet(11, 256, 3)

        if model_path:
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}...")
                # Security: weights_only=True prevents code execution from pickle
                self.model.load_state_dict(torch.load(model_path, weights_only=True))
                self.model.eval()
            else:
                print(f"Warning: Model {model_path} not found. Starting from scratch.")

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y   # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # popleft if MAX_MEMORY is reached
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # nosec
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, training=True):
        # random moves: tradeoff exploration / exploitation
        if training:
            self.epsilon = 80 - self.n_games

        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon: # nosec
            move = random.randint(0, 2) # nosec
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    args = get_args()

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    # Generate unique model name for this run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_filename = f"model_{timestamp}.pth"

    agent = Agent(model_path=args.load)
    game = SnakeGameAI(speed=args.speed)

    # Early stopping variables
    games_without_improvement = 0
    patience = args.patience

    print(f"Starting training. Saving best models to {model_filename}")
    print(f"Early stopping patience: {patience} games")

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save(file_name=model_filename)
                print(f"New Record! Saved {model_filename}")
                
                # Save to leaderboard
                from leaderboard import add_model_record
                add_model_record(model_filename, record, features={"games": agent.n_games})
                
                games_without_improvement = 0
            else:
                games_without_improvement += 1

            print(f'Game {agent.n_games} Score {score} '
                  f'Record {record} No Improv {games_without_improvement}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            if games_without_improvement >= patience:
                print(f"\nEarly Stopping triggered! No improvement for {patience} games.")
                print(f"Best Record: {record}")
                print(f"Model saved as: {model_filename}")
                break

if __name__ == '__main__':
    train()
