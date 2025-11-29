"""
Unit tests for Snake AI project
Tests critical logic to catch bugs like the epsilon issue
"""

import unittest
import numpy as np
from game import SnakeGameAI, Direction, Point
from agent import Agent
from model import LinearQNet


class TestGameLogic(unittest.TestCase):
    """Test game environment logic"""

    def setUp(self):
        self.game = SnakeGameAI()

    def test_initial_state(self):
        """Game should start with score 0 and snake of length 3"""
        self.assertEqual(self.game.score, 0)
        self.assertEqual(len(self.game.snake), 3)

    def test_food_placement_not_on_snake(self):
        """Food should never appear inside the snake body"""
        for _ in range(100):  # Test multiple times due to randomness
            self.game._place_food()
            self.assertNotIn(self.game.food, self.game.snake,
                           "Food appeared inside snake body!")

    def test_collision_detection(self):
        """Collision detection should work correctly"""
        # Test wall collision
        self.game.head = Point(-1, 0)
        self.assertTrue(self.game.is_collision(), "Should detect wall collision")

        # Test self-collision
        self.game.head = Point(100, 100)
        self.game.snake = [Point(100, 100), Point(80, 100), Point(100, 100)]
        self.assertTrue(self.game.is_collision(), "Should detect self-collision")


class TestAgentLogic(unittest.TestCase):
    """Test agent behavior"""

    def setUp(self):
        self.agent = Agent()

    def test_epsilon_behavior_training(self):
        """Epsilon should decrease during training"""
        initial_epsilon = self.agent.epsilon
        self.agent.n_games = 10

        # In training mode, epsilon should update
        state = np.array([0] * 11)
        self.agent.get_action(state, training=True)

        expected_epsilon = 80 - 10  # 70
        self.assertEqual(self.agent.epsilon, expected_epsilon,
                        "Epsilon should update during training")

    def test_epsilon_behavior_inference(self):
        """Epsilon should NOT change during inference"""
        self.agent.epsilon = 0  # Set to 0 for pure exploitation
        self.agent.n_games = 50

        state = np.array([0] * 11)

        # In inference mode, epsilon should stay 0
        self.agent.get_action(state, training=False)

        self.assertEqual(self.agent.epsilon, 0,
                        "Epsilon should NOT change during inference!")

    def test_state_vector_size(self):
        """State vector should always be size 11"""
        game = SnakeGameAI()
        state = self.agent.get_state(game)

        self.assertEqual(len(state), 11,
                        "State vector must have exactly 11 elements")
        self.assertTrue(np.issubdtype(state.dtype, np.integer),
                       "State elements should be integers (0 or 1)")


class TestModelArchitecture(unittest.TestCase):
    """Test model structure"""

    def test_model_output_shape(self):
        """Model should output 3 Q-values"""
        model = LinearQNet(11, 256, 3)

        import torch
        test_input = torch.zeros(11)
        output = model(test_input)

        self.assertEqual(output.shape[0], 3,
                        "Model should output 3 Q-values (straight, right, left)")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
