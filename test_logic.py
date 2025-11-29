"""
Unit tests for Snake AI project
Tests critical logic to catch bugs like the epsilon issue
"""

import unittest
import os
import json
import tempfile
import numpy as np
from game import SnakeGameAI, Direction, Point
from agent import Agent
from model import LinearQNet
from leaderboard import Leaderboard
from play_human import HumanGame


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

    def test_speed_control(self):
        """Game should respect custom speed setting"""
        custom_speed = 100
        game = SnakeGameAI(speed=custom_speed)
        self.assertEqual(game.speed, custom_speed,
                        "Game speed should match custom setting")


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

    def test_loaded_model_epsilon(self):
        """Loaded model should have low epsilon (high n_games)"""
        # Create a temporary model file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_model_path = f.name
            
        try:
            # Save a dummy model
            model = LinearQNet(11, 256, 3)
            import torch
            torch.save(model.state_dict(), temp_model_path)
            
            # Load it with Agent
            agent = Agent(model_path=temp_model_path)
            
            # n_games should be set high
            self.assertGreaterEqual(agent.n_games, 80,
                                  "Loaded model should have high n_games for low epsilon")
            
            # Epsilon should be low or negative
            state = np.array([0] * 11)
            agent.get_action(state, training=True)
            self.assertLessEqual(agent.epsilon, 0,
                               "Loaded model should have low/negative epsilon")
        finally:
            # Clean up
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)

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


class TestLeaderboard(unittest.TestCase):
    """Test leaderboard functionality"""

    def setUp(self):
        # Use temporary file for testing
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.temp_file.close()
        
        # Monkey-patch the leaderboard file path
        import leaderboard
        self.original_file = leaderboard.LEADERBOARD_FILE
        leaderboard.LEADERBOARD_FILE = self.temp_file.name
        
        self.lb = Leaderboard()

    def tearDown(self):
        # Restore original file path
        import leaderboard
        leaderboard.LEADERBOARD_FILE = self.original_file
        
        # Clean up temp file
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)

    def test_add_model_record(self):
        """Should add model record with proper metadata"""
        self.lb.add_model_record("test_model.pth", 50, 
                                features={"architecture": "Linear Q-Network"})
        
        self.assertEqual(len(self.lb.data["models"]), 1)
        self.assertEqual(self.lb.data["models"][0]["score"], 50)
        self.assertEqual(self.lb.data["models"][0]["features"]["architecture"], 
                        "Linear Q-Network")

    def test_add_human_record(self):
        """Should add human record"""
        self.lb.add_human_record("Player1", 25, speed=40)
        
        self.assertEqual(len(self.lb.data["humans"]), 1)
        self.assertEqual(self.lb.data["humans"][0]["score"], 25)
        self.assertEqual(self.lb.data["humans"][0]["speed"], 40)

    def test_leaderboard_sorting(self):
        """Records should be sorted by score descending"""
        self.lb.add_model_record("model1", 10)
        self.lb.add_model_record("model2", 50)
        self.lb.add_model_record("model3", 30)
        
        scores = [r["score"] for r in self.lb.data["models"]]
        self.assertEqual(scores, [50, 30, 10], "Should be sorted descending")


class TestHumanGame(unittest.TestCase):
    """Test human game mode"""

    def test_game_over_flag(self):
        """Game over flag should control message repetition"""
        game = HumanGame()
        self.assertFalse(game.game_over, "Should start with game_over=False")
        
        game.game_over = True
        # Simulate that game is over - further play_step should not execute
        # (tested manually as it requires pygame event loop)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
