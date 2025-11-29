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
        # Also verify initial direction and frame count
        self.assertIsNotNone(self.game.direction)
        self.assertEqual(self.game.frame_iteration, 0)

    def test_food_placement_not_on_snake(self):
        """Food should never appear inside the snake body"""
        for _ in range(100):  # Test multiple times due to randomness
            self.game._place_food()
            self.assertNotIn(self.game.food, self.game.snake,
                           "Food appeared inside snake body!")

    def test_collision_detection(self):
        """Collision detection should work correctly"""
        # Save original head position
        original_head = self.game.head
        
        # Test wall collision - left wall
        self.game.head = Point(-1, 100)
        self.assertTrue(self.game.is_collision(), "Should detect left wall collision")
        
        # Test wall collision - right wall
        self.game.head = Point(self.game.w, 100)
        self.assertTrue(self.game.is_collision(), "Should detect right wall collision")
        
        # Test wall collision - top wall
        self.game.head = Point(100, -1)
        self.assertTrue(self.game.is_collision(), "Should detect top wall collision")
        
        # Test wall collision - bottom wall
        self.game.head = Point(100, self.game.h)
        self.assertTrue(self.game.is_collision(), "Should detect bottom wall collision")

        # Test self-collision
        self.game.head = Point(100, 100)
        self.game.snake = [Point(100, 100), Point(80, 100), Point(100, 100)]
        self.assertTrue(self.game.is_collision(), "Should detect self-collision")
        
        # Test no collision in valid position
        self.game.head = Point(320, 240)  # Center of default 640x480 screen
        self.game.snake = [Point(320, 240), Point(300, 240), Point(280, 240)]
        self.assertFalse(self.game.is_collision(), "Should not detect collision in valid position")

    def test_speed_control(self):
        """Game should respect custom speed setting"""
        custom_speed = 100
        game = SnakeGameAI(speed=custom_speed)
        self.assertEqual(game.speed, custom_speed,
                        "Game speed should match custom setting")
        
        # Test default speed
        default_game = SnakeGameAI()
        from game import DEFAULT_SPEED
        self.assertEqual(default_game.speed, DEFAULT_SPEED,
                        "Default speed should match DEFAULT_SPEED constant")


class TestAgentLogic(unittest.TestCase):
    """Test agent behavior"""

    def setUp(self):
        self.agent = Agent()

    def test_epsilon_behavior_training(self):
        """Epsilon should decrease during training"""
        self.agent.n_games = 10

        # In training mode, epsilon should update
        state = np.array([0] * 11)
        self.agent.get_action(state, training=True)

        expected_epsilon = 80 - 10  # 70
        self.assertEqual(self.agent.epsilon, expected_epsilon,
                        "Epsilon should update during training")
        
        # Test that epsilon goes negative for high n_games
        self.agent.n_games = 100
        self.agent.get_action(state, training=True)
        self.assertEqual(self.agent.epsilon, -20,
                        "Epsilon should be negative for high n_games")

    def test_epsilon_behavior_inference(self):
        """Epsilon should NOT change during inference"""
        self.agent.epsilon = 0  # Set to 0 for pure exploitation
        self.agent.n_games = 50

        state = np.array([0] * 11)

        # In inference mode, epsilon should stay 0
        self.agent.get_action(state, training=False)

        self.assertEqual(self.agent.epsilon, 0,
                        "Epsilon should NOT change during inference!")

    def test_checkpoint_save_and_load(self):
        """Model checkpoint should save and restore n_games correctly"""
        import torch
        
        # Create a temporary model file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_model_path = f.name
            
        try:
            # Create an agent and train it for some games
            agent1 = Agent()
            agent1.n_games = 350  # Simulate 350 games played
            
            # Save the model with checkpoint
            agent1.model.save(file_name=os.path.basename(temp_model_path), n_games=agent1.n_games)
            
            # Move to temp location for loading
            import shutil
            saved_path = os.path.join('./model', os.path.basename(temp_model_path))
            if os.path.exists(saved_path):
                shutil.move(saved_path, temp_model_path)
            
            # Load it with a new Agent
            agent2 = Agent(model_path=temp_model_path)
            
            # CRITICAL: n_games should be restored to 350, NOT reset to 0 or 100
            self.assertEqual(agent2.n_games, 350,
                           f"Loaded model should restore n_games=350, but got {agent2.n_games}")
            
            # Verify epsilon is low (exploitation mode)
            state = np.array([0] * 11)
            agent2.get_action(state, training=True)
            expected_epsilon = 80 - 350  # Should be -270
            self.assertEqual(agent2.epsilon, expected_epsilon,
                           "Epsilon should be very negative for n_games=350")
            
        finally:
            # Clean up
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
            # Clean up model folder
            saved_path = os.path.join('./model', os.path.basename(temp_model_path))
            if os.path.exists(saved_path):
                os.remove(saved_path)

    def test_old_model_format_compatibility(self):
        """Old models (state_dict only) should still load with fallback n_games"""
        import torch
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_model_path = f.name
            
        try:
            # Save OLD format (just state_dict, no checkpoint)
            model = LinearQNet(11, 256, 3)
            torch.save(model.state_dict(), temp_model_path)
            
            # Load it with Agent
            agent = Agent(model_path=temp_model_path)
            
            # Should fall back to n_games=100 for old models
            self.assertEqual(agent.n_games, 100,
                           "Old format models should default to n_games=100")
            
        finally:
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
        
        # Verify all elements are 0 or 1
        for element in state:
            self.assertIn(element, [0, 1],
                        f"State element {element} should be 0 or 1")


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
        
        # Verify output is a tensor
        self.assertIsInstance(output, torch.Tensor,
                            "Model output should be a torch Tensor")

    def test_model_forward_pass(self):
        """Model forward pass should not crash and produce reasonable values"""
        import torch
        model = LinearQNet(11, 256, 3)
        
        # Test with random input
        random_input = torch.randn(11)
        output = model(random_input)
        
        # Output should be finite
        self.assertTrue(torch.isfinite(output).all(),
                       "Model output should be finite (no NaN or Inf)")


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
        
        # Verify date was added
        self.assertIn("date", self.lb.data["models"][0])

    def test_add_human_record(self):
        """Should add human record"""
        self.lb.add_human_record("Player1", 25, speed=40)
        
        self.assertEqual(len(self.lb.data["humans"]), 1)
        self.assertEqual(self.lb.data["humans"][0]["score"], 25)
        self.assertEqual(self.lb.data["humans"][0]["speed"], 40)
        self.assertEqual(self.lb.data["humans"][0]["name"], "Player1")

    def test_leaderboard_sorting(self):
        """Records should be sorted by score descending"""
        self.lb.add_model_record("model1", 10)
        self.lb.add_model_record("model2", 50)
        self.lb.add_model_record("model3", 30)
        
        scores = [r["score"] for r in self.lb.data["models"]]
        self.assertEqual(scores, [50, 30, 10], "Should be sorted descending")
    
    def test_leaderboard_limit(self):
        """Should keep only top 20 records"""
        # Add 25 model records
        for i in range(25):
            self.lb.add_model_record(f"model_{i}", i)
        
        # Should keep only top 20
        self.assertEqual(len(self.lb.data["models"]), 20,
                        "Should keep only top 20 records")
        
        # Verify it kept the highest scores (24 down to 5)
        scores = [r["score"] for r in self.lb.data["models"]]
        self.assertEqual(scores[0], 24, "Top score should be 24")
        self.assertEqual(scores[-1], 5, "Lowest score should be 5")


class TestHumanGame(unittest.TestCase):
    """Test human game mode"""

    def test_game_over_flag_initial(self):
        """Game should start with game_over=False"""
        game = HumanGame()
        self.assertFalse(game.game_over, "Should start with game_over=False")

    def test_game_over_flag_setter(self):
        """Game over flag should be settable"""
        game = HumanGame()
        game.game_over = True
        self.assertTrue(game.game_over, "game_over should be True after setting")
        
        # Reset
        game.game_over = False
        self.assertFalse(game.game_over, "game_over should be False after reset")
    
    def test_human_game_has_session_best(self):
        """Human game should track session_best"""
        game = HumanGame()
        self.assertEqual(game.session_best, 0, "session_best should start at 0")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
