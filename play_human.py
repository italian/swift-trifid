import sys
import argparse
import pygame
from game import SnakeGameAI, Direction, DEFAULT_SPEED

def get_args():
    parser = argparse.ArgumentParser(description='Play Snake as a human')
    parser.add_argument('--speed', type=int, default=DEFAULT_SPEED,
                        help=f'Game speed in FPS (default: {DEFAULT_SPEED})')
    return parser.parse_args()

# Человеческий режим игры
class HumanGame(SnakeGameAI):
    def __init__(self, w=640, h=480, speed=DEFAULT_SPEED):
        super().__init__(w, h, speed)
        self.paused = False
        self.session_best = 0
        self.game_over = False

    def play_human(self):
        """Основной игровой цикл для человека"""
        while True:
            # Обработка событий
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._save_to_leaderboard()
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        self.reset()
                        self.game_over = False

                    # Управление (только если не на паузе и не game over)
                    if not self.paused and not self.game_over:
                        if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                            if self.direction != Direction.RIGHT:
                                self.direction = Direction.LEFT
                        elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                            if self.direction != Direction.LEFT:
                                self.direction = Direction.RIGHT
                        elif event.key == pygame.K_UP or event.key == pygame.K_w:
                            if self.direction != Direction.DOWN:
                                self.direction = Direction.UP
                        elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                            if self.direction != Direction.UP:
                                self.direction = Direction.DOWN

            if not self.paused and not self.game_over:
                # Конвертируем направление в действие для play_step
                action = self._direction_to_action()
                reward, game_over, score = self.play_step(action)

                # Track best score
                if score > self.session_best:
                    self.session_best = score

                if game_over:
                    self.game_over = True
                    print(f'\n🎮 Game Over! Your Score: {score}')
                    print(f'Session Best: {self.session_best}')
                    print('Press R to restart or close window to exit')

            self.clock.tick(self.speed)

    def _save_to_leaderboard(self):
        """Save session best score to leaderboard"""
        if self.session_best > 0:
            from leaderboard import add_human_record
            player_name = input("\nEnter your name for the leaderboard: ").strip() or "Anonymous"
            add_human_record(player_name, self.session_best, speed=self.speed)
            print(f"✅ Your best score ({self.session_best}) saved to leaderboard!")
            print("Run 'python leaderboard.py' to view all records!")

    def _direction_to_action(self):
        """Конвертирует текущее направление в формат действия [straight, right, left]"""
        # Текущее направление уже установлено через события клавиатуры
        # Просто возвращаем 'идти прямо'
        return [1, 0, 0]


if __name__ == '__main__':
    args = get_args()
    
    print("=" * 60)
    print("🐍 SNAKE GAME - HUMAN MODE 🐍")
    print("=" * 60)
    print("\nControls:")
    print("  ↑ / W     - Move Up")
    print("  ↓ / S     - Move Down")
    print("  ← / A     - Move Left")
    print("  → / D     - Move Right")
    print("  SPACE     - Pause/Resume")
    print("  R         - Restart")
    print(f"\nGame Speed: {args.speed} FPS")
    print("\nGood luck! 🍎")
    print("=" * 60)

    game = HumanGame(speed=args.speed)
    game.play_human()
