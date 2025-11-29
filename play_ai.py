import sys
import argparse
import os
import torch
from game import SnakeGameAI
from model import LinearQNet
from agent import Agent

def get_args():
    parser = argparse.ArgumentParser(description='Watch trained AI play Snake')
    parser.add_argument('--model', type=str, default='model/model.pth',
                        help='Path to trained model file')
    parser.add_argument('--speed', type=int, default=20,
                        help='Game speed (lower = faster)')
    parser.add_argument('--games', type=int, default=10,
                        help='Number of games to play (0 = infinite)')
    return parser.parse_args()

def play_ai():
    args = get_args()

    # Проверяем наличие модели
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print("\nAvailable models in ./model/:")
        if os.path.exists('./model'):
            models = [f for f in os.listdir('./model') if f.endswith('.pth')]
            for model in models:
                print(f"  - {model}")
        else:
            print("  (no models found)")
        sys.exit(1)

    # Загружаем модель
    print(f"🤖 Loading AI model: {args.model}")
    agent = Agent(model_path=args.model)
    agent.epsilon = 0  # Отключаем случайные действия

    game = SnakeGameAI(speed=args.speed)

    total_score = 0
    max_score = 0
    games_played = 0

    print("\n" + "=" * 60)
    print("🎮 AI DEMONSTRATION MODE 🤖")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Speed: {args.speed}")
    print(f"Games to play: {'Infinite' if args.games == 0 else args.games}")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            state = agent.get_state(game)
            action = agent.get_action(state, training=False)  # Режим демонстрации
            reward, done, score = game.play_step(action)
            # game.clock.tick(args.speed) # Removed manual tick, handled in game.play_step

            if done:
                games_played += 1
                total_score += score
                max_score = max(max_score, score)
                avg_score = total_score / games_played

                print(f"Game {games_played}: Score = {score:3d} | "
                      f"Avg = {avg_score:5.1f} | Max = {max_score:3d}")

                game.reset()

                if args.games > 0 and games_played >= args.games:
                    break

    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user")

    print("\n" + "=" * 60)
    print("📊 FINAL STATISTICS")
    print("=" * 60)
    print(f"Games Played:  {games_played}")
    print(f"Average Score: {total_score / games_played if games_played > 0 else 0:.2f}")
    print(f"Max Score:     {max_score}")
    print("=" * 60)

    # Save to leaderboard
    from leaderboard import add_model_record
    model_basename = os.path.basename(args.model)
    add_model_record(
        name=model_basename,
        score=max_score,
        features={"avg_score": f"{total_score / games_played if games_played > 0 else 0:.2f}", "games": games_played, "speed": args.speed}
    )
    print(f"\n✅ Record saved to leaderboard as '{model_basename}'")
    print("Run 'python leaderboard.py' to view all records!")


if __name__ == '__main__':
    play_ai()
