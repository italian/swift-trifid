import json
import os
from datetime import datetime

LEADERBOARD_FILE = "leaderboard.json"

class Leaderboard:
    def __init__(self):
        self.filename = LEADERBOARD_FILE
        self._load()

    def _load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            except json.JSONDecodeError:
                self.data = {"models": [], "humans": []}
        else:
            self.data = {"models": [], "humans": []}

    def _save(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4)

    def add_model_record(self, name, score, features=None):
        record = {
            "name": name,
            "score": score,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "features": features or {}
        }
        self.data["models"].append(record)
        self.data["models"].sort(key=lambda x: x["score"], reverse=True)
        self.data["models"] = self.data["models"][:20] # Keep top 20
        self._save()

    def add_human_record(self, name, score, speed=None):
        record = {
            "name": name,
            "score": score,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "speed": speed
        }
        self.data["humans"].append(record)
        self.data["humans"].sort(key=lambda x: x["score"], reverse=True)
        self.data["humans"] = self.data["humans"][:20] # Keep top 20
        self._save()

    def display(self):
        print("\n" + "=" * 100)
        print("🏆 LEADERBOARD - SNAKE AI 🏆".center(100))
        print("=" * 100)
        
        print("\n🤖 TOP AI MODELS\n")
        if not self.data["models"]:
            print("   No trained models yet.")
        else:
            for idx, r in enumerate(self.data["models"], 1):
                print(f"#{idx}. Score: {r['score']:>3} | Model: {r['name']}")
                print(f"    📅 Trained: {r['date']}")
                features = r.get("features", {})
                if isinstance(features, dict):
                    if "architecture" in features:
                        print(f"    🧠 Architecture: {features.get('architecture', 'N/A')}")
                    if "hidden_size" in features:
                        print(f"    🔢 Hidden Size: {features.get('hidden_size', 'N/A')}")
                    
                    # Show training info with indication of continued training
                    total_games = features.get('total_games')
                    session_games = features.get('session_games')
                    if total_games and session_games:
                        if total_games > session_games:
                            # Continued training
                            print(f"    🎮 Total Games: {total_games} (🔄 Continued training: +{session_games} games)")
                        else:
                            # Fresh training
                            print(f"    🎮 Total Games Trained: {total_games}")
                    elif total_games:
                        print(f"    🎮 Total Games Trained: {total_games}")
                    
                    if "mean_score" in features:
                        print(f"    📊 Mean Score (this session): {features.get('mean_score', 'N/A')}")
                    if "learning_rate" in features:
                        print(f"    📈 Learning Rate: {features.get('learning_rate', 'N/A')}")
                    if "gamma" in features:
                        print(f"    🎯 Gamma: {features.get('gamma', 'N/A')}")
                print()

        print("-" * 100)
        print("\n👤 TOP HUMAN PLAYERS\n")
        if not self.data["humans"]:
            print("   No human records yet.")
        else:
            for idx, r in enumerate(self.data["humans"], 1):
                print(f"#{idx}. Score: {r['score']:>3} | Player: {r['name']}")
                print(f"    📅 Date: {r['date']}")
                if r.get("speed"):
                    print(f"    ⚡ Game Speed: {r.get('speed', 'N/A')} FPS")
                print()
        
        print("=" * 100)

def add_model_record(name, score, features=None):
    lb = Leaderboard()
    lb.add_model_record(name, score, features)

def add_human_record(name, score, speed=None):
    lb = Leaderboard()
    lb.add_human_record(name, score, speed)

if __name__ == "__main__":
    lb = Leaderboard()
    lb.display()
