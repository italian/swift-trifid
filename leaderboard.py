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
        print("\n🏆 LEADERBOARD 🏆")
        
        print("\n🤖 TOP AI MODELS")
        if not self.data["models"]:
            print("No records yet.")
        else:
            print(f"{'Score':<10} | {'Model Name':<30} | {'Date':<20} | {'Features'}")
            print("-" * 80)
            for r in self.data["models"]:
                features = str(r.get("features", ""))
                print(f"{r['score']:<10} | {r['name']:<30} | {r['date']:<20} | {features}")

        print("\n👤 TOP HUMANS")
        if not self.data["humans"]:
            print("No records yet.")
        else:
            print(f"{'Score':<10} | {'Player Name':<20} | {'Date':<20} | {'Speed'}")
            print("-" * 70)
            for r in self.data["humans"]:
                speed = str(r.get("speed", "N/A"))
                print(f"{r['score']:<10} | {r['name']:<20} | {r['date']:<20} | {speed}")

def add_model_record(name, score, features=None):
    lb = Leaderboard()
    lb.add_model_record(name, score, features)

def add_human_record(name, score, speed=None):
    lb = Leaderboard()
    lb.add_human_record(name, score, speed)

if __name__ == "__main__":
    lb = Leaderboard()
    lb.display()
