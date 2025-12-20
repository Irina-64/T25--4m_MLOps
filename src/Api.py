# src\Api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from env.durak_env import DurakEnv
from Core.agents import RLAgent
from src.preprocess import state_to_tensor

# ------------------ Конфигурация ------------------
WEIGHTS_PATH = "rl_weights.pth"

# ------------------ FastAPI ------------------
app = FastAPI(title="Durak RL API", version="1.0")

# ------------------ Модель и среда ------------------
# Временная игра для вычисления state_size
from Core.core import DurakGame
temp_game = DurakGame(["PLAYER", "DUMMY"])
state0 = state_to_tensor(temp_game.get_state(0))
state_size = len(state0)

rl_agent = RLAgent(pid=1, state_size=state_size, weights_path=WEIGHTS_PATH)
env = DurakEnv(rl_agent)

# ------------------ Схема запросов ------------------
class ActionRequest(BaseModel):
    action: str  # строка действия, например "6♣" или "take"

class ResetResponse(BaseModel):
    state: List[float]
    info: dict

class StepResponse(BaseModel):
    state: List[float]
    reward: float
    done: bool
    truncated: bool
    info: dict

# ------------------ Эндпоинты ------------------
@app.get("/reset", response_model=ResetResponse)
def reset_game():
    state, info = env.reset()
    return ResetResponse(state=state.tolist(), info=info)

@app.post("/step", response_model=StepResponse)
def step_game(req: ActionRequest):
    try:
        state, reward, done, truncated, info = env.step(req.action)
        return StepResponse(
            state=state.tolist(),
            reward=reward,
            done=done,
            truncated=truncated,
            info=info
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/legal_actions")
def get_legal_actions():
    if env.game is None:
        raise HTTPException(status_code=400, detail="Game not started. Call /reset first.")
    return {"legal_actions": [env._format_action(a) for a in env.game.legal_actions(0)]}

# ------------------ Запуск ------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
