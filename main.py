import argparse
import importlib
import yaml

from app.game import Game
from app import controller


def load_config(path='app/config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def load_impl(name):
    if name == 'classical':
        return importlib.import_module('classical.perception'), importlib.import_module('classical.planner')
    if name == 'dl':
        return importlib.import_module('dl.perception'), importlib.import_module('dl.planner')
    raise ValueError(f'unknown impl: {name}')


def run_episode(game, cfg, perception, planner):
    cap = cfg['eval']['max_frames']
    while not game.done and game.frame < cap:
        frame = game.get_frame()
        obs = perception.detect(frame, cfg)
        action = planner.decide(obs, game.game_speed, cfg)
        controller.apply(action, game)
    return game.score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--render', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--episodes', type=int, default=1)
    ap.add_argument('--impl', choices=['classical', 'dl'], default='classical')
    args = ap.parse_args()

    cfg = load_config()
    perception, planner = load_impl(args.impl)
    headless = not args.render or args.fast
    game = Game(headless=headless, fast=args.fast)

    scores = []
    for ep in range(args.episodes):
        seed = args.seed + ep if args.seed is not None else None
        game.reset(seed=seed)
        score = run_episode(game, cfg, perception, planner)
        scores.append(score)
        if game.done and game.killer:
            k = game.killer
            tag = f'killed by {k["type"]} at speed {k["game_speed"]}'
        else:
            tag = 'reached frame cap'
        print(f'[{args.impl}] episode {ep+1}: score={score} '
              f'cleared={game.obstacles_cleared}, {tag}')

    if len(scores) > 1:
        print(f'mean={sum(scores)/len(scores):.1f} max={max(scores)} min={min(scores)}')


if __name__ == '__main__':
    main()
