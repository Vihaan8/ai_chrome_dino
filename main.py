import argparse
import yaml

from app.game import Game
from app import controller
import perception
import planner


def load_config(path='app/config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def run_episode(game, cfg):
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
    args = ap.parse_args()

    cfg = load_config()
    headless = not args.render or args.fast
    game = Game(headless=headless, fast=args.fast)

    scores = []
    for ep in range(args.episodes):
        seed = args.seed + ep if args.seed is not None else None
        game.reset(seed=seed)
        score = run_episode(game, cfg)
        scores.append(score)
        if game.done and game.killer:
            k = game.killer
            tag = f'killed by {k["type"]} at speed {k["game_speed"]}'
        else:
            tag = 'reached frame cap'
        print(f'episode {ep+1}: score={score} cleared={game.obstacles_cleared} — {tag}')

    if len(scores) > 1:
        print(f'mean={sum(scores)/len(scores):.1f} max={max(scores)} min={min(scores)}')


if __name__ == '__main__':
    main()
