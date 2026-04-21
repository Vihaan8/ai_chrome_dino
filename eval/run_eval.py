import argparse
import copy
import importlib
import json
import os
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from app.game import Game
from app import controller


def load_config():
    with open(os.path.join(ROOT, 'app', 'config.yaml')) as f:
        return yaml.safe_load(f)


def load_impl(name):
    if name == 'classical':
        return importlib.import_module('perception'), importlib.import_module('planner')
    if name == 'dl':
        return importlib.import_module('perception_dl'), importlib.import_module('planner_dl')
    raise ValueError(f'unknown impl: {name}')


def run_one(game, cfg, seed, perception, planner):
    game.reset(seed=seed)
    cap = cfg['eval']['max_frames']
    log = []
    t_perc = 0.0
    t_plan = 0.0
    while not game.done and game.frame < cap:
        fr = game.get_frame()
        a = time.perf_counter()
        obs = perception.detect(fr, cfg)
        b = time.perf_counter()
        action = planner.decide(obs, game.game_speed, cfg)
        c = time.perf_counter()
        t_perc += (b - a) * 1000
        t_plan += (c - b) * 1000
        log.append({
            'frame': game.frame,
            'game_speed': round(game.game_speed, 3),
            'obstacle': obs,
            'action': action,
            'dino_y': game.dino_y,
            'ducking': game.ducking,
            'obstacles_raw': copy.deepcopy(game.obstacles),
        })
        controller.apply(action, game)
    n = max(1, game.frame)
    return {
        'score': game.score,
        'frames': game.frame,
        'hit_cap': not game.done,
        'final_speed': round(game.game_speed, 3),
        'obstacles_cleared': game.obstacles_cleared,
        'killer': game.killer,
        'avg_perception_ms': t_perc / n,
        'avg_planning_ms': t_plan / n,
        'log': log,
    }


def stats(xs):
    xs = sorted(xs)
    n = len(xs)
    mean = sum(xs) / n
    med = xs[n // 2]
    var = sum((x - mean) ** 2 for x in xs) / n
    return {'mean': round(mean, 1), 'median': med, 'max': max(xs),
            'min': min(xs), 'stdev': round(var ** 0.5, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=None)
    ap.add_argument('--impl', choices=['classical', 'dl'], default='classical')
    ap.add_argument('--outdir', default=os.path.join(ROOT, 'eval', 'runs'))
    args = ap.parse_args()

    cfg = load_config()
    perception, planner = load_impl(args.impl)
    episodes = args.episodes or cfg['eval']['episodes']
    seeds = cfg['eval']['seeds']
    save = cfg['eval']['save_runs']
    os.makedirs(args.outdir, exist_ok=True)

    game = Game(headless=True, fast=True)
    scores, frames, cleared, perc_ms, plan_ms = [], [], [], [], []
    caps = 0
    killers = {}

    for i in range(episodes):
        seed = seeds[i % len(seeds)] + (i // len(seeds)) * 1000
        res = run_one(game, cfg, seed, perception, planner)
        scores.append(res['score'])
        frames.append(res['frames'])
        cleared.append(res['obstacles_cleared'])
        perc_ms.append(res['avg_perception_ms'])
        plan_ms.append(res['avg_planning_ms'])
        if res['hit_cap']:
            caps += 1
        else:
            k = res['killer']['type'] if res['killer'] else 'unknown'
            killers[k] = killers.get(k, 0) + 1

        if save:
            fname = f'run_{args.impl}_{seed}_{i}.json'
            with open(os.path.join(args.outdir, fname), 'w') as f:
                json.dump({
                    'impl': args.impl,
                    'seed': seed,
                    'score': res['score'],
                    'frames': res['frames'],
                    'hit_cap': res['hit_cap'],
                    'final_speed': res['final_speed'],
                    'obstacles_cleared': res['obstacles_cleared'],
                    'killer': res['killer'],
                    'avg_perception_ms': res['avg_perception_ms'],
                    'avg_planning_ms': res['avg_planning_ms'],
                    'log': res['log'],
                }, f)

        tag = ' (cap)' if res['hit_cap'] else f' died@spd={res["final_speed"]}'
        print(f'[{i+1}/{episodes}] seed={seed} score={res["score"]} '
              f'cleared={res["obstacles_cleared"]}{tag}')

    thresholds = [1000, 5000, cfg['eval']['max_frames']]
    print('--- summary ---')
    print(f'impl={args.impl} episodes={episodes}')
    print('score:             ', stats(scores))
    print('obstacles cleared: ', stats(cleared))
    for t in thresholds:
        hits = sum(1 for s in scores if s >= t)
        print(f'  % reaching {t:>5d}: {100*hits/episodes:5.1f}%')
    if killers:
        print('death cause:')
        for k, v in sorted(killers.items(), key=lambda kv: -kv[1]):
            print(f'  {k:10s} {v:3d}  ({100*v/episodes:5.1f}%)')
    print(f'avg perception: {sum(perc_ms)/len(perc_ms):.3f} ms/frame')
    print(f'avg planning:   {sum(plan_ms)/len(plan_ms):.3f} ms/frame')


if __name__ == '__main__':
    main()
