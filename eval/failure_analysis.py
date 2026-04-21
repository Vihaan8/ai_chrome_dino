import argparse
import glob
import json
import os


def categorize(run):
    """
    Classify why the episode ended.
      survived        — hit the frame cap without dying
      missed_detection — nearby raw obstacle but perception said absent (most of tail)
      misclassification — perception's type mismatched the real one (ahead-of-dino only)
      late_reaction   — first detection already at small distance
      timing_error    — agent tried to dodge (jump/duck) but still hit
    """
    if run.get('hit_cap'):
        return 'survived'

    log = run['log']
    if not log:
        return 'timing_error'
    tail = log[-15:]

    # most tail frames missed a nearby obstacle
    missed = 0
    for f in tail:
        if f['obstacle']['present']:
            continue
        ahead = [o for o in f.get('obstacles_raw') or [] if o[0] + o[2] > 50]
        if ahead and min(ahead, key=lambda o: o[0])[0] < 200:
            missed += 1
    if missed > len(tail) // 2:
        return 'missed_detection'

    # misclassification: only trust type comparison when detection is ahead of dino
    # (distance <= 0 means the "detection" is the airborne dino, not a real obstacle)
    for f in tail:
        o = f['obstacle']
        if not o['present'] or o['distance'] is None or o['distance'] <= 0:
            continue
        ahead = [r for r in f.get('obstacles_raw') or [] if r[0] + r[2] > 50]
        if not ahead:
            continue
        nearest = min(ahead, key=lambda r: r[0])
        if o['type'] != nearest[4]:
            return 'misclassification'

    # late_reaction: agent never saw the obstacle far away in the tail
    detected = [f for f in tail if f['obstacle']['present']
                and f['obstacle']['distance'] is not None
                and f['obstacle']['distance'] > 0]
    if detected and detected[0]['obstacle']['distance'] < 40:
        return 'late_reaction'

    return 'timing_error'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'runs'))
    args = ap.parse_args()

    cats = {'survived': 0, 'missed_detection': 0, 'misclassification': 0,
            'late_reaction': 0, 'timing_error': 0}
    files = sorted(glob.glob(os.path.join(args.runs, 'run_*.json')))
    if not files:
        print(f'no runs found in {args.runs}')
        return

    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        cats[categorize(data)] += 1

    n = len(files)
    print(f'analyzed {n} runs')
    for k, v in cats.items():
        print(f'  {k:20s} {v:3d}  ({100*v/n:5.1f}%)')


if __name__ == '__main__':
    main()
