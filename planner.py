def decide(obstacle, game_speed, cfg):
    if not obstacle['present']:
        return 'none'
    p = cfg['planner']
    rd = p['base_reaction_distance'] + p['speed_factor'] * game_speed
    if obstacle['distance'] is None or obstacle['distance'] > rd:
        return 'none'
    if obstacle['type'] == 'ground':
        return 'jump'
    # flying: higher Y = lower on screen = too low to duck under → jump
    if obstacle['height'] is not None and obstacle['height'] > p['duck_height_threshold']:
        return 'jump'
    return 'duck'
