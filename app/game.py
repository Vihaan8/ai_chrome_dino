import os
import random
import pygame

SCREEN_W = 600
SCREEN_H = 200
GROUND_Y = 160

DINO_X = 50
DINO_W = 40
DINO_H_STAND = 40
DINO_H_DUCK = 20

GRAVITY = 0.8
JUMP_V = -14.0

START_SPEED = 6.0
SPEED_INC = 0.004

SPAWN_MIN_GAP = 55
SPAWN_MAX_GAP = 140

CACTUS_W = 20
CACTUS_H = 40
PTERO_W = 40
PTERO_H = 20
PTERO_HIGH_Y = 108
PTERO_LOW_Y = 135

# Decoys: light-blue cloud-shaped shapes that drift through the play area.
# They do not collide but perception still detects them (config.yaml threshold
# was raised to 220 so the light color still falls under the dark-pixel filter).
# The shape and color are intentionally obstacle-unlike; distinguishing them
# from real cacti is the job DL perception is meant to do better than classical.
CLOUD_W = 56           # 14 grid cells at 4x scale
CLOUD_H = 24           # 6 grid rows at 4x scale
CLOUD_Y_MIN = 85       # top Y range; cloud bottom ranges from 109 to 159
CLOUD_Y_MAX = 135
CLOUD_COLOR = (170, 200, 230)  # light sky blue
SPAWN_CLOUD_MIN = 160
SPAWN_CLOUD_MAX = 340

BG = (247, 247, 247)
FG = (83, 83, 83)

SCALE = 4  # each grid cell is 4x4 screen pixels

DINO_STAND_A = """
......####
......####
......#.##
......####
.....#####
##########
.#########
..######.#
..##...###
..##....##
""".strip('\n')

DINO_STAND_B = """
......####
......####
......#.##
......####
.....#####
##########
.#########
..######.#
..###..###
...##...##
""".strip('\n')

DINO_DUCK_A = """
........##
.########.
##########
.########.
..##...##.
""".strip('\n')

DINO_DUCK_B = """
........##
.########.
##########
.########.
...##.##..
""".strip('\n')

CACTUS = """
..#..
..#..
..#..
..#..
#.#.#
#####
#####
.###.
..#..
..#..
""".strip('\n')

PTERO_UP = """
#.........
###......#
.#########
....######
......##..
""".strip('\n')

CLOUD = """
....####......
..#########...
.############.
##############
.############.
..#########...
""".strip('\n')

PTERO_DOWN = """
.........#
.........#
##########
#####.####
###....##.
""".strip('\n')


def _sprite(grid, color=FG, scale=SCALE):
    rows = grid.split('\n')
    h = len(rows)
    w = max(len(r) for r in rows)
    surf = pygame.Surface((w * scale, h * scale), pygame.SRCALPHA)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == '#':
                surf.fill(color, (x * scale, y * scale, scale, scale))
    return surf


class Game:
    def __init__(self, headless=True, fast=False):
        self.headless = headless
        self.fast = fast
        if headless:
            os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        pygame.init()
        if headless:
            self.surface = pygame.Surface((SCREEN_W, SCREEN_H))
        else:
            self.surface = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            pygame.display.set_caption('dino')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('monospace', 14, bold=True) if not headless else None
        self._build_sprites()
        self.reset()

    def _build_sprites(self):
        self.spr_dino_run = [_sprite(DINO_STAND_A), _sprite(DINO_STAND_B)]
        self.spr_dino_duck = [_sprite(DINO_DUCK_A), _sprite(DINO_DUCK_B)]
        self.spr_cactus = _sprite(CACTUS)
        self.spr_ptero = [_sprite(PTERO_UP), _sprite(PTERO_DOWN)]
        self.spr_cloud = _sprite(CLOUD, color=CLOUD_COLOR)

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.dino_y = GROUND_Y - DINO_H_STAND
        self.dino_vy = 0.0
        self.dino_h = DINO_H_STAND
        self.ducking = False
        self.on_ground = True
        self.obstacles = []
        self.next_spawn = self.rng.randint(SPAWN_MIN_GAP, SPAWN_MAX_GAP)
        self.next_cloud = self.rng.randint(SPAWN_CLOUD_MIN, SPAWN_CLOUD_MAX)
        self.frame = 0
        self.score = 0
        self.done = False
        self.game_speed = START_SPEED
        self.obstacles_cleared = 0
        self.decoys_triggered = 0
        self.killer = None
        self._render()

    def step(self, action):
        if self.done:
            return

        if action == 'jump' and self.on_ground:
            self.dino_vy = JUMP_V
            self.on_ground = False
            self.ducking = False
        elif action == 'duck' and self.on_ground:
            self.ducking = True
        elif self.on_ground:
            self.ducking = False

        if not self.on_ground:
            self.dino_vy += GRAVITY
            self.dino_y += self.dino_vy
            if self.dino_y >= GROUND_Y - DINO_H_STAND:
                self.dino_y = GROUND_Y - DINO_H_STAND
                self.dino_vy = 0.0
                self.on_ground = True

        if self.ducking and self.on_ground:
            self.dino_h = DINO_H_DUCK
            self.dino_y = GROUND_Y - DINO_H_DUCK
        else:
            self.dino_h = DINO_H_STAND
            if self.on_ground:
                self.dino_y = GROUND_Y - DINO_H_STAND

        self.next_spawn -= 1
        if self.next_spawn <= 0:
            self._spawn()
            self.next_spawn = self.rng.randint(SPAWN_MIN_GAP, SPAWN_MAX_GAP)

        self.next_cloud -= 1
        if self.next_cloud <= 0:
            self._spawn_cloud()
            self.next_cloud = self.rng.randint(SPAWN_CLOUD_MIN, SPAWN_CLOUD_MAX)

        for o in self.obstacles:
            o[0] -= self.game_speed
        kept = []
        for o in self.obstacles:
            if o[0] + o[2] > 0:
                kept.append(o)
            elif o[4] != 'decoy':
                self.obstacles_cleared += 1
        self.obstacles = kept

        dino_rect = (DINO_X, self.dino_y, DINO_W, self.dino_h)
        for o in self.obstacles:
            if o[4] == 'decoy':
                continue
            if self._hit(dino_rect, o):
                self.done = True
                self.killer = {'type': o[4], 'x': o[0], 'y': o[1],
                               'w': o[2], 'h': o[3],
                               'game_speed': round(self.game_speed, 3)}
                break

        self.frame += 1
        self.score = self.frame
        self.game_speed += SPEED_INC
        self._render()
        if not self.fast and not self.headless:
            pygame.event.pump()
            self.clock.tick(60)

    def _spawn(self):
        r = self.rng.random()
        if r < 0.65:
            self.obstacles.append([float(SCREEN_W), float(GROUND_Y - CACTUS_H),
                                   CACTUS_W, CACTUS_H, 'ground'])
        elif r < 0.85:
            self.obstacles.append([float(SCREEN_W), float(PTERO_HIGH_Y),
                                   PTERO_W, PTERO_H, 'flying'])
        else:
            self.obstacles.append([float(SCREEN_W), float(PTERO_LOW_Y),
                                   PTERO_W, PTERO_H, 'flying'])

    def _spawn_cloud(self):
        y = self.rng.randint(CLOUD_Y_MIN, CLOUD_Y_MAX)
        self.obstacles.append([float(SCREEN_W), float(y),
                               CLOUD_W, CLOUD_H, 'decoy'])

    def _hit(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b[0], b[1], b[2], b[3]
        return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by

    def _dino_sprite(self):
        anim = (self.frame // 5) % 2
        if self.ducking and self.on_ground:
            return self.spr_dino_duck[anim]
        return self.spr_dino_run[anim]

    def _obstacle_sprite(self, o):
        if o[4] == 'ground':
            return self.spr_cactus
        if o[4] == 'decoy':
            return self.spr_cloud
        anim = (self.frame // 8) % 2
        return self.spr_ptero[anim]

    def _render(self):
        self.surface.fill(BG)
        pygame.draw.rect(self.surface, FG, (0, GROUND_Y, SCREEN_W, 2))
        self.surface.blit(self._dino_sprite(), (DINO_X, int(self.dino_y)))
        for o in self.obstacles:
            self.surface.blit(self._obstacle_sprite(o), (int(o[0]), int(o[1])))
        if not self.headless and self.font is not None:
            text = f'score {self.score:5d}   speed {self.game_speed:5.2f}   cleared {self.obstacles_cleared}'
            img = self.font.render(text, True, FG)
            self.surface.blit(img, (SCREEN_W - img.get_width() - 8, 8))
        if not self.headless:
            pygame.display.flip()

    def get_frame(self):
        arr = pygame.surfarray.array3d(self.surface)
        arr = arr.transpose(1, 0, 2)
        return arr[:, :, ::-1].copy()
