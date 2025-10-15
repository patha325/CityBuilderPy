#!/usr/bin/env python3
"""
Pygame City Builder — single file, **software 3D** (no extra libs)

This version renders the same grid-based city in a lightweight 3D using
manual projection and a painter's algorithm (back-to-front polygon draw).

Controls
--------
Left click: place selected item on grid
Right click: bulldoze
Number keys 1–6: select tool (1 Road, 2 House, 3 Farm, 4 Factory, 5 Plant, 6 Bulldoze)
S: Quick-save to city_save.json
L: Load from city_save.json
R: Reset new map

Camera / View
------------
W/A/S/D: move on ground plane
Q/E: rotate yaw (left/right)
R/F: pitch up/down (tilt)
Mouse wheel: zoom in/out
Middle-drag: rotate (yaw) while dragging horizontally

Notes
-----
• Pure Pygame; no OpenGL required.
• Mouse picking uses a ray-plane intersection to determine which tile the cursor points to.
• Power model is global capacity vs demand (same as 2D versions).

Author: ChatGPT (MIT License)
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple, Optional, Dict

import pygame as pg

# -------------------------
# Configuration
# -------------------------
GRID_W, GRID_H = 24, 16
TILE_WORLD = 1.0  # world units per tile (XZ plane)
SIDEPANEL_W = 280
HUD_H = 64
BASE_W = 1280
BASE_H = 800
WINDOW_W = BASE_W
WINDOW_H = BASE_H

FPS = 60
TICK_MS = 700  # simulation tick interval

# Costs & economy
COST = {
    "EMPTY": 0,
    "ROAD": 5,
    "HOUSE": 20,
    "FARM": 30,
    "FACTORY": 60,
    "PLANT": 80,
}
HOUSE_CAPACITY = 5
HOUSE_TAX = 2
HOUSE_FOOD_USAGE = 1
FARM_FOOD = 3
FACTORY_MONEY = 10
FACTORY_POWER = 2
HOUSE_POWER = 1
PLANT_POWER = 20

HAPPINESS_DECAY = 0.05
HAPPINESS_MAX = 100.0
HAPPINESS_MIN = 0.0

SAVE_PATH = "city_save.json"

# Colors
COL_BG = (17, 17, 17)
COL_PANEL = (22, 22, 22)
COL_TEXT = (230, 230, 230)
COL_SUB = (170, 170, 170)
COL_ACCENT = (76, 175, 80)
COL_WARN = (230, 100, 70)

# Material colors (rgb)
COLORS = {
    "GROUND": (40, 40, 46),
    "GRID": (55, 55, 60),
}

class B(Enum):
    EMPTY = auto()
    ROAD = auto()
    HOUSE = auto()
    FARM = auto()
    FACTORY = auto()
    PLANT = auto()

BUILD_ORDER = [B.ROAD, B.HOUSE, B.FARM, B.FACTORY, B.PLANT]
TOOLS = BUILD_ORDER + [B.EMPTY]  # EMPTY is bulldoze

# Styling per building (base color and height in world units)
STYLE = {
    B.EMPTY: ((50, 50, 50), 0.01),
    B.ROAD: ((110, 110, 110), 0.03),
    B.HOUSE: ((76, 175, 80), 0.40),
    B.FARM: ((163, 217, 119), 0.12),
    B.FACTORY: ((199, 146, 234), 0.65),
    B.PLANT: ((255, 213, 79), 0.90),
}

@dataclass
class Tile:
    kind: B = B.EMPTY

# -------------------------
# Game Logic
# -------------------------
class Game:
    def __init__(self):
        self.grid: List[List[Tile]] = [[Tile() for _ in range(GRID_W)] for _ in range(GRID_H)]
        self.money: int = 200
        self.food: int = 0
        self.happiness: float = 70.0
        self.population: int = 0
        self.power_capacity: int = 0
        self.power_demand: int = 0
        self.tool_index: int = 0
        self.tick_ms_accum: int = 0
        self.tick_count: int = 0
        self.flash_msg: Optional[str] = None
        self.flash_timer: float = 0.0

    def counts(self) -> Dict[B, int]:
        c = {b:0 for b in B}
        for y in range(GRID_H):
            for x in range(GRID_W):
                c[self.grid[y][x].kind] += 1
        return c

    def adjacent_to_road(self, x: int, y: int) -> bool:
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x+dx, y+dy
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                if self.grid[ny][nx].kind == B.ROAD:
                    return True
        return False

    def compute_power(self) -> bool:
        counts = self.counts()
        capacity = counts[B.PLANT] * PLANT_POWER
        demand = counts[B.HOUSE] * HOUSE_POWER + counts[B.FACTORY] * FACTORY_POWER
        self.power_capacity = capacity
        self.power_demand = demand
        return demand <= capacity

    def occupied_houses(self, powered: bool) -> int:
        occ = 0
        for y in range(GRID_H):
            for x in range(GRID_W):
                if self.grid[y][x].kind == B.HOUSE:
                    if self.adjacent_to_road(x, y) and powered:
                        occ += 1
        return occ

    def tick(self):
        self.tick_count += 1
        powered = self.compute_power()
        counts = self.counts()
        farms = counts[B.FARM]
        factories = counts[B.FACTORY]

        occ_houses = self.occupied_houses(powered)
        self.population = occ_houses * HOUSE_CAPACITY

        # Food
        self.food += farms * FARM_FOOD
        food_needed = occ_houses * HOUSE_FOOD_USAGE
        fed = min(self.food, food_needed)
        self.food -= fed
        starving = food_needed - fed

        # Money
        self.money += factories * FACTORY_MONEY
        self.money += occ_houses * HOUSE_TAX

        # Happiness
        dh = 0.0
        dh += occ_houses * (0.2 if powered else -0.3)
        dh += -2.0 * factories
        dh += -1.5 * starving
        if self.tick_count % 5 == 0:
            if self.happiness < 70:
                dh += 0.5
            elif self.happiness > 70:
                dh -= 0.5
        if self.happiness > 50:
            self.happiness -= HAPPINESS_DECAY
        elif self.happiness < 50:
            self.happiness += HAPPINESS_DECAY
        self.happiness = max(HAPPINESS_MIN, min(HAPPINESS_MAX, self.happiness + dh))

    def set_flash(self, msg: str, secs: float = 1.3):
        self.flash_msg = msg
        self.flash_timer = secs

# -------------------------
# 3D Math Helpers
# -------------------------
import numpy as np  # NOTE: Not used—keeping pure math without numpy.

def deg2rad(a: float) -> float:
    return a * math.pi / 180.0

class Camera:
    def __init__(self):
        # Start offset so the whole grid is visible
        self.pos = [GRID_W * 0.5, 3.0, -GRID_H * 1.2]  # x,y,z (y=up)
        self.yaw = 30.0   # rotate around Y
        self.pitch = 25.0 # look down slightly
        self.fov = 70.0   # degrees
        self.near = 0.1
        self.far = 100.0

    def move_local(self, dx: float, dz: float):
        # Move in camera's XZ plane
        yaw = deg2rad(self.yaw)
        cx, sx = math.cos(yaw), math.sin(yaw)
        # Right vector (in world XZ)
        rx, rz = cx, -sx
        # Forward vector (on ground plane)
        fx, fz = sx, cx
        self.pos[0] += rx*dx + fx*dz
        self.pos[2] += rz*dx + fz*dz

    def rotate(self, dyaw: float, dpitch: float):
        self.yaw = (self.yaw + dyaw) % 360
        self.pitch = max(-80.0, min(80.0, self.pitch + dpitch))

    def zoom(self, delta: float):
        self.fov = max(35.0, min(100.0, self.fov + delta))

    def world_to_camera(self, p: Tuple[float,float,float]) -> Tuple[float,float,float]:
        # Translate
        x = p[0] - self.pos[0]
        y = p[1] - self.pos[1]
        z = p[2] - self.pos[2]
        # Apply yaw (around Y)
        yaw = deg2rad(self.yaw)
        cy, sy = math.cos(yaw), math.sin(yaw)
        xz = x*cy - z*sy
        zz = x*sy + z*cy
        # Apply pitch (around X)
        pitch = deg2rad(self.pitch)
        cp, sp = math.cos(pitch), math.sin(pitch)
        yy = y*cp - zz*sp
        zz2 = y*sp + zz*cp
        return (xz, yy, zz2)

    def project(self, p_cam: Tuple[float,float,float], w: int, h: int) -> Optional[Tuple[int,int,float]]:
        x, y, z = p_cam
        if z <= self.near:
            return None
        f = 1.0 / math.tan(deg2rad(self.fov)*0.5)
        nx = (x * f) / z
        ny = (y * f) / z
        sx = int((nx * 0.5 + 0.5) * (w - SIDEPANEL_W))
        sy = int(((-ny) * 0.5 + 0.5) * (h - 0))
        return (sx, sy, z)

    def screen_ray(self, mx: int, my: int, w: int, h: int) -> Tuple[Tuple[float,float,float], Tuple[float,float,float]]:
        # Convert screen pixel to NDC (-1..1)
        vw = w - SIDEPANEL_W
        ndc_x = (mx / max(1, vw)) * 2.0 - 1.0
        ndc_y = 1.0 - (my / max(1, h)) * 2.0
        f = 1.0 / math.tan(deg2rad(self.fov)*0.5)
        # Ray in camera space
        rx = ndc_x / f
        ry = ndc_y / f
        rz = 1.0
        # Rotate by inverse pitch then inverse yaw to world space
        pitch = deg2rad(self.pitch)
        cp, sp = math.cos(pitch), math.sin(pitch)
        y1 = ry*cp + rz*sp
        z1 = -ry*sp + rz*cp
        yaw = deg2rad(self.yaw)
        cy, sy = math.cos(yaw), math.sin(yaw)
        x2 = rx*cy + z1*sy
        z2 = -rx*sy + z1*cy
        # Normalize
        l = math.sqrt(x2*x2 + y1*y1 + z2*z2) + 1e-9
        dirw = (x2/l, y1/l, z2/l)
        origin = tuple(self.pos)
        return origin, dirw

# -------------------------
# Renderer (software 3D)
# -------------------------
class Renderer:
    def __init__(self, screen: pg.Surface, camera: Camera):
        self.screen = screen
        self.cam = camera
        self.font = pg.font.SysFont(None, 22)
        self.font_small = pg.font.SysFont(None, 18)

    def draw_poly3d(self, verts3d, color):
            """Project and draw a filled 3D polygon (used for ground)."""
            v_cam = [self.cam.world_to_camera(v) for v in verts3d]
            pts2d = []
            for vx, vy, vz in v_cam:
                p = self.cam.project((vx, vy, vz), WINDOW_W, WINDOW_H)
                if p is None:
                    return  # skip if off-screen or behind camera
                sx, sy, _ = p
                pts2d.append((sx, sy))
            if len(pts2d) >= 3:
                try:
                    pg.draw.polygon(self.screen, color, pts2d)
                except Exception:
                    pass

    def draw_world(self, game: Game):
        self.screen.fill(COL_BG)
        # Ground plane rectangle
        ground_pts = [
            (0, 0, 0),
            (GRID_W*TILE_WORLD, 0, 0),
            (GRID_W*TILE_WORLD, 0, GRID_H*TILE_WORLD),
            (0, 0, GRID_H*TILE_WORLD),
        ]
        self.draw_poly3d(ground_pts, COLORS["GROUND"])
        # Grid lines (optional)
        grid_col = COLORS["GRID"]
        for x in range(GRID_W+1):
            self.draw_line3d((x*TILE_WORLD, 0.001, 0), (x*TILE_WORLD, 0.001, GRID_H*TILE_WORLD), grid_col)
        for y in range(GRID_H+1):
            self.draw_line3d((0, 0.001, y*TILE_WORLD), (GRID_W*TILE_WORLD, 0.001, y*TILE_WORLD), grid_col)

        # Collect all tile boxes as polygons, then sort by depth
        polys: List[Tuple[float, List[Tuple[int,int]], Tuple[int,int,int]]] = []
        for gy in range(GRID_H):
            for gx in range(GRID_W):
                kind = game.grid[gy][gx].kind
                base_col, height = STYLE[kind]
                if kind == B.EMPTY:
                    # draw subtle top only
                    self.add_top_quad(polys, gx, gy, 0.02, base_col)
                    continue
                self.add_box(polys, gx, gy, height, base_col)
        # Sort by average z (descending, far to near)
        polys.sort(key=lambda item: -item[0])
        # Draw
        for _, pts2d, col in polys:
            if len(pts2d) >= 3:
                try:
                    pg.draw.polygon(self.screen, col, pts2d)
                    pg.draw.polygon(self.screen, (25,25,25), pts2d, 1)
                except Exception:
                    pass

    def add_top_quad(self, polys, gx, gy, h, color):
        x0 = gx*TILE_WORLD
        z0 = gy*TILE_WORLD
        x1 = x0 + TILE_WORLD
        z1 = z0 + TILE_WORLD
        top = [(x0,h,z0),(x1,h,z0),(x1,h,z1),(x0,h,z1)]
        self.emit_face(polys, top, color)

    def add_box(self, polys, gx, gy, h, color):
        x0 = gx*TILE_WORLD
        z0 = gy*TILE_WORLD
        x1 = x0 + TILE_WORLD
        z1 = z0 + TILE_WORLD
        y0 = 0.0
        y1 = h
        # 6 faces: top, sides
        top = [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)]
        s1 = [(x0,y0,z0),(x0,y1,z0),(x1,y1,z0),(x1,y0,z0)]  # front
        s2 = [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)]  # right
        s3 = [(x1,y0,z1),(x1,y1,z1),(x0,y1,z1),(x0,y0,z1)]  # back
        s4 = [(x0,y0,z1),(x0,y1,z1),(x0,y1,z0),(x0,y0,z0)]  # left
        # Simple shading by face normal toward camera (fake): different tints
        def tint(col, f):
            return (max(0,min(255,int(col[0]*f))), max(0,min(255,int(col[1]*f))), max(0,min(255,int(col[2]*f))))
        self.emit_face(polys, s3, tint(color, 0.55))
        self.emit_face(polys, s4, tint(color, 0.7))
        self.emit_face(polys, s2, tint(color, 0.75))
        self.emit_face(polys, s1, tint(color, 0.9))
        self.emit_face(polys, top, tint(color, 1.0))

    def emit_face(self, polys, verts3d, color):
        # Backface cull (approx): compute normal in camera space and cull if facing away
        v_cam = [self.cam.world_to_camera(v) for v in verts3d]
        # If any behind near plane, still try but skip if too close
        if all(v[2] <= self.cam.near for v in v_cam):
            return
        # Normal from first three vertices
        ax, ay, az = v_cam[0]
        bx, by, bz = v_cam[1]
        cx, cy, cz = v_cam[2]
        ux, uy, uz = bx-ax, by-ay, bz-az
        vx, vy, vz = cx-ax, cy-ay, cz-az
        nx, ny, nz = uy*vz - uz*vy, uz*vx - ux*vz, ux*vy - uy*vx
        # Facing camera if nz < 0 (since camera looks along +z in our camera space)
        if nz >= 0:
            return
        # Project
        pts2d: List[Tuple[int,int]] = []
        zsum = 0.0
        for vx_, vy_, vz_ in v_cam:
            p = self.cam.project((vx_,vy_,vz_), WINDOW_W, WINDOW_H)
            if p is None:
                return
            sx, sy, z = p
            pts2d.append((sx, sy))
            zsum += z
        avgz = zsum / len(v_cam)
        polys.append((avgz, pts2d, color))

    def draw_line3d(self, a, b, color):
        pa = self.cam.world_to_camera(a)
        pb = self.cam.world_to_camera(b)
        pa2 = self.cam.project(pa, WINDOW_W, WINDOW_H)
        pb2 = self.cam.project(pb, WINDOW_W, WINDOW_H)
        if pa2 and pb2:
            pg.draw.line(self.screen, color, (pa2[0], pa2[1]), (pb2[0], pb2[1]))

    def draw_hud(self, game: Game, hovered: Optional[Tuple[int,int]], selected_kind: B):
        # HUD bar
        pg.draw.rect(self.screen, (30,30,30), pg.Rect(0,0,WINDOW_W,HUD_H))
        powered = game.power_demand <= game.power_capacity
        text = (
            f"Money: ${game.money}   Food: {game.food}   "
            f"Power: {game.power_capacity}/{game.power_demand} {'✓' if powered else '✗'}   "
            f"Population: {game.population}   Happiness: {game.happiness:.1f}"
        )
        self.blit(text, 12, 12, COL_TEXT, size=26)
        helptext = "WASD move  Q/E yaw  R/F pitch  Wheel zoom  S save  L load  R reset"
        self.blit(helptext, 12, 38, COL_SUB, size=18)
        # Hover indicator
        if hovered:
            gx, gy = hovered
            self.blit(f"Tile: ({gx},{gy})  Tool: {selected_kind.name}", 12, HUD_H-24, COL_SUB, size=18)

    def draw_sidebar(self, game: Game, buttons):
        x0 = WINDOW_W - SIDEPANEL_W
        pg.draw.rect(self.screen, COL_PANEL, pg.Rect(x0, HUD_H, SIDEPANEL_W, WINDOW_H-HUD_H))
        for rect, label, kind in buttons:
            mouse_over = rect.collidepoint(pg.mouse.get_pos())
            base = (36,36,36) if not mouse_over else (48,48,48)
            pg.draw.rect(self.screen, base, rect, border_radius=8)
            pg.draw.rect(self.screen, (60,60,60), rect, 2, border_radius=8)
            selected = (TOOLS[game.tool_index] == kind)
            if selected:
                pg.draw.rect(self.screen, COL_ACCENT, rect.inflate(-4,-4), 2, border_radius=6)
            self.blit(label, rect.x+12, rect.y+10, COL_TEXT)
            if kind != B.EMPTY:
                self.blit(f"${COST[kind.name]}", rect.right-10, rect.y+10, COL_SUB, align_right=True)

    def draw_flash(self, game: Game):
        if game.flash_msg and game.flash_timer > 0:
            surf = self.font.render(game.flash_msg, True, (255,255,255))
            rect = surf.get_rect()
            box = pg.Rect(12, HUD_H+8, rect.w+16, rect.h+12)
            pg.draw.rect(self.screen, (0,0,0), box)
            pg.draw.rect(self.screen, (255,255,255), box, 1)
            self.screen.blit(surf, (box.x+8, box.y+6))

    def blit(self, text: str, x: int, y: int, color, size: int = 22, align_right: bool=False):
        f = pg.font.SysFont(None, size)
        surf = f.render(text, True, color)
        rect = surf.get_rect()
        if align_right:
            self.screen.blit(surf, (x-rect.w, y))
        else:
            self.screen.blit(surf, (x, y))

# -------------------------
# Application
# -------------------------
class App:
    def __init__(self):
        pg.init()
        pg.display.set_caption("Pygame City Builder — 3D")
        self.screen = pg.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pg.time.Clock()
        self.game = Game()
        self.cam = Camera()
        self.ren = Renderer(self.screen, self.cam)
        self.running = True

        # Sidebar buttons
        self.buttons: List[Tuple[pg.Rect, str, B]] = []
        labels = [
            ("Road", B.ROAD),
            ("House", B.HOUSE),
            ("Farm", B.FARM),
            ("Factory", B.FACTORY),
            ("Power Plant", B.PLANT),
            ("Bulldoze", B.EMPTY),
        ]
        x0 = WINDOW_W - SIDEPANEL_W + 14
        for i, (label, kind) in enumerate(labels):
            y = HUD_H + 18 + i*56
            rect = pg.Rect(x0, y, SIDEPANEL_W-28, 42)
            self.buttons.append((rect, label, kind))

        # Mouse state
        self.dragging = False
        self.last_mx = 0

    # ----------- World helpers -----------
    def grid_from_mouse(self, mx: int, my: int) -> Optional[Tuple[int,int]]:
        # Ignore clicks on sidebar
        if mx >= WINDOW_W - SIDEPANEL_W or my < HUD_H:
            return None
        origin, direction = self.cam.screen_ray(mx, my, WINDOW_W, WINDOW_H)
        # Intersect with ground plane y=0: origin + t*dir; solve origin.y + t*dir.y = 0
        oy, dy = origin[1], direction[1]
        if abs(dy) < 1e-6:
            return None
        t = -oy / dy
        if t <= 0:  # behind camera
            return None
        ix = origin[0] + direction[0]*t
        iz = origin[2] + direction[2]*t
        gx = int(math.floor(ix / TILE_WORLD))
        gy = int(math.floor(iz / TILE_WORLD))
        if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
            return gx, gy
        return None

    def place(self, x: int, y: int, kind: B):
        g = self.game
        current = g.grid[y][x].kind
        if kind == B.EMPTY:
            if current != B.EMPTY:
                base = COST[current.name]
                refund = base//3 if current == B.ROAD else base//2
                g.money += refund
                g.grid[y][x].kind = B.EMPTY
            return
        if current != B.EMPTY:
            return
        price = COST[kind.name]
        if g.money < price:
            g.set_flash("Not enough money", 1.1)
            return
        g.money -= price
        g.grid[y][x].kind = kind

    def quick_save(self):
        data = {
            "money": self.game.money,
            "food": self.game.food,
            "happiness": self.game.happiness,
            "tick": self.game.tick_count,
            "grid": [[tile.kind.name for tile in row] for row in self.game.grid],
        }
        try:
            with open(SAVE_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self.game.set_flash(f"Saved to {SAVE_PATH}")
        except Exception as e:
            self.game.set_flash(f"Save failed: {e}")

    def quick_load(self):
        try:
            with open(SAVE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            g = Game()
            g.money = int(data.get("money", 200))
            g.food = int(data.get("food", 0))
            g.happiness = float(data.get("happiness", 70.0))
            g.tick_count = int(data.get("tick", 0))
            grid_names = data.get("grid", [])
            if grid_names:
                for y in range(min(GRID_H, len(grid_names))):
                    for x in range(min(GRID_W, len(grid_names[y]))):
                        name = grid_names[y][x]
                        try:
                            g.grid[y][x].kind = B[name]
                        except Exception:
                            g.grid[y][x].kind = B.EMPTY
            self.game = g
            self.game.set_flash(f"Loaded from {SAVE_PATH}")
        except Exception as e:
            self.game.set_flash(f"Load failed: {e}")

    # ----------- Main Loop -----------
    def run(self):
        while self.running:
            dt_ms = self.clock.tick(FPS)
            self.handle_events()

            # Simulation tick
            self.game.tick_ms_accum += dt_ms
            if self.game.tick_ms_accum >= TICK_MS:
                self.game.tick_ms_accum %= TICK_MS
                self.game.tick()

            # Flash
            if self.game.flash_timer > 0:
                self.game.flash_timer -= dt_ms/1000.0
                if self.game.flash_timer <= 0:
                    self.game.flash_msg = None

            # Render
            hovered = self.grid_from_mouse(*pg.mouse.get_pos())
            self.ren.draw_world(self.game)
            self.ren.draw_sidebar(self.game, self.buttons)
            self.ren.draw_hud(self.game, hovered, TOOLS[self.game.tool_index])
            self.ren.draw_flash(self.game)
            pg.display.flip()
        pg.quit()

    # ----------- Input -----------
    def handle_events(self):
        keys = pg.key.get_pressed()
        move = 3.0 / FPS
        if keys[pg.K_w]:
            self.cam.move_local(0, +move)
        if keys[pg.K_s]:
            self.cam.move_local(0, -move)
        if keys[pg.K_a]:
            self.cam.move_local(-move, 0)
        if keys[pg.K_d]:
            self.cam.move_local(+move, 0)
        if keys[pg.K_q]:
            self.cam.rotate(-60.0/FPS, 0)
        if keys[pg.K_e]:
            self.cam.rotate(+60.0/FPS, 0)
        if keys[pg.K_r]:
            self.cam.rotate(0, +45.0/FPS)
        if keys[pg.K_f]:
            self.cam.rotate(0, -45.0/FPS)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            elif event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1:  # left
                    # Sidebar buttons
                    for rect, label, kind in self.buttons:
                        if rect.collidepoint(event.pos):
                            self.game.tool_index = TOOLS.index(kind)
                            break
                    else:
                        cell = self.grid_from_mouse(*event.pos)
                        if cell:
                            self.place(cell[0], cell[1], TOOLS[self.game.tool_index])
                elif event.button == 3:  # right
                    cell = self.grid_from_mouse(*event.pos)
                    if cell:
                        self.place(cell[0], cell[1], B.EMPTY)
                elif event.button == 2:  # middle start drag
                    self.dragging = True
                    self.last_mx = event.pos[0]
                elif event.button == 4:  # wheel up
                    self.cam.zoom(-3)
                elif event.button == 5:  # wheel down
                    self.cam.zoom(+3)
            elif event.type == pg.MOUSEBUTTONUP:
                if event.button == 2:
                    self.dragging = False
            elif event.type == pg.MOUSEMOTION and self.dragging:
                dx = event.pos[0] - self.last_mx
                self.cam.rotate(dx * 0.2, 0)
                self.last_mx = event.pos[0]
            elif event.type == pg.KEYDOWN:
                if event.key in (pg.K_1,pg.K_2,pg.K_3,pg.K_4,pg.K_5,pg.K_6):
                    self.game.tool_index = {pg.K_1:0,pg.K_2:1,pg.K_3:2,pg.K_4:3,pg.K_5:4,pg.K_6:5}[event.key]
                elif event.key == pg.K_ESCAPE:
                    self.game.tool_index = len(TOOLS)-1
                elif event.key == pg.K_s:
                    self.quick_save()
                elif event.key == pg.K_l:
                    self.quick_load()
                elif event.key == pg.K_RSHIFT or event.key == pg.K_r:
                    # R is also used for pitch; only reset on capital R via Right Shift + R, or hold and tap R
                    if pg.key.get_mods() & pg.KMOD_SHIFT:
                        self.game = Game()
                        self.game.set_flash("New city")


# -------------------------
# Entry
# -------------------------

def main():
    App().run()

if __name__ == "__main__":
    main()
