#!/usr/bin/env python3
"""
Pygame City Builder — single file

Controls
--------
Left-click: place selected item on grid
Right-click: bulldoze
Number keys 1–6: select tool (1 Road, 2 House, 3 Farm, 4 Factory, 5 Plant, 6 Bulldoze)
S: Quick-save to city_save.json
L: Load from city_save.json
R: Reset new map
Esc: Switch to Bulldoze

Buildings & Effects
-------------------
Road ($5): enables house adjacency
House ($20): +population capacity (5) if on-road and powered; consumes 1 food/tick; pays $2/tick if occupied; +happiness when fed/powered, -happiness otherwise
Farm ($30): +3 food/tick
Factory ($60): +10 money/tick, -2 happiness/tick, power demand 2
Power Plant ($80): +20 power capacity (global)

Powered rule (simple): If total demand <= total capacity, all demanders are powered; otherwise none are (simple global model).

Author: ChatGPT (MIT License)
"""
from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple, Optional, Dict

import pygame as pg

# -------------------------
# Configuration
# -------------------------
GRID_W, GRID_H = 24, 16
TILE = 36
SIDEPANEL_W = 260
HUD_H = 64
WINDOW_W = GRID_W * TILE + SIDEPANEL_W
WINDOW_H = GRID_H * TILE + HUD_H

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
COL_GRIDLINE = (36, 36, 36)
COL_TEXT = (230, 230, 230)
COL_SUB = (170, 170, 170)
COL_ACCENT = (76, 175, 80)
COL_WARN = (230, 100, 70)

class B(Enum):
    EMPTY = auto()
    ROAD = auto()
    HOUSE = auto()
    FARM = auto()
    FACTORY = auto()
    PLANT = auto()

BUILD_ORDER = [B.ROAD, B.HOUSE, B.FARM, B.FACTORY, B.PLANT]
TOOLS = BUILD_ORDER + [B.EMPTY]  # EMPTY is bulldoze

COLOR = {
    B.EMPTY: (43, 43, 43),
    B.ROAD: (142, 142, 142),
    B.HOUSE: (76, 175, 80),
    B.FARM: (163, 217, 119),
    B.FACTORY: (199, 146, 234),
    B.PLANT: (255, 213, 79),
}

OUTLINE = (26, 26, 26)

@dataclass
class Tile:
    kind: B = B.EMPTY

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

        # Happiness dynamics
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

    def set_flash(self, msg: str, secs: float = 1.6):
        self.flash_msg = msg
        self.flash_timer = secs

class App:
    def __init__(self):
        pg.init()
        pg.display.set_caption("Pygame City Builder")
        self.screen = pg.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont(None, 22)
        self.font_small = pg.font.SysFont(None, 18)
        self.font_big = pg.font.SysFont(None, 28)
        self.game = Game()
        self.running = True

        # Precompute sidebar button rects
        self.buttons: List[Tuple[pg.Rect, str, B]] = []
        labels = [
            ("Road", B.ROAD),
            ("House", B.HOUSE),
            ("Farm", B.FARM),
            ("Factory", B.FACTORY),
            ("Power Plant", B.PLANT),
            ("Bulldoze", B.EMPTY),
        ]
        x0 = GRID_W * TILE + 14
        for i, (label, kind) in enumerate(labels):
            y = HUD_H + 18 + i*56
            rect = pg.Rect(x0, y, SIDEPANEL_W-28, 42)
            self.buttons.append((rect, label, kind))

    # ---------------- Drawing ----------------
    def draw(self):
        self.screen.fill(COL_BG)
        self.draw_hud()
        self.draw_grid()
        self.draw_sidebar()
        self.draw_flash()
        pg.display.flip()

    def draw_hud(self):
        pg.draw.rect(self.screen, (30,30,30), pg.Rect(0,0,WINDOW_W,HUD_H))
        g = self.game
        powered = g.power_demand <= g.power_capacity
        text = (
            f"Money: ${g.money}   Food: {g.food}   "
            #f"Power: {g.power_capacity}/{g.power_demand} {'\u2713' if powered else '\u2717'}   "
            f"Population: {g.population}   Happiness: {g.happiness:.1f}"
        )

        self.blit_text(text, 12, 12, COL_TEXT, self.font_big)
        self.blit_text("1)Road  2)House  3)Farm  4)Factory  5)Plant  6)Bulldoze   S)Save  L)Load  R)Reset", 12, 38, COL_SUB)

    def draw_grid(self):
        y0 = HUD_H
        # tiles
        for y in range(GRID_H):
            for x in range(GRID_W):
                tx = x * TILE
                ty = y0 + y * TILE
                kind = self.game.grid[y][x].kind
                pg.draw.rect(self.screen, COLOR[kind], pg.Rect(tx, ty, TILE, TILE))
                pg.draw.rect(self.screen, OUTLINE, pg.Rect(tx, ty, TILE, TILE), 1)
                # glyphs
                if kind == B.HOUSE:
                    pg.draw.rect(self.screen, (46, 125, 50), pg.Rect(tx+10, ty+16, 16, 12))
                    pg.draw.polygon(self.screen, (102, 187, 106), [(tx+8,ty+16),(tx+26,ty+16),(tx+17,ty+9)])
                elif kind == B.FARM:
                    pg.draw.line(self.screen, (90,90,90), (tx+6,ty+24),(tx+TILE-6,ty+24))
                    pg.draw.line(self.screen, (90,90,90), (tx+6,ty+19),(tx+TILE-6,ty+19))
                elif kind == B.FACTORY:
                    pg.draw.rect(self.screen, (156,108,211), pg.Rect(tx+8,ty+20,18,10))
                    pg.draw.rect(self.screen, (181,138,230), pg.Rect(tx+10,ty+14,5,6))
                    pg.draw.rect(self.screen, (181,138,230), pg.Rect(tx+17,ty+11,5,9))
                elif kind == B.PLANT:
                    pg.draw.circle(self.screen, (255, 224, 130), (tx+18,ty+18), 10)
                    pg.draw.line(self.screen, (200,200,200), (tx+18,ty+6),(tx+18,ty+30))
                elif kind == B.ROAD:
                    pg.draw.line(self.screen, (210,210,210), (tx,ty+18),(tx+TILE,ty+18), 3)
        # gridlines overlay
        for x in range(GRID_W+1):
            tx = x*TILE
            pg.draw.line(self.screen, COL_GRIDLINE, (tx, HUD_H), (tx, HUD_H + GRID_H*TILE))
        for y in range(GRID_H+1):
            ty = HUD_H + y*TILE
            pg.draw.line(self.screen, COL_GRIDLINE, (0, ty), (GRID_W*TILE, ty))

    def draw_sidebar(self):
        x0 = GRID_W * TILE
        pg.draw.rect(self.screen, COL_PANEL, pg.Rect(x0, HUD_H, SIDEPANEL_W, WINDOW_H-HUD_H))
        # buttons
        for i, (rect, label, kind) in enumerate(self.buttons):
            mouse_over = rect.collidepoint(pg.mouse.get_pos())
            base = (36,36,36) if not mouse_over else (48,48,48)
            pg.draw.rect(self.screen, base, rect, border_radius=8)
            pg.draw.rect(self.screen, (60,60,60), rect, 2, border_radius=8)
            selected = (TOOLS[self.game.tool_index] == kind)
            if selected:
                pg.draw.rect(self.screen, COL_ACCENT, rect.inflate(-4,-4), 2, border_radius=6)
            self.blit_text(label, rect.x+12, rect.y+10, COL_TEXT)
            if kind != B.EMPTY:
                cost = COST[kind.name]
                self.blit_text(f"${cost}", rect.right-10, rect.y+10, COL_SUB, align_right=True)

        # tips
        tips = [
            "Left-click to build.",
            "Right-click to bulldoze.",
            "Houses need a road + power.",
            "Factories earn $, hurt happiness.",
            "Farms make food.",
        ]
        ty = HUD_H + 18 + len(self.buttons)*56 + 6
        for t in tips:
            self.blit_text(t, x0+16, ty, (155,155,155))
            ty += 20

    def draw_flash(self):
        if self.game.flash_msg and self.game.flash_timer > 0:
            msg = self.game.flash_msg
            surf = self.font_big.render(msg, True, (255,255,255))
            pad = 10
            rect = surf.get_rect()
            bx = 12
            by = 12 + 26
            back = pg.Rect(bx-8, by-4, rect.w+pad*2, rect.h+pad)
            pg.draw.rect(self.screen, (0,0,0,120), back)
            self.screen.blit(surf, (bx, by))

    # --------------- Interaction ---------------
    def grid_from_mouse(self, pos: Tuple[int,int]) -> Optional[Tuple[int,int]]:
        mx, my = pos
        if my < HUD_H: return None
        if mx >= GRID_W * TILE: return None
        gx = mx // TILE
        gy = (my - HUD_H) // TILE
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
            g.set_flash("Not enough money", 1.2)
            return
        # allow placing house off-road, but will be unoccupied until connected.
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

    def handle_mouse(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 1:  # left
                # sidebar buttons
                for rect, label, kind in self.buttons:
                    if rect.collidepoint(event.pos):
                        self.game.tool_index = TOOLS.index(kind)
                        return
                # grid placement
                cell = self.grid_from_mouse(event.pos)
                if cell:
                    x,y = cell
                    kind = TOOLS[self.game.tool_index]
                    self.place(x,y,kind)
            elif event.button == 3:  # right: bulldoze
                cell = self.grid_from_mouse(event.pos)
                if cell:
                    x,y = cell
                    self.place(x,y,B.EMPTY)

    def handle_keys(self, event):
        if event.type == pg.KEYDOWN:
            k = event.key
            if k in (pg.K_1,pg.K_2,pg.K_3,pg.K_4,pg.K_5,pg.K_6):
                self.game.tool_index = {pg.K_1:0,pg.K_2:1,pg.K_3:2,pg.K_4:3,pg.K_5:4,pg.K_6:5}[k]
            elif k == pg.K_ESCAPE:
                self.game.tool_index = len(TOOLS)-1
            elif k == pg.K_s:
                self.quick_save()
            elif k == pg.K_l:
                self.quick_load()
            elif k == pg.K_r:
                self.game = Game()
                self.game.set_flash("New city")

    # --------------- Main Loop ---------------
    def run(self):
        while self.running:
            dt_ms = self.clock.tick(FPS)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False
                self.handle_mouse(event)
                self.handle_keys(event)

            # sim tick
            self.game.tick_ms_accum += dt_ms
            if self.game.tick_ms_accum >= TICK_MS:
                self.game.tick_ms_accum %= TICK_MS
                self.game.tick()

            # flash timer
            if self.game.flash_timer > 0:
                self.game.flash_timer -= dt_ms/1000.0
                if self.game.flash_timer <= 0:
                    self.game.flash_msg = None

            self.draw()
        pg.quit()

    # helpers
    def blit_text(self, text: str, x: int, y: int, color=(255,255,255), font=None, align_right=False):
        font = font or self.font
        surf = font.render(text, True, color)
        rect = surf.get_rect()
        if align_right:
            self.screen.blit(surf, (x-rect.w, y))
        else:
            self.screen.blit(surf, (x, y))


def main():
    App().run()

if __name__ == "__main__":
    main()
