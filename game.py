#!/usr/bin/env python3
"""
Simple City Builder (single file, stdlib only)

Controls
--------
Left-click: place selected item
Right-click: bulldoze
Number keys 1–6: select tool
S: Save to city_save.json
L: Load from city_save.json
R: Reset new map
Esc: Cycle to Bulldoze

Buildings & Effects
-------------------
Road ($5): enables house adjacency
House ($20): +population capacity (5) if on-road and powered; consumes 1 food/tick; pays $2/tick if occupied; +happiness when fed/powered, -happiness otherwise
Farm ($30): +3 food/tick
Factory ($60): +10 money/tick, -2 happiness/tick, power demand 2
Power Plant ($80): +20 power capacity (global)

Powered rule (simple): If total demand <= total capacity, all demanders are powered; otherwise none are (keeps logic simple).

Author: ChatGPT
License: MIT (do as you wish)
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple, Optional, Dict

import tkinter as tk
from tkinter import messagebox, filedialog

# -------------------------
# Configuration
# -------------------------
GRID_W, GRID_H = 20, 15
TILE = 32
SIDEPANEL_W = 220
HUD_H = 56
WINDOW_W = GRID_W * TILE + SIDEPANEL_W
WINDOW_H = GRID_H * TILE + HUD_H

TICK_MS = 700  # simulation tick (ms)

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
HOUSE_TAX = 2           # per occupied house per tick
HOUSE_FOOD_USAGE = 1    # per occupied house per tick
FARM_FOOD = 3           # per farm per tick
FACTORY_MONEY = 10      # per factory per tick
FACTORY_POWER = 2       # demand
HOUSE_POWER = 1         # demand
PLANT_POWER = 20        # capacity

HAPPINESS_DECAY = 0.05  # passive pull to neutral
HAPPINESS_MAX = 100.0
HAPPINESS_MIN = 0.0

# -------------------------
# Data Model
# -------------------------
class B(Enum):
    EMPTY = auto()
    ROAD = auto()
    HOUSE = auto()
    FARM = auto()
    FACTORY = auto()
    PLANT = auto()

BUILD_ORDER = [B.ROAD, B.HOUSE, B.FARM, B.FACTORY, B.PLANT]
TOOLS = BUILD_ORDER + [B.EMPTY]  # EMPTY tool acts as bulldozer

COLOR = {
    B.EMPTY: "#2b2b2b",
    B.ROAD: "#8e8e8e",
    B.HOUSE: "#4caf50",
    B.FARM: "#a3d977",
    B.FACTORY: "#c792ea",
    B.PLANT: "#ffd54f",
}

OUTLINE = "#1a1a1a"

@dataclass
class Tile:
    kind: B = B.EMPTY

# -------------------------
# Game State
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
        self.tool_index: int = 0  # default Road
        self.running: bool = True
        self.tick_count: int = 0

    # ----- derived counts -----
    def counts(self) -> Dict[B, int]:
        c = {b:0 for b in B}
        for y in range(GRID_H):
            for x in range(GRID_W):
                c[self.grid[y][x].kind] += 1
        return c

    def adjacent_to_road(self, x: int, y: int) -> bool:
        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            nx, ny = x+dx, y+dy
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                if self.grid[ny][nx].kind == B.ROAD:
                    return True
        return False

    def compute_power(self):
        # Global simple model
        counts = self.counts()
        capacity = counts[B.PLANT] * PLANT_POWER
        demand = counts[B.HOUSE] * HOUSE_POWER + counts[B.FACTORY] * FACTORY_POWER
        self.power_capacity = capacity
        self.power_demand = demand
        powered = demand <= capacity
        return powered

    def occupied_houses(self, powered: bool) -> int:
        # Houses require adjacency to any road and (if model says) global power
        occ = 0
        for y in range(GRID_H):
            for x in range(GRID_W):
                if self.grid[y][x].kind == B.HOUSE:
                    if self.adjacent_to_road(x, y) and powered:
                        occ += 1
        return occ

    # ----- Simulation tick -----
    def tick(self):
        self.tick_count += 1
        powered = self.compute_power()

        # Production / consumption
        counts = self.counts()
        farms = counts[B.FARM]
        factories = counts[B.FACTORY]

        occ_houses = self.occupied_houses(powered)
        # Population simply capacity of occupied houses * capacity-per-house
        self.population = occ_houses * HOUSE_CAPACITY

        # Food production/consumption
        self.food += farms * FARM_FOOD
        food_needed = occ_houses * HOUSE_FOOD_USAGE
        fed = min(self.food, food_needed)
        self.food -= fed
        starving = food_needed - fed

        # Money
        self.money += factories * FACTORY_MONEY
        self.money += occ_houses * HOUSE_TAX

        # Happiness
        # factories penalize, fed people boost; starvation penalizes more
        dh = 0.0
        dh += occ_houses * 0.2 if powered else -occ_houses * 0.3
        dh += -2.0 * factories
        dh += -1.5 * starving
        # gentle regression toward ~70 base if near neutral economy
        if self.tick_count % 5 == 0:
            if self.happiness < 70:
                dh += 0.5
            elif self.happiness > 70:
                dh -= 0.5

        # Passive decay toward mid
        if self.happiness > 50:
            self.happiness -= HAPPINESS_DECAY
        elif self.happiness < 50:
            self.happiness += HAPPINESS_DECAY

        self.happiness = max(HAPPINESS_MIN, min(HAPPINESS_MAX, self.happiness + dh))

# -------------------------
# UI
# -------------------------
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Simple City Builder (Tkinter)")
        self.game = Game()

        self.canvas = tk.Canvas(root, width=WINDOW_W, height=WINDOW_H, bg="#111")
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.root.bind("<Key>", self.on_key)

        self.draw_all()
        self.schedule_tick()

    # ----- Drawing -----
    def draw_all(self):
        self.canvas.delete("all")
        self.draw_hud()
        self.draw_grid()
        self.draw_sidebar()
        self.draw_cursor_hint()

    def draw_hud(self):
        g = self.game
        # HUD background
        self.canvas.create_rectangle(0, 0, WINDOW_W, HUD_H, fill="#1e1e1e", outline="")
        powered = g.power_demand <= g.power_capacity

        text = (
            f"Money: ${g.money}    Food: {g.food}    Power: {g.power_capacity}/{g.power_demand} {'✓' if powered else '✗'}    "
            f"Population: {g.population}    Happiness: {g.happiness:.1f}"
        )
        self.canvas.create_text(12, 14, anchor="w", fill="#e6e6e6", font=("TkDefaultFont", 11, "bold"), text=text)

        self.canvas.create_text(12, 36, anchor="w", fill="#b0b0b0",
                                text="1)Road  2)House  3)Farm  4)Factory  5)Plant  6)Bulldoze   S)Save  L)Load  R)Reset")

    def draw_grid(self):
        # Map area
        y0 = HUD_H
        for y in range(GRID_H):
            for x in range(GRID_W):
                tx = x * TILE
                ty = y0 + y * TILE
                kind = self.game.grid[y][x].kind
                self.canvas.create_rectangle(
                    tx, ty, tx + TILE, ty + TILE,
                    fill=COLOR[kind], outline=OUTLINE, width=1
                )
                # Minimal icon glyph
                if kind == B.HOUSE:
                    self.canvas.create_rectangle(tx+9, ty+14, tx+23, ty+26, fill="#2e7d32", outline="")
                    self.canvas.create_polygon(tx+8, ty+14, tx+24, ty+14, tx+16, ty+7, fill="#66bb6a", outline="")
                elif kind == B.FARM:
                    self.canvas.create_line(tx+6, ty+22, tx+26, ty+22)
                    self.canvas.create_line(tx+6, ty+18, tx+26, ty+18)
                elif kind == B.FACTORY:
                    self.canvas.create_rectangle(tx+8, ty+18, tx+26, ty+26, fill="#9c6cd3", outline="")
                    self.canvas.create_rectangle(tx+10, ty+12, tx+14, ty+18, fill="#b58ae6", outline="")
                    self.canvas.create_rectangle(tx+16, ty+9, tx+20, ty+18, fill="#b58ae6", outline="")
                elif kind == B.PLANT:
                    self.canvas.create_oval(tx+8, ty+8, tx+24, ty+24, fill="#ffe082", outline="")
                    self.canvas.create_line(tx+16, ty+4, tx+16, ty+28)
                elif kind == B.ROAD:
                    self.canvas.create_line(tx, ty+16, tx+TILE, ty+16, fill="#cfcfcf", width=3)

        # Grid lines lighter overlay (optional aesthetic)
        for x in range(GRID_W+1):
            tx = x*TILE
            self.canvas.create_line(tx, HUD_H, tx, HUD_H + GRID_H*TILE, fill="#222")
        for y in range(GRID_H+1):
            ty = HUD_H + y*TILE
            self.canvas.create_line(0, ty, GRID_W*TILE, ty, fill="#222")

    def draw_sidebar(self):
        # Side panel
        x0 = GRID_W * TILE
        self.canvas.create_rectangle(x0, HUD_H, WINDOW_W, WINDOW_H, fill="#161616", outline="")

        # Tool buttons
        labels = [
            ("Road", B.ROAD),
            ("House", B.HOUSE),
            ("Farm", B.FARM),
            ("Factory", B.FACTORY),
            ("Power Plant", B.PLANT),
            ("Bulldoze", B.EMPTY),
        ]
        for i, (label, kind) in enumerate(labels):
            y = HUD_H + 16 + i*50
            w = SIDEPANEL_W - 24
            x = x0 + 12
            y2 = y + 36
            selected = (TOOLS[self.game.tool_index] == kind)
            self.canvas.create_rectangle(x, y, x+w, y2, fill="#242424", outline="#333", width=2)
            if selected:
                self.canvas.create_rectangle(x+2, y+2, x+w-2, y2-2, outline="#4caf50", width=2)
            self.canvas.create_text(x+12, y+18, anchor="w", fill="#e6e6e6", font=("TkDefaultFont", 11, "bold"), text=label)
            cost = COST[kind.name] if kind != B.EMPTY else 0
            if kind != B.EMPTY:
                self.canvas.create_text(x+w-10, y+18, anchor="e", fill="#b0b0b0", text=f"${cost}")

        # Tips
        tip_y = HUD_H + 16 + len(labels)*50 + 8
        tips = [
            "Left-click to build.",
            "Right-click to bulldoze.",
            "Houses need a road and power.",
            "Power is global but capped.",
            "Factories earn money, hurt happiness.",
            "Farms make food.",
        ]
        self.canvas.create_text(x0+12, tip_y, anchor="nw", fill="#9e9e9e",
                                text="\n".join(tips))

    def draw_cursor_hint(self):
        # Optional: highlight tile under cursor
        pass

    # ----- Input -----
    def screen_to_grid(self, sx: int, sy: int) -> Optional[Tuple[int,int]]:
        if sy < HUD_H: return None
        if sx >= GRID_W * TILE: return None
        gx = sx // TILE
        gy = (sy - HUD_H) // TILE
        if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
            return gx, gy
        return None

    def place(self, x: int, y: int, kind: B):
        g = self.game
        current = g.grid[y][x].kind
        if kind == B.EMPTY:
            # Bulldoze refund 50% (rounded down), except road 30%
            refund = 0
            if current != B.EMPTY:
                base = COST[current.name]
                if current == B.ROAD:
                    refund = base // 3
                else:
                    refund = base // 2
                g.money += refund
                g.grid[y][x].kind = B.EMPTY
            return

        if current != B.EMPTY:
            return  # occupied

        price = COST[kind.name]
        if g.money < price:
            return
        # Adjacency rule for house: must touch a road to place (soft gate)
        if kind == B.HOUSE and not g.adjacent_to_road(x, y):
            # allow placement, but discourage with extra cost? Keep simple: allow but won't be occupied until road exists.
            pass

        g.money -= price
        g.grid[y][x].kind = kind

    def on_left_click(self, ev):
        pos = self.screen_to_grid(ev.x, ev.y)
        if not pos: return
        x, y = pos
        kind = TOOLS[self.game.tool_index]
        self.place(x, y, kind)
        self.draw_all()

    def on_right_click(self, ev):
        pos = self.screen_to_grid(ev.x, ev.y)
        if not pos: return
        x, y = pos
        self.place(x, y, B.EMPTY)
        self.draw_all()

    def on_key(self, ev):
        key = ev.keysym.lower()
        if key in ("1","2","3","4","5","6"):
            self.game.tool_index = int(key)-1
        elif key == "escape":
            self.game.tool_index = len(TOOLS)-1  # bulldoze
        elif key == "s":
            self.save_dialog()
        elif key == "l":
            self.load_dialog()
        elif key == "r":
            if messagebox.askyesno("Reset", "Start a new city?"):
                self.game = Game()
        self.draw_all()

    # ----- Save/Load -----
    def save_dialog(self):
        path = filedialog.asksaveasfilename(
            title="Save City",
            defaultextension=".json",
            filetypes=[("JSON","*.json")],
            initialfile="city_save.json",
        )
        if not path: return
        self.save_to(path)

    def save_to(self, path: str):
        data = {
            "money": self.game.money,
            "food": self.game.food,
            "happiness": self.game.happiness,
            "grid": [[tile.kind.name for tile in row] for row in self.game.grid],
            "tick": self.game.tick_count,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        messagebox.showinfo("Saved", f"Saved to {path}")

    def load_dialog(self):
        path = filedialog.askopenfilename(
            title="Load City",
            filetypes=[("JSON","*.json")],
            initialfile="city_save.json",
        )
        if not path: return
        self.load_from(path)

    def load_from(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
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
            messagebox.showinfo("Loaded", f"Loaded from {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {e}")

    # ----- Simulation loop -----
    def schedule_tick(self):
        if not self.game.running:
            return
        self.root.after(TICK_MS, self.step)

    def step(self):
        self.game.tick()
        self.draw_all()
        self.schedule_tick()

# -------------------------
# Main
# -------------------------
def main():
    root = tk.Tk()
    app = App(root)
    root.resizable(False, False)
    root.mainloop()

if __name__ == "__main__":
    main()
