#!/usr/bin/env python3
"""
One‑file OpenGL City Builder (Python)

Dependencies (install first):
    pip install pygame PyOpenGL PyOpenGL_accelerate

Run:
    python one_file_opengl_city_builder.py

Controls:
    Mouse Left       – Place building on hovered tile (current brush)
    Mouse Right      – Demolish building on hovered tile
    Mouse Wheel      – Zoom in/out
    WASD             – Pan camera on X/Z plane
    Q / E            – Rotate camera left/right
    R / F            – Increase / decrease building height (brush)
    1..5             – Choose building type (brush)
    G                – Toggle grid lines
    P                – Save to city.json
    L                – Load from city.json
    C                – Clear map
    H                – Toggle help overlay
    ESC              – Quit

Notes:
- This is a compact teaching/demo project using legacy OpenGL (immediate mode)
  via PyOpenGL to keep everything in a single file. It should run on most
  desktops without additional shaders or assets. For production, consider
  modern OpenGL with VBO/VAO and a scene graph.
"""
from __future__ import annotations
import math
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

# ------------------------------ Data Model ------------------------------ #
@dataclass
class BuildingBrush:
    btype: int = 1          # building type 1..5 (affects color/style)
    height: int = 2         # floors (1..20)

@dataclass
class Camera:
    x: float = 0.0
    y: float = 25.0
    z: float = 35.0
    yaw: float = 45.0       # degrees around Y
    pitch: float = -35.0    # degrees (downward is negative)
    fov: float = 55.0

# City: mapping (grid_x, grid_z) -> (btype, height)
City = Dict[Tuple[int, int], Tuple[int, int]]

# ------------------------------ Helpers ------------------------------ #

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def set_perspective(width: int, height: int, fov: float):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    aspect = max(1.0, width / float(max(1, height)))
    gluPerspective(fov, aspect, 0.1, 2000.0)
    glMatrixMode(GL_MODELVIEW)


def apply_camera(cam: Camera):
    glLoadIdentity()
    # Rotate then translate (note reversed order for OpenGL fixed pipeline)
    glRotatef(-cam.pitch, 1, 0, 0)
    glRotatef(-cam.yaw, 0, 1, 0)
    glTranslatef(-cam.x, -cam.y, -cam.z)


def screen_ray(x: int, y: int, viewport, modelview, projection):
    """Return world-space ray origin+direction from screen coords (x,y)."""
    # OpenGL screen Y is inverted relative to pygame
    sy = viewport[3] - y
    # Unproject near and far points
    near = gluUnProject(x, sy, 0.0, modelview, projection, viewport)
    far = gluUnProject(x, sy, 1.0, modelview, projection, viewport)
    ox, oy, oz = near
    fx, fy, fz = far
    dx, dy, dz = (fx - ox, fy - oy, fz - oz)
    length = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
    dx, dy, dz = dx/length, dy/length, dz/length
    return (ox, oy, oz), (dx, dy, dz)


def ray_plane_y(ray_o, ray_d, y=0.0) -> Optional[Tuple[float,float,float]]:
    """Intersect ray with horizontal plane at height y. Return point or None."""
    ox, oy, oz = ray_o
    dx, dy, dz = ray_d
    if abs(dy) < 1e-6:
        return None
    t = (y - oy) / dy
    if t < 0:
        return None
    return (ox + dx*t, oy + dy*t, oz + dz*t)


def world_to_grid(wx: float, wz: float, tile: float) -> Tuple[int,int]:
    gx = int(math.floor((wx + 0.5*tile) / tile))
    gz = int(math.floor((wz + 0.5*tile) / tile))
    return gx, gz


def draw_grid(size=100, step=1.0):
    glLineWidth(1.0)
    glBegin(GL_LINES)
    glColor3f(0.15, 0.15, 0.15)
    extent = size * step
    for i in range(-size, size+1):
        glVertex3f(i*step, 0, -extent)
        glVertex3f(i*step, 0,  extent)
        glVertex3f(-extent, 0, i*step)
        glVertex3f( extent, 0, i*step)
    glEnd()


def color_for_type(t: int) -> Tuple[float,float,float]:
    palette = {
        1: (0.6, 0.7, 0.8),   # residential
        2: (0.8, 0.6, 0.4),   # commercial
        3: (0.6, 0.8, 0.6),   # park/green
        4: (0.75, 0.55, 0.75),# civic
        5: (0.7, 0.7, 0.5),   # industrial
    }
    return palette.get(t, (0.7, 0.7, 0.7))


def draw_cube(x, y, z, sx, sy, sz, color):
    r, g, b = color
    glColor3f(r, g, b)
    # Draw a simple lit-look cube using quads with basic vertex normals
    # (still using immediate mode for simplicity)
    glBegin(GL_QUADS)
    # +Y
    glNormal3f(0, 1, 0)
    glVertex3f(x - sx, y + sy, z - sz)
    glVertex3f(x + sx, y + sy, z - sz)
    glVertex3f(x + sx, y + sy, z + sz)
    glVertex3f(x - sx, y + sy, z + sz)
    # -Y
    glNormal3f(0, -1, 0)
    glVertex3f(x - sx, y - sy, z + sz)
    glVertex3f(x + sx, y - sy, z + sz)
    glVertex3f(x + sx, y - sy, z - sz)
    glVertex3f(x - sx, y - sy, z - sz)
    # +X
    glNormal3f(1, 0, 0)
    glVertex3f(x + sx, y - sy, z - sz)
    glVertex3f(x + sx, y + sy, z - sz)
    glVertex3f(x + sx, y + sy, z + sz)
    glVertex3f(x + sx, y - sy, z + sz)
    # -X
    glNormal3f(-1, 0, 0)
    glVertex3f(x - sx, y - sy, z + sz)
    glVertex3f(x - sx, y + sy, z + sz)
    glVertex3f(x - sx, y + sy, z - sz)
    glVertex3f(x - sx, y - sy, z - sz)
    # +Z
    glNormal3f(0, 0, 1)
    glVertex3f(x - sx, y - sy, z + sz)
    glVertex3f(x + sx, y - sy, z + sz)
    glVertex3f(x + sx, y + sy, z + sz)
    glVertex3f(x - sx, y + sy, z + sz)
    # -Z
    glNormal3f(0, 0, -1)
    glVertex3f(x + sx, y - sy, z - sz)
    glVertex3f(x - sx, y - sy, z - sz)
    glVertex3f(x - sx, y + sy, z - sz)
    glVertex3f(x + sx, y + sy, z - sz)
    glEnd()


def draw_building_tile(gx: int, gz: int, btype: int, height: int, tile_size: float):
    # Base footprint centered on tile center
    cx = gx * tile_size
    cz = gz * tile_size
    footprint = 0.45 * tile_size
    floor_h = 1.0
    h = height * floor_h
    color = color_for_type(btype)
    # Slight style variation by type
    if btype == 3:  # park/green – low plinth with a tree-like pillar
        draw_cube(cx, h*0.05, cz, footprint, h*0.05, footprint, (0.2, 0.4, 0.2))
        draw_cube(cx, h*0.6, cz, footprint*0.12, h*0.6, footprint*0.12, (0.25,0.35,0.25))
        draw_cube(cx, h*0.95, cz, footprint*0.45, h*0.15, footprint*0.45, (0.25,0.5,0.25))
    else:
        # Tower
        draw_cube(cx, h*0.5, cz, footprint*0.9, h*0.5, footprint*0.9, (color[0]*0.9, color[1]*0.9, color[2]*0.9))
        # Roof/cap
        draw_cube(cx, h - 0.1, cz, footprint, 0.1, footprint, color)
        # Entrance plinth
        draw_cube(cx, 0.05, cz, footprint, 0.05, footprint, (color[0]*0.8, color[1]*0.8, color[2]*0.8))


def draw_hover_tile(gx: int, gz: int, tile_size: float):
    cx = gx * tile_size
    cz = gz * tile_size
    s = 0.5 * tile_size
    glDisable(GL_LIGHTING)
    glLineWidth(2.0)
    glColor3f(1.0, 0.8, 0.2)
    glBegin(GL_LINE_LOOP)
    glVertex3f(cx - s, 0.01, cz - s)
    glVertex3f(cx + s, 0.01, cz - s)
    glVertex3f(cx + s, 0.01, cz + s)
    glVertex3f(cx - s, 0.01, cz + s)
    glEnd()
    glEnable(GL_LIGHTING)


def draw_ground(size_tiles=100, tile_size=1.0):
    extent = size_tiles * tile_size
    glDisable(GL_LIGHTING)
    glColor3f(0.12, 0.12, 0.12)
    glBegin(GL_QUADS)
    glVertex3f(-extent, 0, -extent)
    glVertex3f( extent, 0, -extent)
    glVertex3f( extent, 0,  extent)
    glVertex3f(-extent, 0,  extent)
    glEnd()
    glEnable(GL_LIGHTING)


def draw_text_2d(surface, text, x, y, color=(255,255,255), size=16):
    font = pygame.font.SysFont("consolas,menlo,monaco,dejavu sans mono,arial", size)
    s = font.render(text, True, color)
    surface.blit(s, (x, y))

# ------------------------------ Main App ------------------------------ #
class CityBuilderApp:
    def __init__(self, width=1280, height=720):
        pygame.init()
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("One‑file OpenGL City Builder")
        self.clock = pygame.time.Clock()
        self.width = width
        self.height = height

        # OpenGL basic setup
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (0.3, 1.0, 0.2, 0.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.9, 0.9, 0.9, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.12, 0.12, 0.12, 1.0))
        glShadeModel(GL_SMOOTH)

        set_perspective(width, height, 55.0)
        self.cam = Camera()
        self.brush = BuildingBrush()
        self.city: City = {}
        self.tile_size = 2.0
        self.show_grid = True
        self.show_help = True

        # Interaction state
        self.mouse_last = None

    # --------------- Persistence --------------- #
    def save(self, path="city.json"):
        data = {
            "tile_size": self.tile_size,
            "city": {f"{x},{z}": [bt, h] for (x, z), (bt, h) in self.city.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(self.city)} tiles to {path}")

    def load(self, path="city.json"):
        if not os.path.exists(path):
            print("No save file to load.")
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.tile_size = float(data.get("tile_size", 2.0))
        self.city.clear()
        for k, v in data.get("city", {}).items():
            xs, zs = k.split(",")
            self.city[(int(xs), int(zs))] = (int(v[0]), int(v[1]))
        print(f"Loaded {len(self.city)} tiles from {path}")

    # --------------- Input Handling --------------- #
    def handle_input(self, dt):
        # Keyboard
        keys = pygame.key.get_pressed()
        speed = 15.0 * dt
        rot_speed = 70.0 * dt

        # Move relative to yaw (on XZ plane)
        yaw_rad = math.radians(self.cam.yaw)
        forward = (math.sin(yaw_rad), 0, math.cos(yaw_rad))
        right = (math.cos(yaw_rad), 0, -math.sin(yaw_rad))

        if keys[K_w]:
            self.cam.x += forward[0] * speed
            self.cam.z += forward[2] * speed
        if keys[K_s]:
            self.cam.x -= forward[0] * speed
            self.cam.z -= forward[2] * speed
        if keys[K_a]:
            self.cam.x -= right[0] * speed
            self.cam.z -= right[2] * speed
        if keys[K_d]:
            self.cam.x += right[0] * speed
            self.cam.z += right[2] * speed
        if keys[K_q]:
            self.cam.yaw += rot_speed
        if keys[K_e]:
            self.cam.yaw -= rot_speed

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit(0)
            elif event.type == VIDEORESIZE:
                self.width, self.height = event.w, event.h
                pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
                glViewport(0, 0, self.width, self.height)
                set_perspective(self.width, self.height, self.cam.fov)
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)
                elif event.key == K_g:
                    self.show_grid = not self.show_grid
                elif event.key == K_h:
                    self.show_help = not self.show_help
                elif event.key == K_p:
                    self.save()
                elif event.key == K_l:
                    self.load()
                elif event.key == K_c:
                    self.city.clear()
                elif event.key in (K_1, K_2, K_3, K_4, K_5):
                    self.brush.btype = int(event.key - K_0)
                elif event.key == K_r:
                    self.brush.height = clamp(self.brush.height + 1, 1, 40)
                elif event.key == K_f:
                    self.brush.height = clamp(self.brush.height - 1, 1, 40)
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 4:  # wheel up
                    self.cam.fov = clamp(self.cam.fov - 2.5, 25.0, 90.0)
                    set_perspective(self.width, self.height, self.cam.fov)
                elif event.button == 5:  # wheel down
                    self.cam.fov = clamp(self.cam.fov + 2.5, 25.0, 90.0)
                    set_perspective(self.width, self.height, self.cam.fov)
                elif event.button in (1, 3):
                    self.on_click(place=(event.button == 1))

    def hovered_tile(self) -> Optional[Tuple[int,int]]:
        mx, my = pygame.mouse.get_pos()
        # Extract matrices for unprojection
        model = glGetDoublev(GL_MODELVIEW_MATRIX)
        proj = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)
        ray_o, ray_d = screen_ray(mx, my, viewport, model, proj)
        hit = ray_plane_y(ray_o, ray_d, y=0.0)
        if not hit:
            return None
        hx, hy, hz = hit
        gx, gz = world_to_grid(hx, hz, self.tile_size)
        return (gx, gz)

    def on_click(self, place: bool):
        cell = self.hovered_tile()
        if not cell:
            return
        if place:
            self.city[cell] = (self.brush.btype, self.brush.height)
        else:
            if cell in self.city:
                del self.city[cell]

    # --------------- Rendering --------------- #
    def render_scene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(0, 0, self.width, self.height)
        apply_camera(self.cam)

        draw_ground(200, self.tile_size)
        if self.show_grid:
            draw_grid(200, self.tile_size)

        # Draw buildings
        glEnable(GL_LIGHTING)
        for (gx, gz), (bt, h) in self.city.items():
            draw_building_tile(gx, gz, bt, h, self.tile_size)

        # Hover outline
        cell = self.hovered_tile()
        if cell is not None:
            draw_hover_tile(cell[0], cell[1], self.tile_size)

    def render_ui(self):
        """Swap the OpenGL buffers. Avoid pygame blits/update on OPENGL displays."""
        pygame.display.flip()

    # --------------- Main Loop --------------- #
    def run(self):
        while True:
            dt = self.clock.tick(60) / 1000.0
            self.handle_input(dt)
            # Draw 3D scene
            self.render_scene()
            # Draw 2D overlay
            self.render_ui()


if __name__ == "__main__":
    try:
        app = CityBuilderApp(1280, 720)
        app.run()
    except Exception as e:
        print("Fatal error:", e)
        raise
