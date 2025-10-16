# city_builder_plus.py
# OpenGL 3.3 core + pygame city-builder skeleton with:
# - Grid preview wireframe cube (hover)
# - Roads as a separate instanced mesh (thin height); adjacency bonuses
# - Save/Load to JSON/NPZ
# - HUD overlay via pygame font -> GL texture
# - Citizens moving on roads using A* pathfinding (simple commute loop)

import sys, os, math, time, json, ctypes, random
import numpy as np
import pygame
from pygame.locals import (
    DOUBLEBUF, OPENGL, RESIZABLE, VIDEORESIZE, WINDOWSIZECHANGED,
    QUIT, KEYDOWN, K_ESCAPE, K_m, K_w, K_a, K_s, K_d, K_SPACE, K_c,
    MOUSEBUTTONDOWN
)

from OpenGL.GL import *
# (We use explicit names via GL imports above for brevity; still 3.3-safe.)

# ---------------- Shaders ----------------
VERT_SRC = """#version 330 core
layout (location=0) in vec3 aPos;
layout (location=1) in vec3 aColor;

layout (location=2) in vec3 iOffset; // per-instance world offset (tile center)
layout (location=3) in vec3 iColor;  // per-instance color

out vec3 vColor;

layout (std140) uniform Matrices {
    mat4 uProjection;
    mat4 uView;
};

uniform mat4 uModel;

void main(){
    vec3 worldPos = aPos + iOffset;
    gl_Position = uProjection * uView * uModel * vec4(worldPos, 1.0);
    vColor = iColor * aColor;
}
"""

FRAG_SRC = """#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main(){
    FragColor = vec4(vColor, 1.0);
}
"""

# Screen-space shader for HUD quad (no depth)
HUD_VERT = """#version 330 core
layout (location=0) in vec2 aPos;
layout (location=1) in vec2 aUV;
out vec2 vUV;
void main(){
    vUV = aUV;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""
HUD_FRAG = """#version 330 core
in vec2 vUV;
uniform sampler2D uTex;
out vec4 FragColor;
void main(){
    vec4 c = texture(uTex, vUV);
    // Pre-multiplied alpha not required; pygame gives straight alpha.
    FragColor = c;
}
"""

# ---------------- Math utils ----------------
def normalize(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def perspective(fovy_deg, aspect, znear, zfar):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    M = np.zeros((4,4), dtype=np.float32)
    M[0,0] = f / max(aspect, 1e-6)
    M[1,1] = f
    M[2,2] = (zfar + znear) / (znear - zfar)
    M[2,3] = (2 * zfar * znear) / (znear - zfar)
    M[3,2] = -1.0
    return M

def look_at(eye, center, up):
    f = normalize(center - eye); s = normalize(np.cross(f, up)); u = np.cross(s, f)
    M = np.eye(4, dtype=np.float32)
    M[0,0:3] = s; M[1,0:3] = u; M[2,0:3] = -f
    T = np.eye(4, dtype=np.float32)
    T[0,3], T[1,3], T[2,3] = -eye
    return M @ T

def translation(t):
    M = np.eye(4, dtype=np.float32)
    M[0,3], M[1,3], M[2,3] = t
    return M

def scale(sx, sy, sz):
    M = np.eye(4, dtype=np.float32)
    M[0,0]=sx; M[1,1]=sy; M[2,2]=sz
    return M

def invert(m):
    return np.linalg.inv(m)

# ---------------- GL helpers ----------------
def compile_shader(src, stype):
    sid = glCreateShader(stype); glShaderSource(sid, src); glCompileShader(sid)
    ok = glGetShaderiv(sid, GL_COMPILE_STATUS)
    if not ok:
        raise RuntimeError(glGetShaderInfoLog(sid).decode())
    return sid

def link_program(vs, fs):
    pid = glCreateProgram()
    glAttachShader(pid, vs); glAttachShader(pid, fs); glLinkProgram(pid)
    ok = glGetProgramiv(pid, GL_LINK_STATUS)
    if not ok:
        raise RuntimeError(glGetProgramInfoLog(pid).decode())
    glDeleteShader(vs); glDeleteShader(fs)
    return pid

# -------------- Geometry (cube + ground + hud quad) --------------
def make_cube():
    # Unit cube centered at origin; base color = 1 (instance color drives)
    verts = np.array([
        # pos               # col
        -0.5,-0.5,-0.5,     1,1,1,
         0.5,-0.5,-0.5,     1,1,1,
         0.5, 0.5,-0.5,     1,1,1,
        -0.5, 0.5,-0.5,     1,1,1,
        -0.5,-0.5, 0.5,     1,1,1,
         0.5,-0.5, 0.5,     1,1,1,
         0.5, 0.5, 0.5,     1,1,1,
        -0.5, 0.5, 0.5,     1,1,1,
    ], dtype=np.float32)
    idx = np.array([
        0,1,2, 2,3,0,  4,5,6, 6,7,4,
        0,4,7, 7,3,0,  1,5,6, 6,2,1,
        3,2,6, 6,7,3,  0,1,5, 5,4,0,
    ], dtype=np.uint32)

    vao = glGenVertexArrays(1); glBindVertexArray(vao)
    vbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
    ebo = glGenBuffers(1); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)
    stride = 6*4
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,stride,ctypes.c_void_p(0))
    glEnableVertexAttribArray(1); glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,stride,ctypes.c_void_p(12))
    glBindVertexArray(0)
    return vao, idx.size

def make_ground():
    # Flat 2x2 quad at origin (y=0)
    verts = np.array([
        -1,0,-1,  0.8,0.8,0.8,
         1,0,-1,  0.8,0.8,0.8,
         1,0, 1,  0.8,0.8,0.8,
        -1,0, 1,  0.8,0.8,0.8,
    ], dtype=np.float32)
    idx = np.array([0,1,2, 2,3,0], dtype=np.uint32)
    vao = glGenVertexArrays(1); glBindVertexArray(vao)
    vbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
    ebo = glGenBuffers(1); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)
    stride = 6*4
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,stride,ctypes.c_void_p(0))
    glEnableVertexAttribArray(1); glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,stride,ctypes.c_void_p(12))
    glBindVertexArray(0)
    return vao, idx.size

def make_hud_quad():
    # Fullscreen-aligned quad in NDC; we draw HUD in upper-left via UV crop
    verts = np.array([
        #   x,   y,   u,   v
        -1.0,  1.0,  0.0, 0.0,
        -1.0,  0.6,  0.0, 1.0,
        +1.0,  0.6,  1.0, 1.0,
        +1.0,  1.0,  1.0, 0.0,
    ], dtype=np.float32)
    idx = np.array([0,1,2, 2,3,0], dtype=np.uint32)
    vao = glGenVertexArrays(1); glBindVertexArray(vao)
    vbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
    ebo = glGenBuffers(1); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)
    stride = 4*4
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,stride,ctypes.c_void_p(0))
    glEnableVertexAttribArray(1); glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,stride,ctypes.c_void_p(8))
    glBindVertexArray(0)
    return vao, idx.size

# -------------- UBO (proj/view) --------------
def create_matrices_ubo():
    ubo = glGenBuffers(1)
    glBindBuffer(GL_UNIFORM_BUFFER, ubo)
    glBufferData(GL_UNIFORM_BUFFER, 2*64, None, GL_DYNAMIC_DRAW)
    glBindBufferRange(GL_UNIFORM_BUFFER, 0, ubo, 0, 2*64)
    glBindBuffer(GL_UNIFORM_BUFFER, 0)
    return ubo

def update_matrices_ubo(ubo, proj, view):
    glBindBuffer(GL_UNIFORM_BUFFER, ubo)
    glBufferSubData(GL_UNIFORM_BUFFER, 0, 64,  proj.T.astype(np.float32))
    glBufferSubData(GL_UNIFORM_BUFFER, 64,64,  view.T.astype(np.float32))
    glBindBuffer(GL_UNIFORM_BUFFER, 0)

# -------------- Camera --------------
class Camera:
    def __init__(self, pos=np.array([0.0, 6.0, 10.0], dtype=np.float32), yaw=-90.0, pitch=-30.0):
        self.pos = pos.astype(np.float32); self.yaw=yaw; self.pitch=pitch
        self.front=np.array([0,0,-1],np.float32); self.right=np.array([1,0,0],np.float32); self.up=np.array([0,1,0],np.float32)
        self._update_vectors()
    def _update_vectors(self):
        cx, sx = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        cy, sy = math.cos(math.radians(self.pitch)), math.sin(math.radians(self.pitch))
        self.front = normalize(np.array([cx*cy, sy, sx*cy], np.float32))
        self.right = normalize(np.cross(self.front, np.array([0,1,0], np.float32)))
        self.up    = normalize(np.cross(self.right, self.front))
    def process_mouse(self, dx, dy, sens=0.12):
        self.yaw += dx * sens; self.pitch = max(-89.0, min(89.0, self.pitch - dy*sens)); self._update_vectors()
    def move(self, d, speed, dt):
        self.pos += d * (speed*dt)
    def view(self):
        return look_at(self.pos, self.pos + self.front, self.up)

# -------------- Instancing helpers --------------
class InstanceBuffer:
    def __init__(self, vao, loc_offset=2, loc_color=3):
        self.vao = vao; self.vbo = glGenBuffers(1); self.count = 0
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        stride = 6 * 4
        glEnableVertexAttribArray(loc_offset); glVertexAttribPointer(loc_offset,3,GL_FLOAT,GL_FALSE,stride,ctypes.c_void_p(0)); glVertexAttribDivisor(loc_offset,1)
        glEnableVertexAttribArray(loc_color);  glVertexAttribPointer(loc_color, 3,GL_FLOAT,GL_FALSE,stride,ctypes.c_void_p(12)); glVertexAttribDivisor(loc_color,1)
        glBindVertexArray(0)
    def upload(self, oc: np.ndarray):
        self.count = 0 if oc is None else oc.shape[0]
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        if self.count == 0:
            glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)
        else:
            glBufferData(GL_ARRAY_BUFFER, oc.nbytes, oc.astype(np.float32), GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

# -------------- Picking --------------
def ndc_from_mouse(mx, my, w, h):
    x = (2.0 * mx) / max(w,1) - 1.0
    y = 1.0 - (2.0 * my) / max(h,1)
    return x, y

def make_ray(mx, my, w, h, proj, view):
    inv = invert(proj @ view)
    x, y = ndc_from_mouse(mx, my, w, h)
    near = np.array([x, y, -1.0, 1.0], np.float32)
    far  = np.array([x, y,  1.0, 1.0], np.float32)
    p0 = inv @ near; p0 = p0[:3] / max(p0[3], 1e-6)
    p1 = inv @ far;  p1 = p1[:3] / max(p1[3],  1e-6)
    dirv = normalize(p1 - p0); return p0, dirv

def ray_plane_y0_intersect(o, d):
    if abs(d[1]) < 1e-6: return None
    t = -o[1] / d[1]
    if t < 0: return None
    return o + t*d

# -------------- Game data --------------
# Building IDs
HOUSE=1; FARM=2; FACTORY=3; POWERPLANT=4; ROAD=9

# Power model:
# - Roads transmit power.
# - Power plants generate capacity (power_gen).
# - Consumers (e.g., houses/factories) use power (power_use).
# - Farms don't require power in this simple model.
BUILDINGS = {
    HOUSE:   {"name":"House",   "color":(0.95,0.55,0.55), "cost":10,  "income":1,  "power_use":1},
    FARM:    {"name":"Farm",    "color":(0.55,0.95,0.55), "cost":15,  "income":2,  "power_use":0},
    FACTORY: {"name":"Factory", "color":(0.55,0.65,0.95), "cost":30,  "income":5,  "power_use":2},
    POWERPLANT: {"name":"Power plant", "color":(0.55,0.95,0.95), "cost":40,  "income":50, "power_gen":20},
}
ROAD_COLOR = (0.25,0.25,0.25)
ADJ_BONUS_PER_ROAD_FOR_HOUSE = 1  # N/E/S/W only

def neighbors4(i,k,N):
    for di,dk in ((1,0),(-1,0),(0,1),(0,-1)):
        ni, nk = i+di, k+dk
        if 0 <= ni < N and 0 <= nk < N:
            yield ni, nk

class City:
    def __init__(self, size):
        self.N = size
        self.grid = np.zeros((self.N, self.N), dtype=np.int8)
        self.money = 5000
        self.selected = HOUSE
        self.last_income = time.perf_counter()
        self._roads_dirty = True
        self._bldg_dirty = True
        # Power state
        self._power_dirty = True
        self.powered = np.zeros((self.N, self.N), dtype=bool)  # which buildings are powered (consumers only)
        self.power_supply = 0   # total capacity from plants
        self.power_demand = 0   # total requested by connected consumers
        self.power_used = 0     # how much capacity allocated
    def can_place(self, i, k, kind):
        if not (0 <= i < self.N and 0 <= k < self.N): return False
        if kind == ROAD:
            return self.grid[i,k] in (0, ROAD)  # allow upgrading empty to road; road->road no-op
        else:
            return self.grid[i,k] == 0  # cannot build on road for simplicity
    def place(self, i, k):
        kind = self.selected
        cost = BUILDINGS[kind]["cost"] if kind in BUILDINGS else 2  # road cheap
        if self.money < cost or not self.can_place(i,k,kind): return False
        self.money -= cost
        self.grid[i,k] = kind
        if kind == ROAD: self._roads_dirty = True
        else: self._bldg_dirty = True
        self._power_dirty = True
        return True
    def remove(self, i, k):
        if not (0 <= i < self.N and 0 <= k < self.N): return False
        cur = int(self.grid[i,k])
        if cur == 0: return False
        refund = 1 if cur == ROAD else int(BUILDINGS[cur]["cost"]*0.5)
        self.money += refund
        self.grid[i,k] = 0
        if cur == ROAD: self._roads_dirty = True
        else: self._bldg_dirty = True
        self._power_dirty = True
        return True
    def compute_power(self):
        # Recompute power graph and allocation if dirty
        if not (self._roads_dirty or self._bldg_dirty or self._power_dirty):
            return
        N = self.N
        self.powered[:,:] = False
        # Capacity from plants
        plants = list(zip(*np.where(self.grid == POWERPLANT)))
        supply = 0
        for (pi, pk) in plants:
            props = BUILDINGS.get(POWERPLANT, {})
            supply += int(props.get("power_gen", 0))
        self.power_supply = supply

        # Find energized road network via multi-source BFS seeded by roads adjacent to plants
        road_mask = (self.grid == ROAD)
        energized = np.zeros_like(road_mask, dtype=bool)
        from collections import deque
        q = deque()
        for (pi, pk) in plants:
            for ni, nk in neighbors4(pi, pk, N):
                if road_mask[ni, nk] and not energized[ni, nk]:
                    energized[ni, nk] = True
                    q.append((ni, nk))
        while q:
            ci, ck = q.popleft()
            for ni, nk in neighbors4(ci, ck, N):
                if road_mask[ni, nk] and not energized[ni, nk]:
                    energized[ni, nk] = True
                    q.append((ni, nk))

        # Collect consumers connected to energized roads or directly adjacent to plants
        consumers = []  # list of (i,k,power_use)
        total_demand = 0
        for i in range(N):
            for k in range(N):
                t = int(self.grid[i,k])
                if t in (0, ROAD):
                    continue
                p_use = int(BUILDINGS[t].get("power_use", 0))
                if p_use <= 0:
                    continue
                # Connected if adjacent to energized road or directly adjacent to plant
                connected = False
                for ni, nk in neighbors4(i, k, N):
                    if energized[ni, nk] or self.grid[ni, nk] == POWERPLANT:
                        connected = True
                        break
                if connected:
                    consumers.append((i, k, p_use))
                    total_demand += p_use
        self.power_demand = total_demand

        # Allocate capacity in a stable order (top-left to bottom-right)
        consumers.sort(key=lambda x: (x[0], x[1]))
        remaining = self.power_supply
        used = 0
        for i, k, p_use in consumers:
            if remaining >= p_use:
                self.powered[i, k] = True
                remaining -= p_use
                used += p_use
            else:
                self.powered[i, k] = False
        self.power_used = used
        self._power_dirty = False
    def income_tick(self):
        now = time.perf_counter()
        if now - self.last_income < 1.0: return 0
        self.last_income = now
        # Ensure power state is up to date for income gating
        self.compute_power()
        income = 0
        # base income, gated by power for consumers that require it
        for i in range(self.N):
            for k in range(self.N):
                t = int(self.grid[i, k])
                if t in (0, ROAD):
                    continue
                props = BUILDINGS[t]
                p_use = int(props.get("power_use", 0))
                requires_power = p_use > 0
                if requires_power and not self.powered[i, k]:
                    continue
                income += int(props.get("income", 0))
        # road adjacency bonus for houses (only if the house is powered)
        bonus = 0
        for i in range(self.N):
            for k in range(self.N):
                if self.grid[i, k] == HOUSE and self.powered[i, k]:
                    for ni, nk in neighbors4(i, k, self.N):
                        if self.grid[ni, nk] == ROAD:
                            bonus += ADJ_BONUS_PER_ROAD_FOR_HOUSE
        income += bonus
        self.money += income
        return income
    # Instance data builders
    def build_instanced_arrays(self):
        # Make sure power info is current so we can visualize powered/unpowered buildings
        self.compute_power()
        bldg_offsets, bldg_colors = [], []
        road_offsets, road_colors = [], []
        for i in range(self.N):
            for k in range(self.N):
                t = int(self.grid[i,k])
                if t == 0: continue
                if t == ROAD:
                    road_offsets.append([i+0.5, 0.05, k+0.5])  # slightly above ground
                    road_colors.append(list(ROAD_COLOR))
                else:
                    base_col = BUILDINGS[t]["color"]
                    p_use = int(BUILDINGS[t].get("power_use", 0))
                    if p_use > 0 and not self.powered[i, k]:
                        # Dim unpowered consumers
                        bcol = tuple([c * 0.3 for c in base_col])
                    else:
                        bcol = base_col
                    bldg_offsets.append([i+0.5, 0.5, k+0.5])  # cube sits on plane
                    bldg_colors.append(list(bcol))
        def pack(offs, cols):
            if not offs: return np.zeros((0,6), np.float32)
            return np.concatenate([np.asarray(offs,np.float32), np.asarray(cols,np.float32)], axis=1)
        return pack(bldg_offsets,bldg_colors), pack(road_offsets,road_colors)

    # Save/load
    def save_json(self, path):
        data = {"N": int(self.N), "money": int(self.money), "selected": int(self.selected), "grid": self.grid.tolist()}
        with open(path, "w") as f: json.dump(data, f)
    def load_json(self, path):
        with open(path, "r") as f: data = json.load(f)
        if data["N"] != self.N:
            raise ValueError("Grid size mismatch.")
        self.grid = np.array(data["grid"], dtype=np.int8)
        self.money = int(data.get("money", 50))
        self.selected = int(data.get("selected", HOUSE))
        self._roads_dirty = self._bldg_dirty = True
        self._power_dirty = True
    def save_npz(self, path):
        np.savez(path, N=self.N, money=self.money, selected=self.selected, grid=self.grid)
    def load_npz(self, path):
        z = np.load(path, allow_pickle=False)
        if int(z["N"]) != self.N: raise ValueError("Grid size mismatch.")
        self.grid = z["grid"].astype(np.int8)
        self.money = int(z["money"]); self.selected = int(z["selected"])
        self._roads_dirty = self._bldg_dirty = True
        self._power_dirty = True

# -------------- Citizens & A* --------------
class Citizen:
    def __init__(self, start_tile, path_tiles):
        self.pos = np.array([start_tile[0]+0.5, 0.35, start_tile[1]+0.5], np.float32)
        self.path = list(path_tiles)  # list of (i,k)
        self.seg_idx = 0
        self.speed = 2.0  # tiles per second
        self.color = np.array([1.0, 0.9, 0.2], np.float32)

    def update(self, dt):
        if self.seg_idx >= len(self.path): return
        target = np.array([self.path[self.seg_idx][0]+0.5, 0.35, self.path[self.seg_idx][1]+0.5], np.float32)
        delta = target - self.pos
        dist = np.linalg.norm(delta)
        step = self.speed * dt
        if dist <= step:
            self.pos = target
            self.seg_idx += 1
        else:
            self.pos += (delta / max(dist,1e-6)) * step

def a_star(grid, start, goal):
    # 4-neighbor A*
    N = grid.shape[0]
    def passable(i,k):
        return 0 <= i < N and 0 <= k < N and grid[i,k] == ROAD
    def h(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    if not passable(*start) or not passable(*goal): return []
    import heapq
    openq = []
    heapq.heappush(openq, (0, start))
    came, g = {start: None}, {start: 0}
    while openq:
        _, cur = heapq.heappop(openq)
        if cur == goal: break
        for n in neighbors4(*cur, N):
            if not passable(*n): continue
            ng = g[cur] + 1
            if ng < g.get(n, 1e9):
                g[n] = ng
                f = ng + h(n, goal)
                heapq.heappush(openq, (f, n))
                came[n] = cur
    if goal not in came: return []
    path = []
    cur = goal
    while cur:
        path.append(cur)
        cur = came[cur]
    path.reverse()
    return path

class Population:
    def __init__(self, city):
        self.city = city
        self.citizens = []  # list of Citizen
        self._cooldown = 0.0
    def update(self, dt):
        # spawn at most 1 per second if possible
        self._cooldown -= dt
        if self._cooldown <= 0.0:
            self._cooldown = 1.0
            # try to create a commuter from a random house to a random factory via roads
            houses = list(zip(*np.where(self.city.grid == HOUSE)))
            facts  = list(zip(*np.where(self.city.grid == FACTORY)))
            if houses and facts:
                h = random.choice(houses); f = random.choice(facts)
                # find nearest road to each (or require they sit on road neighbors)
                def nearest_road(tile):
                    i,k = tile
                    best=None; bestd=1e9
                    for ri,rk in zip(*np.where(self.city.grid == ROAD)):
                        d = abs(ri-i)+abs(rk-k)
                        if d < bestd: bestd, best=(d,(ri,rk))
                    if best:
                        return best
                    else:
                        return None
                
                rs = nearest_road(h); rg = nearest_road(f)
                if rs and rg:
                    path = a_star(self.city.grid, rs, rg)
                    if path:
                        self.citizens.append(Citizen(rs, path))
        # move all
        for c in self.citizens:
            c.update(dt)
        # cap population
        if len(self.citizens) > 200:
            self.citizens = self.citizens[-200:]

    def instanced_data(self):
        if not self.citizens:
            return np.zeros((0,6), np.float32)
        offs = [[c.pos[0], c.pos[1], c.pos[2]] for c in self.citizens]
        cols = [c.color.tolist() for c in self.citizens]
        return np.concatenate([np.asarray(offs,np.float32), np.asarray(cols,np.float32)], axis=1)

# -------------- HUD (pygame font -> GL texture) --------------
class HUD:
    def __init__(self, w, h):
        pygame.font.init()
        # Prefer a robust monospace fallback list
        self.font = pygame.font.SysFont(["Consolas", "Menlo", "DejaVu Sans Mono", "Monospace"], 18)
        self.tex = glGenTextures(1)
        self.w = w; self.h = h  # HUD texture size
        glBindTexture(GL_TEXTURE_2D, self.tex)
        # Ensure tight byte packing from pygame surfaces
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.w, self.h, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)
        self.last_text = ""
        self.hud_vao, self.hud_icount = make_hud_quad()
        hud_vs = compile_shader(HUD_VERT, GL_VERTEX_SHADER)
        hud_fs = compile_shader(HUD_FRAG, GL_FRAGMENT_SHADER)
        self.hud_prog = link_program(hud_vs, hud_fs)
        
    def update_text(self, lines):
        text = "\n".join(lines)
        if text == self.last_text:
            return
        self.last_text = text
        # Render via pygame
        surf = pygame.Surface((self.w, self.h), flags=pygame.SRCALPHA)
        surf.fill((0,0,0,120))
        y = 6
        for line in lines:
            s = self.font.render(line, True, (255,255,255))
            surf.blit(s, (8, y)); y += s.get_height()+2
        # Do not flip vertically; our HUD quad UVs use top-left origin
        data = pygame.image.tostring(surf, "RGBA", False)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.w, self.h, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glBindTexture(GL_TEXTURE_2D, 0)

    def draw(self, lines):
        # HUD draw
        self.update_text(lines)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glUseProgram(self.hud_prog)
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, self.tex)
        glBindVertexArray(self.hud_vao)
        glDrawElements(GL_TRIANGLES, self.hud_icount, GL_UNSIGNED_INT, ctypes.c_void_p(0))
        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)

# -------------- Pygame / GL init --------------
def init_pygame(width=1280, height=720, title="City Builder Plus (OpenGL 3.3)"):
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
    pygame.display.gl_set_attribute(pygame.GL_STENCIL_SIZE, 8)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
    flags = OPENGL | DOUBLEBUF | RESIZABLE
    screen = pygame.display.set_mode((width, height), flags, vsync=1)
    pygame.display.set_caption(title)
    return screen

# -------------- Main --------------
def main():
    GRID = 32
    city = City(GRID)
    pop = Population(city)

    screen = init_pygame()
    w, h = screen.get_size()
    glViewport(0,0,w,h)

    print("GL_VENDOR:  ", glGetString(GL_VENDOR).decode())
    print("GL_RENDERER:", glGetString(GL_RENDERER).decode())
    print("GL_VERSION: ", glGetString(GL_VERSION).decode())
    print("GLSL:       ", glGetString(GL_SHADING_LANGUAGE_VERSION).decode())

    glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LEQUAL)

    # Programs
    p_vs = compile_shader(VERT_SRC, GL_VERTEX_SHADER)
    p_fs = compile_shader(FRAG_SRC, GL_FRAGMENT_SHADER)
    prog = link_program(p_vs, p_fs)

    

    # UBO binding
    ubo = create_matrices_ubo()
    blk = glGetUniformBlockIndex(prog, "Matrices")
    glUniformBlockBinding(prog, blk, 0)

    # Meshes & instance buffers
    # Use separate VAOs per instance category to avoid attrib pointer overwrites
    cube_vao_buildings, cube_icount = make_cube()
    cube_vao_roads, _ = make_cube()
    cube_vao_people, _ = make_cube()
    ground_vao, ground_icount = make_ground()
    
    inst_buildings = InstanceBuffer(cube_vao_buildings)
    inst_roads     = InstanceBuffer(cube_vao_roads)
    inst_people    = InstanceBuffer(cube_vao_people)  # tiny cubes

    # Uniform locs
    u_model = glGetUniformLocation(prog, "uModel")

    # Camera/proj
    cam = Camera()
    fovy, znear, zfar = 60.0, 0.1, 500.0
    proj = perspective(fovy, w/max(h,1), znear, zfar)
    update_matrices_ubo(ubo, proj, cam.view())

    # Ground transform: scale to grid and center
    ground_model = translation([GRID/2.0, 0.0, GRID/2.0]) @ scale(GRID/2.0, 1.0, GRID/2.0)

    # HUD
    hud = HUD(w,h/4) #Resonable size (Check UV map) in make_hud_quad():

    # State
    mouse_captured = False
    pygame.mouse.set_visible(True)
    pygame.event.set_grab(False)

    def refresh_instances(force=False):
        if city._bldg_dirty or force:
            oc_bldg, _ = city.build_instanced_arrays()
            inst_buildings.upload(oc_bldg)
            city._bldg_dirty = False
        if city._roads_dirty or force:
            _, oc_road = city.build_instanced_arrays()
            inst_roads.upload(oc_road)
            city._roads_dirty = False

    refresh_instances(force=True)

    clock = pygame.time.Clock()
    running = True
    hover_tile = None

    def hud_lines():
        sel = city.selected
        name = BUILDINGS[sel]["name"] if sel in BUILDINGS else "Road"
        counts = {HOUSE:int(np.sum(city.grid==HOUSE)),
                  FARM:int(np.sum(city.grid==FARM)),
                  FACTORY:int(np.sum(city.grid==FACTORY)),
                  POWERPLANT:int(np.sum(city.grid==POWERPLANT)),
                  ROAD:int(np.sum(city.grid==ROAD))}
        # Power summary
        need_power = 0
        powered = 0
        for i in range(GRID):
            for k in range(GRID):
                t = int(city.grid[i, k])
                if t in (0, ROAD):
                    continue
                if int(BUILDINGS[t].get("power_use", 0)) > 0:
                    need_power += 1
                    if city.powered[i, k]:
                        powered += 1
        return [
            f"Money: {city.money}",
            f"Selected [{sel}]: {name}",
            f"Counts: House={counts[HOUSE]} Farm={counts[FARM]} Factory={counts[FACTORY]} Power plant={counts[POWERPLANT]} Roads={counts[ROAD]}",
            f"Power: supply={city.power_supply} used={city.power_used} demand={city.power_demand} powered_buildings={powered}/{need_power}",
            "Controls: 1/2/3/4=House/Farm/Factory/Plant, R=Road tool, LMB place, RMB remove",
            "           M toggle mouse capture, WASD/Space/C move, F5/F6 JSON save/load, F9/F10 NPZ",
            "Adjacency: Houses +1 income per adjacent road (N/E/S/W). Citizens use roads. Roads transmit power."
        ]

    # Start with HUD text
    hud.update_text(hud_lines())

    while running:
        dt = clock.tick(120) / 1000.0

        for ev in pygame.event.get():
            if ev.type == QUIT:
                running = False
            elif ev.type == KEYDOWN:
                if ev.key == K_ESCAPE:
                    running = False
                elif ev.key == K_m:
                    mouse_captured = not mouse_captured
                    pygame.mouse.set_visible(not mouse_captured)
                    pygame.event.set_grab(mouse_captured)
                elif ev.unicode in ('1','2','3','4'):
                    city.selected = int(ev.unicode)
                    hud.update_text(hud_lines())
                elif ev.unicode.lower() == 'r':
                    city.selected = ROAD
                    hud.update_text(hud_lines())
                elif ev.key == pygame.K_F5:
                    city.save_json("city.json"); print("Saved city.json"); hud.update_text(hud_lines())
                elif ev.key == pygame.K_F6:
                    try:
                        city.load_json("city.json"); refresh_instances(force=True); print("Loaded city.json"); hud.update_text(hud_lines())
                    except Exception as e:
                        print("Load JSON failed:", e)
                elif ev.key == pygame.K_F9:
                    city.save_npz("city.npz"); print("Saved city.npz"); hud.update_text(hud_lines())
                elif ev.key == pygame.K_F10:
                    try:
                        city.load_npz("city.npz"); refresh_instances(force=True); print("Loaded city.npz"); hud.update_text(hud_lines())
                    except Exception as e:
                        print("Load NPZ failed:", e)
            elif ev.type in (VIDEORESIZE, WINDOWSIZECHANGED):
                w,h = screen.get_size()
                glViewport(0,0,w,max(1,h))
                proj = perspective(fovy, w/max(h,1), znear, zfar)
                update_matrices_ubo(ubo, proj, cam.view())
            elif ev.type == MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                origin, dirv = make_ray(mx, my, w, h, proj, cam.view())
                hit = ray_plane_y0_intersect(origin, dirv)
                if hit is not None:
                    i = int(math.floor(hit[0]))
                    k = int(math.floor(hit[2]))
                    if ev.button == 1:   # place
                        if city.place(i,k):
                            refresh_instances()
                            hud.update_text(hud_lines())
                    elif ev.button == 3: # remove
                        if city.remove(i,k):
                            refresh_instances()
                            hud.update_text(hud_lines())

        # Movement
        keys = pygame.key.get_pressed()
        move = np.zeros(3, np.float32)
        speed = 8.0
        if keys[K_w]: move += cam.front
        if keys[K_s]: move -= cam.front
        if keys[K_d]: move += cam.right
        if keys[K_a]: move -= cam.right
        if keys[K_SPACE]: move += cam.up
        if keys[K_c]: move -= cam.up
        if np.linalg.norm(move) > 0: cam.move(normalize(move), speed, dt)

        # Mouse look
        if mouse_captured:
            dx, dy = pygame.mouse.get_rel()
            cam.process_mouse(dx, dy, 0.15)
        else:
            pygame.mouse.get_rel()

        # Hover tile computation
        mx, my = pygame.mouse.get_pos()
        origin, dirv = make_ray(mx, my, w, h, proj, cam.view())
        hit = ray_plane_y0_intersect(origin, dirv)
        hover_tile = None
        if hit is not None:
            i = int(math.floor(hit[0])); k = int(math.floor(hit[2]))
            if 0 <= i < GRID and 0 <= k < GRID:
                hover_tile = (i, k)
        # Economy + population
        income = city.income_tick()
        pop.update(dt)
        inst_people.upload(pop.instanced_data())

        # Update UBO
        update_matrices_ubo(ubo, proj, cam.view())

        # Draw 3D
        glClearColor(0.12,0.15,0.17,1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(prog)
        # Ground
        glUniformMatrix4fv(u_model, 1, GL_FALSE, ground_model.T.astype(np.float32))
        glBindVertexArray(ground_vao)
        glDrawElements(GL_TRIANGLES, ground_icount, GL_UNSIGNED_INT, ctypes.c_void_p(0))
        glBindVertexArray(0)
        # Roads (thin: scale Y)
        glBindVertexArray(cube_vao_roads)
        model_roads = scale(1.0, 0.1, 1.0)  # thin slabs centered at i+0.5, y slightly above ground (set in offsets)
        glUniformMatrix4fv(u_model, 1, GL_FALSE, model_roads.T.astype(np.float32))
        if inst_roads.count > 0:
            glDrawElementsInstanced(GL_TRIANGLES, cube_icount, GL_UNSIGNED_INT, ctypes.c_void_p(0), inst_roads.count)
        # Buildings (full height)
        model_b = scale(1.0, 1.0, 1.0)
        glUniformMatrix4fv(u_model, 1, GL_FALSE, model_b.T.astype(np.float32))
        glBindVertexArray(cube_vao_buildings)
        if inst_buildings.count > 0:
            glDrawElementsInstanced(GL_TRIANGLES, cube_icount, GL_UNSIGNED_INT, ctypes.c_void_p(0), inst_buildings.count)
        # Citizens (tiny cubes)
        model_p = scale(0.25, 0.25, 0.25)
        glUniformMatrix4fv(u_model, 1, GL_FALSE, model_p.T.astype(np.float32))
        glBindVertexArray(cube_vao_people)
        if inst_people.count > 0:
            glDrawElementsInstanced(GL_TRIANGLES, cube_icount, GL_UNSIGNED_INT, ctypes.c_void_p(0), inst_people.count)

        glBindVertexArray(0)
        # Wireframe hover cube (preview)
        if hover_tile is not None:
            i, k = hover_tile
            height = 0.1 if city.selected == ROAD else 1.0
            yoff   = 0.05 if city.selected == ROAD else 0.5
            model_preview = translation([i + 0.5, yoff, k + 0.5]) @ scale(1.001, height, 1.001)

            glUseProgram(prog)  # <-- ensure the right program is current
            glUniformMatrix4fv(u_model, 1, GL_FALSE, model_preview.T.astype(np.float32))
            glBindVertexArray(cube_vao_buildings)

            # Temporarily disable instanced attributes so we don't read from empty instance buffers
            glDisableVertexAttribArray(2)  # iOffset
            glDisableVertexAttribArray(3)  # iColor
            # Provide constant "current values" for disabled attribs (shader multiplies by aColor anyway)
            glVertexAttrib3f(2, 0.0, 0.0, 0.0)  # iOffset
            glVertexAttrib3f(3, 1.0, 1.0, 0.0)  # iColor (yellow wireframe)

            glEnable(GL_POLYGON_OFFSET_LINE); glPolygonOffset(-1.0, -1.0)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glDrawElements(GL_TRIANGLES, cube_icount, GL_UNSIGNED_INT, ctypes.c_void_p(0))
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glDisable(GL_POLYGON_OFFSET_LINE)

            # Restore instanced attributes for later instanced draws
            glEnableVertexAttribArray(2)
            glEnableVertexAttribArray(3)
            glBindVertexArray(0)
        
        hud.draw(hud_lines())

        pygame.display.flip()
    pygame.quit()

if __name__ == "__main__":
    main()
