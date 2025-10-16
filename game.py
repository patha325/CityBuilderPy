# city_builder_pygame_opengl.py
# Single-file, modern OpenGL 3.3 + pygame "city builder" skeleton:
# - UBO for projection/view
# - Indexed cube mesh
# - Instanced buildings (colors per instance)
# - Grid picking (mouse ray -> ground plane)
# - Simple economy & HUD via window caption

import sys, time, math, ctypes
import numpy as np
import pygame
from pygame.locals import (
    DOUBLEBUF, OPENGL, RESIZABLE, VIDEORESIZE, WINDOWSIZECHANGED,
    QUIT, KEYDOWN, K_ESCAPE, K_m, K_w, K_a, K_s, K_d, K_SPACE, K_c,
    MOUSEBUTTONDOWN
)

from OpenGL.GL import (
    # info / state
    glGetString, GL_VERSION, GL_VENDOR, GL_RENDERER, GL_SHADING_LANGUAGE_VERSION,
    glViewport, glClearColor, glClear, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    glEnable, GL_DEPTH_TEST, glDepthFunc, GL_LEQUAL,
    # shaders / program
    glCreateShader, glShaderSource, glCompileShader, glGetShaderiv, glGetShaderInfoLog,
    glCreateProgram, glAttachShader, glLinkProgram, glGetProgramiv, glGetProgramInfoLog,
    glDeleteShader, glUseProgram, glGetUniformLocation, glUniform1f, glUniformMatrix4fv,
    glGetUniformBlockIndex, glUniformBlockBinding,
    # buffers / arrays
    glGenVertexArrays, glBindVertexArray,
    glGenBuffers, glBindBuffer, glBufferData, glBufferSubData, glBindBufferRange,
    glEnableVertexAttribArray, glVertexAttribPointer,
    glDrawElements, glDrawElementsInstanced,
    glVertexAttribDivisor,
    # enums
    GL_TRUE, GL_FALSE,
    GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
    GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER,
    GL_STATIC_DRAW, GL_DYNAMIC_DRAW,
    GL_FLOAT, GL_UNSIGNED_INT, GL_TRIANGLES,
    GL_UNIFORM_BUFFER
)

# ---------------- Shaders ----------------
VERT_SRC = """#version 330 core
layout (location=0) in vec3 aPos;
layout (location=1) in vec3 aColor;

layout (location=2) in vec3 iOffset; // per-instance world offset (x,z on grid; y is base)
layout (location=3) in vec3 iColor;  // per-instance color

out vec3 vColor;

layout (std140) uniform Matrices {
    mat4 uProjection;
    mat4 uView;
};

uniform mat4 uModel;   // for ground/cube base transforms (no per-instance scale/rot here)

void main(){
    // Raise cubes to sit on the ground (add 0.5 to y so 1x1x1 cube rests on plane y=0)
    vec3 worldPos = aPos + iOffset + vec3(0.0, 0.5, 0.0);
    gl_Position = uProjection * uView * uModel * vec4(worldPos, 1.0);
    vColor = iColor * aColor; // modulate base color by instance color
}
"""

FRAG_SRC = """#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main(){
    FragColor = vec4(vColor, 1.0);
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
    f = normalize(center - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float32)
    M[0,0:3] = s; M[1,0:3] = u; M[2,0:3] = -f
    T = np.eye(4, dtype=np.float32)
    T[0,3], T[1,3], T[2,3] = -eye
    return M @ T

def rotation_y(a):
    c,s = math.cos(a), math.sin(a)
    M = np.eye(4, dtype=np.float32)
    M[0,0]=c; M[0,2]=s; M[2,0]=-s; M[2,2]=c
    return M

def scale(sx, sy, sz):
    M = np.eye(4, dtype=np.float32)
    M[0,0]=sx; M[1,1]=sy; M[2,2]=sz
    return M

def translation(t):
    M = np.eye(4, dtype=np.float32)
    M[0,3], M[1,3], M[2,3] = t
    return M

def invert(m):
    return np.linalg.inv(m)

# ---------------- GL helpers ----------------
def compile_shader(src, stype):
    sid = glCreateShader(stype)
    glShaderSource(sid, src)
    glCompileShader(sid)
    ok = ctypes.c_int(0)
    glGetShaderiv(sid, 0x8B81, ok)  # GL_COMPILE_STATUS
    if ok.value != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(sid).decode())
    return sid

def link_program(vs, fs):
    pid = glCreateProgram()
    glAttachShader(pid, vs); glAttachShader(pid, fs)
    glLinkProgram(pid)
    ok = ctypes.c_int(0)
    glGetProgramiv(pid, 0x8B82, ok)  # GL_LINK_STATUS
    if ok.value != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(pid).decode())
    glDeleteShader(vs); glDeleteShader(fs)
    return pid

# -------------- Geometry (cube + ground) --------------
def make_cube():
    # Unit cube centered at origin, per-vertex "base" color = 1 (so instance color shows)
    verts = np.array([
        # pos                # base color
        -0.5,-0.5,-0.5,      1,1,1,
         0.5,-0.5,-0.5,      1,1,1,
         0.5, 0.5,-0.5,      1,1,1,
        -0.5, 0.5,-0.5,      1,1,1,
        -0.5,-0.5, 0.5,      1,1,1,
         0.5,-0.5, 0.5,      1,1,1,
         0.5, 0.5, 0.5,      1,1,1,
        -0.5, 0.5, 0.5,      1,1,1,
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
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))

    # Instance buffer placeholder (created later)
    glBindVertexArray(0)
    return vao, idx.size

def make_ground():
    # A flat 2x2 quad at origin on y=0 (we'll scale to grid)
    verts = np.array([
        -1, 0, -1,   0.8,0.8,0.8,
         1, 0, -1,   0.8,0.8,0.8,
         1, 0,  1,   0.8,0.8,0.8,
        -1, 0,  1,   0.8,0.8,0.8,
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

# -------------- UBO (proj/view) --------------
def create_matrices_ubo():
    ubo = glGenBuffers(1)
    glBindBuffer(GL_UNIFORM_BUFFER, ubo)
    glBufferData(GL_UNIFORM_BUFFER, 2*64, None, GL_DYNAMIC_DRAW)
    glBindBufferRange(GL_UNIFORM_BUFFER, 0, ubo, 0, 2*64)
    glBindBuffer(GL_UNIFORM_BUFFER, 0)
    return ubo

def update_matrices_ubo(ubo, proj, view):
    # NOTE: transpose -> column-major for std140
    glBindBuffer(GL_UNIFORM_BUFFER, ubo)
    glBufferSubData(GL_UNIFORM_BUFFER, 0,   64, proj.T.astype(np.float32))
    glBufferSubData(GL_UNIFORM_BUFFER, 64,  64, view.T.astype(np.float32))
    glBindBuffer(GL_UNIFORM_BUFFER, 0)

# -------------- Camera --------------
class Camera:
    def __init__(self, pos=np.array([0.0, 4.0, 6.0], dtype=np.float32), yaw=-90.0, pitch=-25.0):
        self.pos = pos.astype(np.float32)
        self.yaw = yaw; self.pitch = pitch
        self.front = np.array([0,0,-1], dtype=np.float32)
        self.up = np.array([0,1,0], dtype=np.float32)
        self.right = np.array([1,0,0], dtype=np.float32)
        self._update_vectors()
    def _update_vectors(self):
        cx, sx = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        cy, sy = math.cos(math.radians(self.pitch)), math.sin(math.radians(self.pitch))
        self.front = normalize(np.array([cx*cy, sy, sx*cy], dtype=np.float32))
        self.right = normalize(np.cross(self.front, np.array([0,1,0], dtype=np.float32)))
        self.up = normalize(np.cross(self.right, self.front))
    def process_mouse(self, dx, dy, sensitivity=0.12):
        self.yaw += dx * sensitivity
        self.pitch = max(-89.0, min(89.0, self.pitch - dy * sensitivity))
        self._update_vectors()
    def move(self, dir_vec, speed, dt):
        self.pos += dir_vec * (speed*dt)
    def view_matrix(self):
        return look_at(self.pos, self.pos + self.front, self.up)

# -------------- Instancing --------------
class InstanceBuffer:
    def __init__(self, vao):
        self.vao = vao
        self.vbo = glGenBuffers(1)
        self.count = 0
        # attach to VAO as locations 2 (offset) and 3 (color)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        stride = 6 * 4  # 3 floats offset + 3 floats color
        # iOffset @ loc=2
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glVertexAttribDivisor(2, 1)
        # iColor @ loc=3
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glVertexAttribDivisor(3, 1)
        glBindVertexArray(0)
    def upload(self, offsets_colors: np.ndarray):
        """offsets_colors: shape (N,6) -> [ox,oy,oz, r,g,b]"""
        self.count = 0 if offsets_colors is None else offsets_colors.shape[0]
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        if self.count == 0:
            glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)
        else:
            glBufferData(GL_ARRAY_BUFFER, offsets_colors.nbytes, offsets_colors.astype(np.float32), GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

# -------------- Picking --------------
def ndc_from_mouse(mx, my, w, h):
    x = (2.0 * mx) / max(w,1) - 1.0
    y = 1.0 - (2.0 * my) / max(h,1)
    return x, y

def make_ray(mx, my, w, h, proj, view):
    inv = invert(proj @ view)
    x, y = ndc_from_mouse(mx, my, w, h)
    near = np.array([x, y, -1.0, 1.0], dtype=np.float32)
    far  = np.array([x, y,  1.0, 1.0], dtype=np.float32)
    p_near = inv @ near; p_near = p_near[:3] / max(p_near[3], 1e-6)
    p_far  = inv @ far;  p_far  = p_far[:3]  / max(p_far[3],  1e-6)
    dirv = normalize(p_far - p_near)
    return p_near, dirv

def ray_plane_y0_intersect(origin, dirv):
    if abs(dirv[1]) < 1e-6:
        return None
    t = -origin[1] / dirv[1]
    if t < 0: return None
    return origin + t * dirv

# -------------- Game data --------------
BUILDINGS = {
    1: {"name":"House",   "color":(0.9,0.5,0.5), "cost":10,  "income":1},
    2: {"name":"Farm",    "color":(0.5,0.9,0.5), "cost":15,  "income":2},
    3: {"name":"Factory", "color":(0.5,0.6,0.9), "cost":30,  "income":5},
}

class City:
    def __init__(self, size):
        self.N = size
        self.grid = np.zeros((self.N, self.N), dtype=np.int8)  # 0 = empty, else building id
        self.money = 50
        self.selected = 1
        self.last_income = time.perf_counter()
    def can_place(self, i, k):
        return 0 <= i < self.N and 0 <= k < self.N and self.grid[i,k] == 0
    def place(self, i, k):
        b = BUILDINGS[self.selected]
        if self.money >= b["cost"] and self.can_place(i,k):
            self.money -= b["cost"]
            self.grid[i,k] = self.selected
            return True
        return False
    def remove(self, i, k):
        if 0 <= i < self.N and 0 <= k < self.N and self.grid[i,k] != 0:
            bid = int(self.grid[i,k])
            refund = int(BUILDINGS[bid]["cost"] * 0.5)
            self.money += refund
            self.grid[i,k] = 0
            return True
        return False
    def tick_income(self):
        now = time.perf_counter()
        if now - self.last_income >= 1.0:
            income = 0
            for bid, props in BUILDINGS.items():
                count = int(np.sum(self.grid == bid))
                income += count * props["income"]
            self.money += income
            self.last_income = now

    def instanced_data(self):
        """Return per-instance array (N,6) [ox,oy,oz, r,g,b]"""
        offsets = []
        colors = []
        for i in range(self.N):
            for k in range(self.N):
                bid = int(self.grid[i,k])
                if bid != 0:
                    # center at (i+0.5, 0, k+0.5), y base 0
                    offsets.append([i + 0.5, 0.0, k + 0.5])
                    colors.append(list(BUILDINGS[bid]["color"]))
        if not offsets:
            return np.zeros((0,6), dtype=np.float32)
        oc = np.concatenate([np.asarray(offsets, np.float32),
                             np.asarray(colors,  np.float32)], axis=1)
        return oc

# -------------- Pygame init --------------
def init_pygame(width=1280, height=720, title="City Builder (OpenGL 3.3)"):
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
    GRID = 32  # grid side length
    city = City(GRID)

    screen = init_pygame()
    w, h = screen.get_size()
    glViewport(0,0,w,h)

    print("GL_VENDOR:  ", glGetString(GL_VENDOR).decode())
    print("GL_RENDERER:", glGetString(GL_RENDERER).decode())
    print("GL_VERSION: ", glGetString(GL_VERSION).decode())
    print("GLSL:       ", glGetString(GL_SHADING_LANGUAGE_VERSION).decode())

    glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LEQUAL)

    # Compile/link + program setup
    vs = compile_shader(VERT_SRC, GL_VERTEX_SHADER)
    fs = compile_shader(FRAG_SRC, GL_FRAGMENT_SHADER)
    program = link_program(vs, fs)

    # UBO binding
    ubo = create_matrices_ubo()
    block = glGetUniformBlockIndex(program, "Matrices")
    glUniformBlockBinding(program, block, 0)

    # Meshes
    cube_vao, cube_index_count = make_cube()
    ground_vao, ground_index_count = make_ground()
    instances = InstanceBuffer(cube_vao)

    # Per-draw uniforms
    u_loc_model = glGetUniformLocation(program, "uModel")
    u_loc_time  = glGetUniformLocation(program, "uTime")  # kept for parity (unused here)

    # Camera/projection
    cam = Camera()
    fovy, znear, zfar = 60.0, 0.1, 500.0
    proj = perspective(fovy, w/max(h,1), znear, zfar)
    update_matrices_ubo(ubo, proj, cam.view_matrix())

    # Ground transform: scale to grid and center at grid midpoint
    ground_model = translation([GRID/2.0, 0.0, GRID/2.0]) @ scale(GRID/2.0, 1.0, GRID/2.0)

    # Initial instance data
    instances.upload(city.instanced_data())

    # Input & loop
    mouse_captured = False
    pygame.mouse.set_visible(True)
    pygame.event.set_grab(False)
    clock = pygame.time.Clock()
    t0 = time.perf_counter()
    running = True

    def set_caption():
        sel = city.selected
        name = BUILDINGS[sel]["name"]
        counts = {bid: int(np.sum(city.grid == bid)) for bid in BUILDINGS}
        cap = f"City Builder — Money: {city.money} — Selected: {sel}:{name} — " + \
              " | ".join([f"{BUILDINGS[b]['name']}={counts[b]}" for b in BUILDINGS])
        pygame.display.set_caption(cap)

    set_caption()

    while running:
        dt = clock.tick(120) / 1000.0
        for ev in pygame.event.get():
            if ev.type == QUIT: running = False
            elif ev.type == KEYDOWN:
                if ev.key == K_ESCAPE: running = False
                elif ev.key == K_m:
                    mouse_captured = not mouse_captured
                    pygame.mouse.set_visible(not mouse_captured)  # toggle cursor
                    pygame.event.set_grab(mouse_captured)
                elif ev.unicode in ('1','2','3'):
                    city.selected = int(ev.unicode)
                    set_caption()
            elif ev.type in (VIDEORESIZE, WINDOWSIZECHANGED):
                w,h = screen.get_size()
                glViewport(0,0,w,max(1,h))
                proj = perspective(fovy, w/max(h,1), znear, zfar)
                update_matrices_ubo(ubo, proj, cam.view_matrix())
            elif ev.type == MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                origin, dirv = make_ray(mx, my, w, h, proj, cam.view_matrix())
                hit = ray_plane_y0_intersect(origin, dirv)
                if hit is not None:
                    i = int(math.floor(hit[0]))
                    k = int(math.floor(hit[2]))
                    if ev.button == 1:   # LMB place
                        if city.place(i,k):
                            instances.upload(city.instanced_data())
                            set_caption()
                    elif ev.button == 3: # RMB remove
                        if city.remove(i,k):
                            instances.upload(city.instanced_data())
                            set_caption()

        # Movement
        keys = pygame.key.get_pressed()
        move = np.zeros(3, np.float32)
        speed = 6.0
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

        # Economy tick
        city.tick_income()

        # Update UBO (view)
        update_matrices_ubo(ubo, proj, cam.view_matrix())

        # Draw
        glClearColor(0.12, 0.15, 0.17, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(program)
        # Ground
        glUniformMatrix4fv(u_loc_model, 1, GL_FALSE, ground_model.T.astype(np.float32))
        glUniform1f(u_loc_time, time.perf_counter() - t0)
        glBindVertexArray(ground_vao)
        glDrawElements(GL_TRIANGLES, ground_index_count, GL_UNSIGNED_INT, ctypes.c_void_p(0))
        glBindVertexArray(0)

        # Buildings (instances)
        glBindVertexArray(cube_vao)
        glUniformMatrix4fv(u_loc_model, 1, GL_FALSE, np.eye(4, dtype=np.float32).T)  # identity
        if instances.count > 0:
            glDrawElementsInstanced(GL_TRIANGLES, cube_index_count, GL_UNSIGNED_INT, ctypes.c_void_p(0), instances.count)
        glBindVertexArray(0)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
