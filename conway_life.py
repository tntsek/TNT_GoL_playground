#!/opt/homebrew/bin/python3.12
"""Conway's Game of Life with image import, pixel editing, text stamping,
selection tool, and adjustable speed.

Large-grid computation runs asynchronously in a worker process so the UI
stays responsive at 60 fps regardless of grid size.
"""

import pygame
import pygame.freetype
from PIL import Image, ImageEnhance
import sys
import os
from concurrent.futures import ProcessPoolExecutor

# ---------------------------------------------------------------------------
# Game of Life logic
# ---------------------------------------------------------------------------

def make_grid(rows, cols, val=0):
    return [[val] * cols for _ in range(rows)]


def copy_grid(grid):
    return [row[:] for row in grid]


def step_grid(grid, wrap=True):
    """Advance one generation (list-of-lists). Used for small grids."""
    rows = len(grid)
    cols = len(grid[0])
    new = make_grid(rows, cols)
    for r in range(rows):
        for c in range(cols):
            total = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if wrap:
                        nr %= rows
                        nc %= cols
                    else:
                        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                            continue
                    total += grid[nr][nc]
            if grid[r][c]:
                new[r][c] = 1 if total in (2, 3) else 0
            else:
                new[r][c] = 1 if total == 3 else 0
    return new


# ---------------------------------------------------------------------------
# Worker function for async computation (runs in separate process)
# Uses flat bytes for fast inter-process transfer.
# ---------------------------------------------------------------------------

def _step_worker(data, rows, cols, wrap, steps=1):
    """Compute *steps* generations on flat byte data. Returns flat bytes."""
    buf = bytearray(data)
    out = bytearray(rows * cols)
    for _ in range(steps):
        for r in range(rows):
            ri = r * cols
            for c in range(cols):
                total = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if wrap:
                            nr %= rows
                            nc %= cols
                        elif nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                            continue
                        total += buf[nr * cols + nc]
                alive = buf[ri + c]
                if alive:
                    out[ri + c] = 1 if total == 2 or total == 3 else 0
                else:
                    out[ri + c] = 1 if total == 3 else 0
            # end col
        # end row — swap buffers
        buf, out = out, bytearray(rows * cols)
    return bytes(buf)


def grid_population(grid):
    return sum(sum(row) for row in grid)


# ---------------------------------------------------------------------------
# Pixel font bitmaps — widths 3, 4, 5 (all 5 rows tall)
# Each character maps to a tuple of row-strings ("1"=pixel, "0"=empty).
# ---------------------------------------------------------------------------

PIXEL_FONTS = {
    3: {
        'A': ("010","101","111","101","101"),
        'B': ("110","101","110","101","110"),
        'C': ("011","100","100","100","011"),
        'D': ("110","101","101","101","110"),
        'E': ("111","100","110","100","111"),
        'F': ("111","100","110","100","100"),
        'G': ("011","100","101","101","011"),
        'H': ("101","101","111","101","101"),
        'I': ("111","010","010","010","111"),
        'J': ("011","001","001","101","010"),
        'K': ("101","101","110","101","101"),
        'L': ("100","100","100","100","111"),
        'M': ("101","111","101","101","101"),
        'N': ("101","111","111","101","101"),
        'O': ("010","101","101","101","010"),
        'P': ("110","101","110","100","100"),
        'Q': ("010","101","101","010","001"),
        'R': ("110","101","110","101","101"),
        'S': ("011","100","010","001","110"),
        'T': ("111","010","010","010","010"),
        'U': ("101","101","101","101","010"),
        'V': ("101","101","101","010","010"),
        'W': ("101","101","111","111","101"),
        'X': ("101","101","010","101","101"),
        'Y': ("101","101","010","010","010"),
        'Z': ("111","001","010","100","111"),
        '0': ("111","101","101","101","111"),
        '1': ("010","110","010","010","111"),
        '2': ("110","001","010","100","111"),
        '3': ("110","001","010","001","110"),
        '4': ("101","101","111","001","001"),
        '5': ("111","100","110","001","110"),
        '6': ("011","100","111","101","010"),
        '7': ("111","001","010","100","100"),
        '8': ("010","101","010","101","010"),
        '9': ("010","101","011","001","010"),
        '.': ("000","000","000","000","010"),
        ',': ("000","000","000","010","100"),
        '!': ("010","010","010","000","010"),
        '?': ("110","001","010","000","010"),
        '-': ("000","000","111","000","000"),
        '+': ("000","010","111","010","000"),
        ':': ("000","010","000","010","000"),
        '/': ("001","001","010","100","100"),
        '(': ("010","100","100","100","010"),
        ')': ("010","001","001","001","010"),
        '_': ("000","000","000","000","111"),
        '=': ("000","111","000","111","000"),
        '#': ("101","111","101","111","101"),
        '*': ("101","010","101","000","000"),
    },
    4: {
        'A': ("0110","1001","1111","1001","1001"),
        'B': ("1110","1001","1110","1001","1110"),
        'C': ("0111","1000","1000","1000","0111"),
        'D': ("1110","1001","1001","1001","1110"),
        'E': ("1111","1000","1110","1000","1111"),
        'F': ("1111","1000","1110","1000","1000"),
        'G': ("0111","1000","1011","1001","0111"),
        'H': ("1001","1001","1111","1001","1001"),
        'I': ("1111","0110","0110","0110","1111"),
        'J': ("0011","0001","0001","1001","0110"),
        'K': ("1001","1010","1100","1010","1001"),
        'L': ("1000","1000","1000","1000","1111"),
        'M': ("1001","1111","1001","1001","1001"),
        'N': ("1001","1101","1011","1001","1001"),
        'O': ("0110","1001","1001","1001","0110"),
        'P': ("1110","1001","1110","1000","1000"),
        'Q': ("0110","1001","1001","0110","0011"),
        'R': ("1110","1001","1110","1010","1001"),
        'S': ("0111","1000","0110","0001","1110"),
        'T': ("1111","0110","0110","0110","0110"),
        'U': ("1001","1001","1001","1001","0110"),
        'V': ("1001","1001","1001","0110","0110"),
        'W': ("1001","1001","1001","1111","0110"),
        'X': ("1001","0110","0110","0110","1001"),
        'Y': ("1001","1001","0110","0110","0110"),
        'Z': ("1111","0001","0110","1000","1111"),
        '0': ("0110","1001","1001","1001","0110"),
        '1': ("0100","1100","0100","0100","1111"),
        '2': ("0110","1001","0010","0100","1111"),
        '3': ("1110","0001","0110","0001","1110"),
        '4': ("1001","1001","1111","0001","0001"),
        '5': ("1111","1000","1110","0001","1110"),
        '6': ("0110","1000","1110","1001","0110"),
        '7': ("1111","0001","0010","0100","0100"),
        '8': ("0110","1001","0110","1001","0110"),
        '9': ("0110","1001","0111","0001","0110"),
        '.': ("0000","0000","0000","0000","0100"),
        ',': ("0000","0000","0000","0100","1000"),
        '!': ("0110","0110","0110","0000","0110"),
        '?': ("0110","1001","0010","0000","0010"),
        '-': ("0000","0000","1111","0000","0000"),
        '+': ("0000","0100","1110","0100","0000"),
        ':': ("0000","0110","0000","0110","0000"),
        '/': ("0001","0010","0010","0100","1000"),
        '(': ("0010","0100","0100","0100","0010"),
        ')': ("0100","0010","0010","0010","0100"),
        '_': ("0000","0000","0000","0000","1111"),
        '=': ("0000","1111","0000","1111","0000"),
        '#': ("1010","1111","1010","1111","1010"),
        '*': ("1001","0110","1001","0000","0000"),
    },
    5: {
        'A': ("01110","10001","11111","10001","10001"),
        'B': ("11110","10001","11110","10001","11110"),
        'C': ("01111","10000","10000","10000","01111"),
        'D': ("11110","10001","10001","10001","11110"),
        'E': ("11111","10000","11110","10000","11111"),
        'F': ("11111","10000","11110","10000","10000"),
        'G': ("01111","10000","10011","10001","01111"),
        'H': ("10001","10001","11111","10001","10001"),
        'I': ("11111","00100","00100","00100","11111"),
        'J': ("00111","00001","00001","10001","01110"),
        'K': ("10001","10010","11100","10010","10001"),
        'L': ("10000","10000","10000","10000","11111"),
        'M': ("10001","11011","10101","10001","10001"),
        'N': ("10001","11001","10101","10011","10001"),
        'O': ("01110","10001","10001","10001","01110"),
        'P': ("11110","10001","11110","10000","10000"),
        'Q': ("01110","10001","10001","01110","00011"),
        'R': ("11110","10001","11110","10010","10001"),
        'S': ("01111","10000","01110","00001","11110"),
        'T': ("11111","00100","00100","00100","00100"),
        'U': ("10001","10001","10001","10001","01110"),
        'V': ("10001","10001","10001","01010","00100"),
        'W': ("10001","10001","10101","11011","10001"),
        'X': ("10001","01010","00100","01010","10001"),
        'Y': ("10001","01010","00100","00100","00100"),
        'Z': ("11111","00010","00100","01000","11111"),
        '0': ("01110","10001","10001","10001","01110"),
        '1': ("00100","01100","00100","00100","11111"),
        '2': ("01110","10001","00110","01000","11111"),
        '3': ("11110","00001","01110","00001","11110"),
        '4': ("10001","10001","11111","00001","00001"),
        '5': ("11111","10000","11110","00001","11110"),
        '6': ("01110","10000","11110","10001","01110"),
        '7': ("11111","00001","00010","00100","00100"),
        '8': ("01110","10001","01110","10001","01110"),
        '9': ("01110","10001","01111","00001","01110"),
        '.': ("00000","00000","00000","00000","00100"),
        ',': ("00000","00000","00000","00100","01000"),
        '!': ("00100","00100","00100","00000","00100"),
        '?': ("01110","10001","00110","00000","00100"),
        '-': ("00000","00000","11111","00000","00000"),
        '+': ("00000","00100","11111","00100","00000"),
        ':': ("00000","00100","00000","00100","00000"),
        '/': ("00001","00010","00100","01000","10000"),
        '(': ("00010","00100","00100","00100","00010"),
        ')': ("01000","00100","00100","00100","01000"),
        '_': ("00000","00000","00000","00000","11111"),
        '=': ("00000","11111","00000","11111","00000"),
        '#': ("01010","11111","01010","11111","01010"),
        '*': ("10001","01010","10001","00000","00000"),
    },
}


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

BG         = (17, 17, 17)
PANEL_BG   = (28, 28, 28)
CELL_ALIVE = (208, 208, 208)
CELL_DEAD  = (26, 26, 26)
GRID_LINE  = (42, 42, 42)
TEXT_COLOR  = (200, 200, 200)
TEXT_DIM    = (120, 120, 120)
BTN_BG     = (55, 55, 55)
BTN_HOVER  = (75, 75, 75)
BTN_GREEN  = (26, 107, 42)
BTN_GREEN_H= (34, 136, 58)
BTN_RED    = (107, 26, 26)
BTN_RED_H  = (136, 34, 34)
BTN_AMBER  = (107, 90, 26)
BTN_AMBER_H= (136, 116, 34)
BTN_BLUE   = (26, 60, 107)
BTN_BLUE_H = (34, 80, 136)
SLIDER_BG  = (50, 50, 50)
SLIDER_FG  = (74, 158, 255)
WHITE      = (255, 255, 255)
INPUT_BG   = (35, 35, 35)
INPUT_ACTIVE = (50, 60, 80)
INPUT_BORDER = (80, 80, 80)
INPUT_BORDER_ACTIVE = (74, 158, 255)
COMPUTING_COLOR = (255, 180, 50)
GHOST_COLOR = (60, 160, 90)          # text preview on grid
SEL_BORDER_COLOR = (74, 158, 255)    # selection rectangle border
SEL_FLOAT_COLOR = (74, 130, 200)     # floating selection tint

GRID_LINE_THICK = (70, 70, 70)        # every-5th-cell accent line

BTN_MODE_ACTIVE   = (40, 80, 130)
BTN_MODE_ACTIVE_H = (50, 100, 160)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

class Button:
    def __init__(self, rect, text, color=BTN_BG, hover_color=BTN_HOVER):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.hovered = False

    def draw(self, surf, font):
        c = self.hover_color if self.hovered else self.color
        pygame.draw.rect(surf, c, self.rect, border_radius=4)
        tw, th = font.get_rect(self.text)[2:4]
        x = self.rect.x + (self.rect.w - tw) // 2
        y = self.rect.y + (self.rect.h - th) // 2
        font.render_to(surf, (x, y), self.text, WHITE)

    def hit(self, pos):
        return self.rect.collidepoint(pos)

    def update_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)


class Slider:
    def __init__(self, rect, min_val, max_val, value, step=1, label="",
                 fmt=None):
        self.rect = pygame.Rect(rect)
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.step = step
        self.label = label
        self.fmt = fmt
        self.dragging = False

    def draw(self, surf, font):
        font.render_to(surf, (self.rect.x, self.rect.y - 18), self.label, TEXT_DIM)
        val_text = self.fmt(self.value) if self.fmt else str(self.value)
        vw = font.get_rect(val_text)[2]
        font.render_to(surf, (self.rect.right - vw, self.rect.y - 18), val_text, TEXT_COLOR)
        track_y = self.rect.y + self.rect.h // 2
        pygame.draw.line(surf, SLIDER_BG, (self.rect.x, track_y),
                         (self.rect.right, track_y), 4)
        frac = (self.value - self.min_val) / max(1e-9, self.max_val - self.min_val)
        fill_x = self.rect.x + int(frac * self.rect.w)
        pygame.draw.line(surf, SLIDER_FG, (self.rect.x, track_y),
                         (fill_x, track_y), 4)
        pygame.draw.circle(surf, WHITE, (fill_x, track_y), 7)

    def hit(self, pos):
        expanded = self.rect.inflate(0, 20)
        return expanded.collidepoint(pos)

    def update_drag(self, mx):
        frac = max(0.0, min(1.0, (mx - self.rect.x) / self.rect.w))
        raw = self.min_val + frac * (self.max_val - self.min_val)
        self.value = round(raw / self.step) * self.step
        if isinstance(self.step, float):
            self.value = round(self.value, 2)
            self.value = max(self.min_val, min(self.max_val, self.value))
        else:
            self.value = int(self.value)
            self.value = max(self.min_val, min(self.max_val, self.value))


class NumberInput:
    """A small text field for entering integer values."""

    def __init__(self, rect, value, min_val=1, max_val=999, label=""):
        self.rect = pygame.Rect(rect)
        self.value = value
        self.min_val = min_val
        self.max_val = max_val
        self.label = label
        self.active = False
        self.text = str(value)

    def draw(self, surf, font):
        if self.label:
            font.render_to(surf, (self.rect.x, self.rect.y - 16), self.label, TEXT_DIM)
        bg = INPUT_ACTIVE if self.active else INPUT_BG
        border = INPUT_BORDER_ACTIVE if self.active else INPUT_BORDER
        pygame.draw.rect(surf, bg, self.rect, border_radius=3)
        pygame.draw.rect(surf, border, self.rect, width=1, border_radius=3)
        display = self.text if self.active else str(self.value)
        if self.active:
            display += "|"
        tw, th = font.get_rect(display)[2:4]
        tx = self.rect.x + 6
        ty = self.rect.y + (self.rect.h - th) // 2
        font.render_to(surf, (tx, ty), display, TEXT_COLOR)

    def hit(self, pos):
        return self.rect.collidepoint(pos)

    def activate(self):
        self.active = True
        self.text = str(self.value)

    def deactivate(self):
        self.active = False
        self._commit()

    def handle_key(self, event):
        if not self.active:
            return False
        if event.key == pygame.K_RETURN or event.key == pygame.K_TAB:
            self.deactivate()
            return True
        elif event.key == pygame.K_ESCAPE:
            self.text = str(self.value)
            self.active = False
            return True
        elif event.key == pygame.K_BACKSPACE:
            self.text = self.text[:-1]
            return True
        elif event.unicode.isdigit():
            self.text += event.unicode
            return True
        return False

    def _commit(self):
        try:
            v = int(self.text)
            self.value = max(self.min_val, min(self.max_val, v))
        except ValueError:
            pass
        self.text = str(self.value)


class TextInput:
    """A text field for entering arbitrary text (for the text-stamp feature)."""

    def __init__(self, rect, label="", max_len=64):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.max_len = max_len
        self.active = False
        self.text = ""

    def draw(self, surf, font):
        if self.label:
            font.render_to(surf, (self.rect.x, self.rect.y - 16), self.label, TEXT_DIM)
        bg = INPUT_ACTIVE if self.active else INPUT_BG
        border = INPUT_BORDER_ACTIVE if self.active else INPUT_BORDER
        pygame.draw.rect(surf, bg, self.rect, border_radius=3)
        pygame.draw.rect(surf, border, self.rect, width=1, border_radius=3)
        display = self.text
        if self.active:
            display += "|"
        # Truncate display if too wide
        max_w = self.rect.w - 12
        rct = font.get_rect(display)
        tw, th = rct[2], rct[3]
        shown = display
        while tw > max_w and len(shown) > 1:
            shown = shown[1:]
            tw = font.get_rect(shown)[2]
        ty = self.rect.y + (self.rect.h - th) // 2
        font.render_to(surf, (self.rect.x + 6, ty), shown, TEXT_COLOR)

    def hit(self, pos):
        return self.rect.collidepoint(pos)

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def handle_key(self, event):
        if not self.active:
            return False
        if event.key == pygame.K_RETURN or event.key == pygame.K_TAB:
            self.deactivate()
            return True
        elif event.key == pygame.K_ESCAPE:
            self.deactivate()
            return True
        elif event.key == pygame.K_BACKSPACE:
            self.text = self.text[:-1]
            return True
        elif event.unicode and event.unicode.isprintable() and len(self.text) < self.max_len:
            self.text += event.unicode
            return True
        return False


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

# Threshold: grids with more cells than this use async computation
ASYNC_THRESHOLD = 4096
HISTORY_MAX = 500


class LifeApp:
    PANEL_W = 230

    def __init__(self):
        pygame.init()
        pygame.freetype.init()

        info = pygame.display.Info()
        self.win_w = min(1200, info.current_w - 100)
        self.win_h = min(800, info.current_h - 100)
        self.screen = pygame.display.set_mode((self.win_w, self.win_h),
                                               pygame.RESIZABLE)
        pygame.display.set_caption("Conway's Game of Life")

        self.font = pygame.freetype.SysFont("Helvetica,Arial,sans-serif", 13)
        self.font_sm = pygame.freetype.SysFont("Helvetica,Arial,sans-serif", 11)
        self.font_hd = pygame.freetype.SysFont("Helvetica,Arial,sans-serif", 15)
        self.font_hd.strong = True

        # Grid state
        self.grid_rows = 64
        self.grid_cols = 64
        self.grid = make_grid(self.grid_rows, self.grid_cols)
        self.generation = 0
        self.running = False
        self.speed = 10
        self.threshold = 128

        # Boundary
        self.wrap = True

        # Grid display
        self.thick_lines = False  # thicker line every 5 cells

        # History
        self.history = []

        # Generation 0 snapshot (for reset)
        self.gen0_grid = None   # saved by first step or on image load
        self.gen0_rows = 0
        self.gen0_cols = 0

        # Source image
        self.source_image = None
        self.img_rotation = 0
        self.dark_is_alive = True
        self.img_contrast = 1.0

        # Display
        self.alive_is_light = True

        # Drawing
        self.drawing = False
        self.draw_value = 1

        # Zoom & pan
        self.zoom = 1.0       # 1.0 = fit-to-window
        self.pan_x = 0.0      # pixel offset from centred position
        self.pan_y = 0.0
        self._panning = False  # middle-mouse or alt-click panning
        self._pan_anchor = None

        # Play direction
        self.play_direction = 1  # +1 forward, -1 backward

        # --- Interaction mode ---
        self.mode = "draw"  # "draw", "text", "select"

        # Text mode state
        self.text_char_width = 3
        self.text_char_height = 5
        self.text_spacing = 1
        self.text_space_width = 3
        self.text_cursor_grid = None  # (row, col) preview anchor

        # Select mode state
        self.sel_state = None      # None, "selecting", "floating"
        self.sel_start = None      # (row, col) rubber-band anchor
        self.sel_end = None        # (row, col) rubber-band end
        self.sel_data = None       # 2D list of cell values
        self.sel_data_rows = 0
        self.sel_data_cols = 0
        self.sel_origin = None     # (row, col) where data was lifted
        self.sel_offset = None     # (row, col) current float position
        self.sel_dragging = False  # mouse-dragging the floating selection
        self.sel_drag_anchor = None  # (row, col) offset for drag

        # Timing
        self.clock = pygame.time.Clock()
        self.accum_ms = 0.0

        # Async computation
        self._pool = ProcessPoolExecutor(max_workers=1)
        self._pending_future = None  # Future or None
        self._pending_gen = 0        # generation count the future will produce

        self._build_ui()

    # -----------------------------------------------------------------------
    # Grid ↔ flat bytes conversion
    # -----------------------------------------------------------------------

    def _grid_to_bytes(self):
        rows, cols = self.grid_rows, self.grid_cols
        buf = bytearray(rows * cols)
        idx = 0
        for r in range(rows):
            row = self.grid[r]
            for c in range(cols):
                buf[idx] = row[c]
                idx += 1
        return bytes(buf)

    def _bytes_to_grid(self, data):
        rows, cols = self.grid_rows, self.grid_cols
        self.grid = make_grid(rows, cols)
        idx = 0
        for r in range(rows):
            row = self.grid[r]
            for c in range(cols):
                row[c] = data[idx]
                idx += 1

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        px = self.win_w - self.PANEL_W + 14
        pw = self.PANEL_W - 28
        bh = 30
        gap = 5
        input_h = 24

        y = 10
        self._header_y = y; y += 22

        self.btn_load = Button((px, y, pw, bh), "Load Image", BTN_BG, BTN_HOVER)
        y += bh + gap

        self._dim_label_y = y; y += 16
        input_w = (pw - gap - 20) // 2
        self.input_width = NumberInput((px, y, input_w, input_h),
                                       self.grid_cols, 4, 512, "Width")
        x_label_x = px + input_w + 4
        self._x_label_pos = (x_label_x, y + 2)
        self.input_height = NumberInput((px + input_w + 20, y, input_w, input_h),
                                        self.grid_rows, 4, 512, "Height")
        y += input_h + gap

        self.btn_apply = Button((px, y, pw, bh), "Apply Size", BTN_BG, BTN_HOVER)
        y += bh + gap

        half = (pw - gap) // 2
        self.btn_boundary = Button((px, y, half, bh),
                                   "Wrap" if self.wrap else "Fixed",
                                   BTN_BG, BTN_HOVER)
        self.btn_thick = Button((px + half + gap, y, half, bh),
                                "Grid 5s: On" if self.thick_lines else "Grid 5s: Off",
                                BTN_BG, BTN_HOVER)
        y += bh + gap

        self.slider_thresh = Slider((px, y + 18, pw, 14), 0, 255, self.threshold, 1, "Threshold")
        y += 44

        # --- Image adjust ---
        self._img_header_y = y; y += 22

        self.slider_contrast = Slider((px, y + 18, pw, 14), 0.5, 3.0, self.img_contrast, 0.1,
                                       "Contrast", fmt=lambda v: f"{v:.1f}x")
        y += 44

        self.btn_rotate = Button((px, y, pw, bh), "Rotate Image", BTN_BG, BTN_HOVER)
        y += bh + gap

        self.btn_reapply = Button((px, y, pw, bh), "Re-apply Image", BTN_BLUE, BTN_BLUE_H)
        y += bh + gap

        self._sep1_y = y; y += 10

        # --- Simulation ---
        self._sim_header_y = y; y += 22
        self._gen_y = y; y += 18
        self._pop_y = y; y += 20
        self._size_y = y; y += 22

        quarter = (pw - gap * 3) // 4
        self.btn_back = Button((px, y, quarter, bh), "< Back", BTN_BG, BTN_HOVER)
        self.btn_play_rev = Button((px + (quarter + gap), y, quarter, bh),
                                   "<< Play", BTN_BG, BTN_HOVER)
        self.btn_play = Button((px + (quarter + gap) * 2, y, quarter, bh),
                               "Play >>", BTN_GREEN, BTN_GREEN_H)
        self.btn_step = Button((px + (quarter + gap) * 3, y, quarter, bh),
                               "Step >", BTN_BG, BTN_HOVER)
        y += bh + gap + 2

        self.slider_speed = Slider((px, y + 18, pw, 14), 1, 60, self.speed, 1, "Speed (gen/s)")
        y += 46

        self._sep2_y = y; y += 10

        # --- Edit ---
        self._edit_header_y = y; y += 22

        # Mode selector buttons
        mode_w = (pw - gap * 2) // 3
        self.btn_mode_draw = Button((px, y, mode_w, bh), "Draw")
        self.btn_mode_text = Button((px + mode_w + gap, y, mode_w, bh), "Text")
        self.btn_mode_select = Button((px + (mode_w + gap) * 2, y, mode_w, bh), "Select")
        y += bh + gap

        # --- Mode-specific controls ---
        self._mode_controls_y = y

        # Text mode controls (always built, only shown in text mode)
        self.text_input = TextInput((px, y + 16, pw, input_h), "Text")
        text_y = y + 16 + input_h + gap + 4

        # Font config — two rows of two number inputs
        ti_w = (pw - gap - 30) // 2  # input width (leave room for labels)
        self.input_char_width = NumberInput((px, text_y + 16, ti_w, input_h),
                                            self.text_char_width, 3, 5, "Char W")
        self.input_char_height = NumberInput((px + ti_w + 30, text_y + 16, ti_w, input_h),
                                             self.text_char_height, 3, 20, "Char H")
        text_y += 16 + input_h + gap + 2

        self.input_text_spacing = NumberInput((px, text_y + 16, ti_w, input_h),
                                              self.text_spacing, 0, 10, "Spacing")
        self.input_space_width = NumberInput((px + ti_w + 30, text_y + 16, ti_w, input_h),
                                             self.text_space_width, 1, 20, "Space W")
        text_y += 16 + input_h + gap + 2

        self._text_hint_y = text_y
        text_y += 16

        # Calculate y offset based on mode
        if self.mode == "text":
            y = text_y
        elif self.mode == "select":
            self._select_hint_y = y
            y += 48  # room for instructions
        else:
            y += 2  # tiny gap for draw mode

        # --- Common edit buttons (below mode-specific area) ---
        half = (pw - gap) // 2
        self.btn_swap_live = Button((px, y, half, bh), "Swap Live", BTN_BG, BTN_HOVER)
        self.btn_swap_colors = Button((px + half + gap, y, half, bh), "Swap Clr", BTN_BG, BTN_HOVER)
        y += bh + gap

        self._legend_y = y
        y += 18

        self.btn_clear = Button((px, y, pw, bh), "Clear", BTN_RED, BTN_RED_H)
        y += bh + gap

        # Reset button — positioned here but only shown when generation > 0
        self._reset_btn_y = y
        self.btn_reset = Button((px, y, pw, bh), "Reset to Gen 0", BTN_BLUE, BTN_BLUE_H)

        self.buttons = [self.btn_load, self.btn_apply, self.btn_play,
                        self.btn_play_rev,
                        self.btn_step, self.btn_back, self.btn_clear,
                        self.btn_rotate, self.btn_reapply,
                        self.btn_swap_live, self.btn_swap_colors,
                        self.btn_boundary, self.btn_thick,
                        self.btn_mode_draw, self.btn_mode_text,
                        self.btn_mode_select]
        self.sliders = [self.slider_thresh, self.slider_speed,
                        self.slider_contrast]
        self.text_inputs = [self.input_char_width, self.input_char_height,
                            self.input_text_spacing, self.input_space_width]
        self.inputs = [self.input_width, self.input_height]

    def _reposition_ui(self):
        self.win_w, self.win_h = self.screen.get_size()
        self._build_ui()

    # -----------------------------------------------------------------------
    # Grid geometry
    # -----------------------------------------------------------------------

    def _grid_area(self):
        gw = self.win_w - self.PANEL_W
        gh = self.win_h
        return 0, 0, gw, gh

    def _cell_metrics(self):
        gx, gy, gw, gh = self._grid_area()
        rows, cols = self.grid_rows, self.grid_cols
        base_cell = min(gw / cols, gh / rows)
        cell = base_cell * self.zoom
        # Centre of grid area + pan offset
        cx = gx + gw / 2 + self.pan_x
        cy = gy + gh / 2 + self.pan_y
        ox = cx - cell * cols / 2
        oy = cy - cell * rows / 2
        return cell, ox, oy

    def _pixel_at(self, mx, my):
        cell, ox, oy = self._cell_metrics()
        c = int((mx - ox) // cell)
        r = int((my - oy) // cell)
        if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
            return r, c
        return None

    # -----------------------------------------------------------------------
    # Text rendering helpers
    # -----------------------------------------------------------------------

    def _get_text_pixels(self, text):
        """Return list of (dr, dc) offsets for live pixels in *text*.

        Glyphs are defined at 5 rows tall and are scaled vertically to
        *text_char_height* using nearest-neighbour sampling.
        """
        cw = self.text_char_width
        ch = self.text_char_height
        sp = self.text_spacing
        sw = self.text_space_width
        font = PIXEL_FONTS.get(cw, PIXEL_FONTS[3])
        pixels = []
        col_off = 0
        for char in text.upper():
            if char == ' ':
                col_off += sw
                continue
            glyph = font.get(char)
            if glyph is None:
                col_off += cw + sp
                continue
            # Scale 5-row glyph to ch rows (nearest-neighbour)
            for out_row in range(ch):
                src_row = int(out_row * 5 / ch)
                src_row = min(src_row, 4)
                row_str = glyph[src_row]
                for col_i, px in enumerate(row_str):
                    if px == '1':
                        pixels.append((out_row, col_off + col_i))
            col_off += cw + sp
        return pixels

    def _get_text_bounds(self, text):
        """Return (height, width) of the text block in cells."""
        cw = self.text_char_width
        ch = self.text_char_height
        sp = self.text_spacing
        sw = self.text_space_width
        w = 0
        for char in text.upper():
            if char == ' ':
                w += sw
            else:
                w += cw + sp
        # Remove trailing spacing
        if text and text[-1] != ' ':
            w -= sp
        return ch, max(w, 0)

    # -----------------------------------------------------------------------
    # Selection helpers
    # -----------------------------------------------------------------------

    def _sel_rect(self):
        """Return normalized (r1, c1, r2, c2) from sel_start/sel_end."""
        if self.sel_start is None or self.sel_end is None:
            return None
        r1 = min(self.sel_start[0], self.sel_end[0])
        c1 = min(self.sel_start[1], self.sel_end[1])
        r2 = max(self.sel_start[0], self.sel_end[0])
        c2 = max(self.sel_start[1], self.sel_end[1])
        return r1, c1, r2, c2

    def _capture_selection(self):
        """Lift cells from the grid into sel_data and clear the area."""
        rect = self._sel_rect()
        if rect is None:
            return
        r1, c1, r2, c2 = rect
        h = r2 - r1 + 1
        w = c2 - c1 + 1
        self.sel_data = []
        for r in range(r1, r2 + 1):
            row = []
            for c in range(c1, c2 + 1):
                row.append(self.grid[r][c])
                self.grid[r][c] = 0
            self.sel_data.append(row)
        self.sel_data_rows = h
        self.sel_data_cols = w
        self.sel_origin = (r1, c1)
        self.sel_offset = (r1, c1)
        self.sel_state = "floating"

    def _commit_selection(self):
        """Stamp floating selection onto the grid at the current offset."""
        if self.sel_data is None or self.sel_offset is None:
            self.sel_state = None
            return
        or_, oc = self.sel_offset
        for dr in range(self.sel_data_rows):
            for dc in range(self.sel_data_cols):
                gr = or_ + dr
                gc = oc + dc
                if 0 <= gr < self.grid_rows and 0 <= gc < self.grid_cols:
                    if self.sel_data[dr][dc]:
                        self.grid[gr][gc] = 1
        self._clear_selection()

    def _cancel_selection(self):
        """Put selection back at its original position."""
        if self.sel_data is None or self.sel_origin is None:
            self._clear_selection()
            return
        or_, oc = self.sel_origin
        for dr in range(self.sel_data_rows):
            for dc in range(self.sel_data_cols):
                gr = or_ + dr
                gc = oc + dc
                if 0 <= gr < self.grid_rows and 0 <= gc < self.grid_cols:
                    if self.sel_data[dr][dc]:
                        self.grid[gr][gc] = 1
        self._clear_selection()

    def _clear_selection(self):
        self.sel_state = None
        self.sel_start = None
        self.sel_end = None
        self.sel_data = None
        self.sel_origin = None
        self.sel_offset = None
        self.sel_dragging = False
        self.sel_drag_anchor = None

    def _move_selection(self, dr, dc):
        """Move the floating selection by (dr, dc) cells."""
        if self.sel_offset is None:
            return
        r, c = self.sel_offset
        self.sel_offset = (r + dr, c + dc)

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------

    def _draw(self):
        self.screen.fill(BG)
        self._draw_grid_area()
        self._draw_panel()
        pygame.display.flip()

    def _draw_grid_area(self):
        cell, ox, oy = self._cell_metrics()
        rows, cols = self.grid_rows, self.grid_cols

        if self.alive_is_light:
            ca, cd = CELL_ALIVE, CELL_DEAD
        else:
            ca, cd = CELL_DEAD, CELL_ALIVE

        # Build raw RGB bytes directly — much faster than PIL pixel-by-pixel
        raw = bytearray(rows * cols * 3)
        idx = 0
        for r in range(rows):
            row = self.grid[r]
            for c in range(cols):
                clr = ca if row[c] else cd
                raw[idx]   = clr[0]
                raw[idx+1] = clr[1]
                raw[idx+2] = clr[2]
                idx += 3

        surf = pygame.image.frombuffer(bytes(raw), (cols, rows), "RGB")

        target_w = int(cell * cols)
        target_h = int(cell * rows)
        if target_w < 1 or target_h < 1:
            return
        scaled = pygame.transform.scale(surf, (target_w, target_h))
        self.screen.blit(scaled, (int(ox), int(oy)))

        if cell >= 6:
            x_end = int(ox + cols * cell)
            y_end = int(oy + rows * cell)
            for r in range(rows + 1):
                yy = int(oy + r * cell)
                thick = self.thick_lines and r % 5 == 0
                w = 2 if thick else 1
                clr = GRID_LINE_THICK if thick else GRID_LINE
                pygame.draw.line(self.screen, clr,
                                 (int(ox), yy), (x_end, yy), w)
            for c in range(cols + 1):
                xx = int(ox + c * cell)
                thick = self.thick_lines and c % 5 == 0
                w = 2 if thick else 1
                clr = GRID_LINE_THICK if thick else GRID_LINE
                pygame.draw.line(self.screen, clr,
                                 (xx, int(oy)), (xx, y_end), w)

        # --- Overlays ---

        # Text ghost preview
        if self.mode == "text" and self.text_cursor_grid is not None:
            text = self.text_input.text
            if text:
                gr, gc = self.text_cursor_grid
                pixels = self._get_text_pixels(text)
                ghost_surf = pygame.Surface((int(cell), int(cell)), pygame.SRCALPHA)
                ghost_surf.fill((*GHOST_COLOR, 140))
                for dr, dc in pixels:
                    pr = gr + dr
                    pc = gc + dc
                    if 0 <= pr < rows and 0 <= pc < cols:
                        sx = int(ox + pc * cell)
                        sy = int(oy + pr * cell)
                        self.screen.blit(ghost_surf, (sx, sy))

        # Selection rubber-band
        if self.mode == "select" and self.sel_state == "selecting":
            rect = self._sel_rect()
            if rect:
                r1, c1, r2, c2 = rect
                sx = int(ox + c1 * cell)
                sy = int(oy + r1 * cell)
                sw = int((c2 - c1 + 1) * cell)
                sh = int((r2 - r1 + 1) * cell)
                sel_surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
                sel_surf.fill((74, 158, 255, 40))
                self.screen.blit(sel_surf, (sx, sy))
                pygame.draw.rect(self.screen, SEL_BORDER_COLOR,
                                 (sx, sy, sw, sh), 2)

        # Floating selection
        if self.mode == "select" and self.sel_state == "floating" and self.sel_data:
            or_, oc = self.sel_offset
            float_surf = pygame.Surface((int(cell), int(cell)), pygame.SRCALPHA)
            float_surf.fill((*SEL_FLOAT_COLOR, 160))
            for dr in range(self.sel_data_rows):
                for dc in range(self.sel_data_cols):
                    if self.sel_data[dr][dc]:
                        pr = or_ + dr
                        pc = oc + dc
                        sx = int(ox + pc * cell)
                        sy = int(oy + pr * cell)
                        self.screen.blit(float_surf, (sx, sy))
            # Border around floating area
            sx = int(ox + oc * cell)
            sy = int(oy + or_ * cell)
            sw = int(self.sel_data_cols * cell)
            sh = int(self.sel_data_rows * cell)
            pygame.draw.rect(self.screen, SEL_BORDER_COLOR,
                             (sx, sy, sw, sh), 2)

    def _draw_panel(self):
        px = self.win_w - self.PANEL_W
        pygame.draw.rect(self.screen, PANEL_BG,
                         (px, 0, self.PANEL_W, self.win_h))

        x = px + 14

        # Headers
        self.font_hd.render_to(self.screen, (x, self._header_y), "Image Import", TEXT_COLOR)
        self.font_hd.render_to(self.screen, (x, self._img_header_y), "Image Adjust", TEXT_COLOR)
        self.font_hd.render_to(self.screen, (x, self._sim_header_y), "Simulation", TEXT_COLOR)
        self.font_hd.render_to(self.screen, (x, self._edit_header_y), "Edit", TEXT_COLOR)

        # "x" between dimension inputs
        self.font.render_to(self.screen, self._x_label_pos, "x", TEXT_DIM)

        # Separators
        for sy in (self._sep1_y, self._sep2_y):
            pygame.draw.line(self.screen, GRID_LINE,
                             (x, sy + 4), (x + self.PANEL_W - 28, sy + 4))

        # Info
        gen_text = f"Generation: {self.generation}"
        if self._pending_future is not None:
            gen_text += "  computing..."
        self.font.render_to(self.screen, (x, self._gen_y), gen_text,
                            COMPUTING_COLOR if self._pending_future else TEXT_COLOR)
        self.font.render_to(self.screen, (x, self._pop_y),
                            f"Population: {grid_population(self.grid)}", TEXT_COLOR)
        self.font_sm.render_to(self.screen, (x, self._size_y),
                               f"Grid: {self.grid_cols} x {self.grid_rows}"
                               f"  ({'Wrap' if self.wrap else 'Fixed'})", TEXT_DIM)

        # Update play buttons — highlight the active direction
        if self.running and self.play_direction == 1:
            self.btn_play.text = "Pause"
            self.btn_play.color = BTN_AMBER
            self.btn_play.hover_color = BTN_AMBER_H
        else:
            self.btn_play.text = "Play >>"
            self.btn_play.color = BTN_GREEN
            self.btn_play.hover_color = BTN_GREEN_H

        if self.running and self.play_direction == -1:
            self.btn_play_rev.text = "Pause"
            self.btn_play_rev.color = BTN_AMBER
            self.btn_play_rev.hover_color = BTN_AMBER_H
        else:
            self.btn_play_rev.text = "<< Play"
            if self.history:
                self.btn_play_rev.color = BTN_GREEN
                self.btn_play_rev.hover_color = BTN_GREEN_H
            else:
                self.btn_play_rev.color = (40, 40, 40)
                self.btn_play_rev.hover_color = (40, 40, 40)

        # Update toggle button texts
        self.btn_boundary.text = "Wrap" if self.wrap else "Fixed"
        self.btn_thick.text = "Grid 5s: On" if self.thick_lines else "Grid 5s: Off"

        # Dim back button if no history
        if not self.history:
            self.btn_back.color = (40, 40, 40)
            self.btn_back.hover_color = (40, 40, 40)
        else:
            self.btn_back.color = BTN_BG
            self.btn_back.hover_color = BTN_HOVER

        # Dim image adjust buttons if no source image
        has_img = self.source_image is not None
        for btn in (self.btn_rotate, self.btn_reapply):
            if not has_img:
                btn.color = (40, 40, 40)
                btn.hover_color = (40, 40, 40)
            else:
                if btn is self.btn_reapply:
                    btn.color = BTN_BLUE
                    btn.hover_color = BTN_BLUE_H
                else:
                    btn.color = BTN_BG
                    btn.hover_color = BTN_HOVER

        # Mode button colors
        for mbtn, mname in ((self.btn_mode_draw, "draw"),
                            (self.btn_mode_text, "text"),
                            (self.btn_mode_select, "select")):
            if self.mode == mname:
                mbtn.color = BTN_MODE_ACTIVE
                mbtn.hover_color = BTN_MODE_ACTIVE_H
            else:
                mbtn.color = BTN_BG
                mbtn.hover_color = BTN_HOVER

        for b in self.buttons:
            b.draw(self.screen, self.font)
        for s in self.sliders:
            s.draw(self.screen, self.font_sm)
        for inp in self.inputs:
            inp.draw(self.screen, self.font_sm)

        # Mode-specific controls
        if self.mode == "text":
            self.text_input.draw(self.screen, self.font_sm)
            for ti in self.text_inputs:
                ti.draw(self.screen, self.font_sm)
            self.font_sm.render_to(self.screen, (x, self._text_hint_y),
                                   "Click grid to place text", TEXT_DIM)
        elif self.mode == "select":
            hy = self._mode_controls_y
            if self.sel_state == "floating":
                self.font_sm.render_to(self.screen, (x, hy),
                                       "Arrows/drag to move", TEXT_DIM)
                self.font_sm.render_to(self.screen, (x, hy + 14),
                                       "Enter=commit  Esc=cancel", TEXT_DIM)
            else:
                self.font_sm.render_to(self.screen, (x, hy),
                                       "Drag on grid to select area", TEXT_DIM)
                self.font_sm.render_to(self.screen, (x, hy + 14),
                                       "Then move with arrows/drag", TEXT_DIM)

        # Show reset button only when there's a gen0 snapshot and we've advanced
        show_reset = self.gen0_grid is not None and self.generation > 0
        if show_reset:
            self.btn_reset.draw(self.screen, self.font)

        # Live/Dead legend
        ly = self._legend_y
        sq = 12
        if self.alive_is_light:
            live_clr, dead_clr = CELL_ALIVE, CELL_DEAD
        else:
            live_clr, dead_clr = CELL_DEAD, CELL_ALIVE

        pygame.draw.rect(self.screen, live_clr, (x, ly, sq, sq))
        pygame.draw.rect(self.screen, INPUT_BORDER, (x, ly, sq, sq), 1)
        self.font_sm.render_to(self.screen, (x + sq + 4, ly), "= Live", TEXT_COLOR)

        lbl_w = self.font_sm.get_rect("= Live")[2]
        x2 = x + sq + 4 + lbl_w + 12
        pygame.draw.rect(self.screen, dead_clr, (x2, ly, sq, sq))
        pygame.draw.rect(self.screen, INPUT_BORDER, (x2, ly, sq, sq), 1)
        self.font_sm.render_to(self.screen, (x2 + sq + 4, ly), "= Dead", TEXT_COLOR)

        # Hint — anchor below reset button if visible, otherwise below clear
        if show_reset:
            hint_y = self.btn_reset.rect.bottom + 10
        else:
            hint_y = self.btn_clear.rect.bottom + 10
        self.font_sm.render_to(self.screen, (x, hint_y),
                               "Scroll=zoom  Cmd/Ctrl+drag=pan", TEXT_DIM)
        self.font_sm.render_to(self.screen, (x, hint_y + 14),
                               "Space=play  Arrows=step", TEXT_DIM)

    # -----------------------------------------------------------------------
    # Events
    # -----------------------------------------------------------------------

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode((event.w, event.h),
                                                       pygame.RESIZABLE)
                self._reposition_ui()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Cmd (Mac) or Ctrl + click = pan
                mods = pygame.key.get_mods()
                if mods & (pygame.KMOD_META | pygame.KMOD_CTRL):
                    gx, gy, gw, gh = self._grid_area()
                    if event.pos[0] < gx + gw:
                        self._panning = True
                        self._pan_anchor = event.pos
                else:
                    self._on_mousedown(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if self._panning:
                    self._panning = False
                    self._pan_anchor = None
                else:
                    self._on_mouseup(event.pos)
            elif event.type == pygame.MOUSEMOTION:
                self._on_mousemove(event.pos, event.buttons)
            elif event.type == pygame.KEYDOWN:
                self._on_keydown(event)
            # --- Scroll-wheel zoom ---
            elif event.type == pygame.MOUSEWHEEL:
                self._on_scroll(event)
            # --- Middle-mouse pan ---
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
                gx, gy, gw, gh = self._grid_area()
                if event.pos[0] < gx + gw:
                    self._panning = True
                    self._pan_anchor = event.pos
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 2:
                self._panning = False
                self._pan_anchor = None
        return True

    def _on_scroll(self, event):
        """Zoom towards the mouse pointer."""
        mx, my = pygame.mouse.get_pos()
        gx, gy, gw, gh = self._grid_area()
        if mx > gx + gw:
            return  # mouse is over the panel, ignore
        old_zoom = self.zoom
        factor = 1.15 if event.y > 0 else 1 / 1.15
        self.zoom = max(0.2, min(50.0, self.zoom * factor))
        # Adjust pan so the point under the cursor stays fixed
        # Point relative to grid area centre
        rx = mx - (gx + gw / 2)
        ry = my - (gy + gh / 2)
        scale = self.zoom / old_zoom
        self.pan_x = rx - scale * (rx - self.pan_x)
        self.pan_y = ry - scale * (ry - self.pan_y)

    def _on_keydown(self, event):
        # Text input fields first
        if self.text_input.active:
            self.text_input.handle_key(event)
            return
        # Number inputs (grid size + text font config)
        for inp in self.inputs:
            if inp.active:
                if inp.handle_key(event):
                    self._sync_text_inputs()
                return
        if self.mode == "text":
            for inp in self.text_inputs:
                if inp.active:
                    if inp.handle_key(event):
                        self._sync_text_inputs()
                    return

        # Select mode keys
        if self.mode == "select" and self.sel_state == "floating":
            if event.key == pygame.K_UP:
                self._move_selection(-1, 0); return
            elif event.key == pygame.K_DOWN:
                self._move_selection(1, 0); return
            elif event.key == pygame.K_LEFT:
                self._move_selection(0, -1); return
            elif event.key == pygame.K_RIGHT:
                self._move_selection(0, 1); return
            elif event.key == pygame.K_RETURN:
                self._commit_selection(); return
            elif event.key == pygame.K_ESCAPE:
                self._cancel_selection(); return

        # Select mode — cancel with Escape even when not floating
        if self.mode == "select" and event.key == pygame.K_ESCAPE:
            if self.sel_state == "selecting":
                self._clear_selection()
            return

        # Global keys
        if event.key == pygame.K_SPACE:
            self._toggle_play(self.play_direction)
        elif event.key == pygame.K_RIGHT:
            self._do_step()
        elif event.key == pygame.K_LEFT:
            self._do_back()
        elif event.key == pygame.K_c:
            self._do_clear()
        elif event.key == pygame.K_0:
            # Reset zoom & pan
            self.zoom = 1.0; self.pan_x = 0.0; self.pan_y = 0.0
        elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.zoom = min(50.0, self.zoom * 1.25)
        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.zoom = max(0.2, self.zoom / 1.25)

    def _on_mousedown(self, pos):
        # --- Number inputs (grid size + text font config) ---
        all_inputs = list(self.inputs)
        if self.mode == "text":
            all_inputs += self.text_inputs

        clicked_input = False
        for inp in all_inputs:
            if inp.hit(pos):
                if not inp.active:
                    for other in all_inputs:
                        if other is not inp and other.active:
                            other.deactivate()
                    self.text_input.deactivate()
                    inp.activate()
                clicked_input = True
            else:
                if inp.active:
                    inp.deactivate()

        # Text input field
        if self.mode == "text" and self.text_input.hit(pos):
            if not self.text_input.active:
                for inp in all_inputs:
                    if inp.active:
                        inp.deactivate()
                self.text_input.activate()
            clicked_input = True
        else:
            if self.text_input.active:
                self.text_input.deactivate()

        if clicked_input:
            self._sync_text_inputs()
            return

        # --- Sliders ---
        for s in self.sliders:
            if s.hit(pos):
                s.dragging = True
                s.update_drag(pos[0])
                self._sync_sliders()
                return

        # --- Buttons ---
        if self.btn_load.hit(pos):
            self._do_load_image(); return
        if self.btn_apply.hit(pos):
            self._do_apply_size(); return
        if self.btn_boundary.hit(pos):
            self.wrap = not self.wrap; return
        if self.btn_thick.hit(pos):
            self.thick_lines = not self.thick_lines; return
        if self.btn_play.hit(pos):
            self._toggle_play(1); return
        if self.btn_play_rev.hit(pos):
            self._toggle_play(-1); return
        if self.btn_step.hit(pos):
            self._do_step(); return
        if self.btn_back.hit(pos):
            self._do_back(); return
        if self.btn_clear.hit(pos):
            self._do_clear(); return
        if self.gen0_grid is not None and self.generation > 0 and self.btn_reset.hit(pos):
            self._do_reset(); return
        if self.btn_swap_live.hit(pos):
            self._do_swap_live(); return
        if self.btn_swap_colors.hit(pos):
            self._do_swap_colors(); return
        if self.btn_rotate.hit(pos):
            self._do_img_rotate(); return
        if self.btn_reapply.hit(pos):
            self._do_reapply_image(); return

        # Mode selector buttons
        if self.btn_mode_draw.hit(pos):
            self._set_mode("draw"); return
        if self.btn_mode_text.hit(pos):
            self._set_mode("text"); return
        if self.btn_mode_select.hit(pos):
            self._set_mode("select"); return

        # --- Grid interactions based on mode ---
        p = self._pixel_at(*pos)
        if p is None:
            # Clicked outside grid — commit floating selection if any
            if self.mode == "select" and self.sel_state == "floating":
                self._commit_selection()
            return

        if self.mode == "draw":
            self._cancel_pending()
            r, c = p
            self.draw_value = 0 if self.grid[r][c] else 1
            self.grid[r][c] = self.draw_value
            self.drawing = True

        elif self.mode == "text":
            # Stamp text at cursor position
            if self.text_input.text:
                self._stamp_text(p[0], p[1])

        elif self.mode == "select":
            r, c = p
            if self.sel_state == "floating":
                # Check if click is inside the floating selection
                or_, oc = self.sel_offset
                if (or_ <= r < or_ + self.sel_data_rows and
                        oc <= c < oc + self.sel_data_cols):
                    # Start dragging the floating selection
                    self.sel_dragging = True
                    self.sel_drag_anchor = (r - or_, c - oc)
                else:
                    # Click outside — commit and start new selection
                    self._commit_selection()
                    self.sel_start = (r, c)
                    self.sel_end = (r, c)
                    self.sel_state = "selecting"
            else:
                # Start rubber-band selection
                self.sel_start = (r, c)
                self.sel_end = (r, c)
                self.sel_state = "selecting"

    def _on_mouseup(self, pos):
        self.drawing = False
        for s in self.sliders:
            s.dragging = False

        if self.mode == "select":
            if self.sel_state == "selecting":
                # Finish rubber-band — capture cells
                rect = self._sel_rect()
                if rect and (rect[2] > rect[0] or rect[3] > rect[1]):
                    self._capture_selection()
                else:
                    # Too small (single cell) — still capture it
                    if rect:
                        self._capture_selection()
                    else:
                        self._clear_selection()
            if self.sel_dragging:
                self.sel_dragging = False
                self.sel_drag_anchor = None

    def _on_mousemove(self, pos, buttons):
        # Middle-mouse panning
        if self._panning and self._pan_anchor:
            dx = pos[0] - self._pan_anchor[0]
            dy = pos[1] - self._pan_anchor[1]
            self.pan_x += dx
            self.pan_y += dy
            self._pan_anchor = pos
            return

        for b in self.buttons:
            b.update_hover(pos)
        for s in self.sliders:
            if s.dragging:
                s.update_drag(pos[0])
                self._sync_sliders()
        if self.mode == "draw":
            if self.drawing and buttons[0]:
                p = self._pixel_at(*pos)
                if p:
                    r, c = p
                    self.grid[r][c] = self.draw_value

        elif self.mode == "text":
            # Update text ghost preview position
            p = self._pixel_at(*pos)
            self.text_cursor_grid = p

        elif self.mode == "select":
            if self.sel_state == "selecting" and buttons[0]:
                p = self._pixel_at(*pos)
                if p:
                    self.sel_end = p
            elif self.sel_state == "floating" and self.sel_dragging and buttons[0]:
                p = self._pixel_at(*pos)
                if p and self.sel_drag_anchor:
                    dr, dc = self.sel_drag_anchor
                    self.sel_offset = (p[0] - dr, p[1] - dc)

    def _sync_sliders(self):
        self.speed = self.slider_speed.value
        self.threshold = self.slider_thresh.value
        self.img_contrast = self.slider_contrast.value

    def _sync_text_inputs(self):
        self.text_char_width = self.input_char_width.value
        self.text_char_height = self.input_char_height.value
        self.text_spacing = self.input_text_spacing.value
        self.text_space_width = self.input_space_width.value

    # -----------------------------------------------------------------------
    # Play / direction
    # -----------------------------------------------------------------------

    def _toggle_play(self, direction):
        """Toggle play in the given direction (+1 forward, -1 backward).

        If already playing in that direction, pause.
        If playing in the opposite direction, switch direction.
        If paused, start playing in the given direction.
        """
        if self.running and self.play_direction == direction:
            self.running = False
        else:
            self.play_direction = direction
            self.running = True
        self.accum_ms = 0.0

    # -----------------------------------------------------------------------
    # Mode switching
    # -----------------------------------------------------------------------

    def _set_mode(self, new_mode):
        if new_mode == self.mode:
            return
        # Auto-commit floating selection when leaving select mode
        if self.mode == "select" and self.sel_state == "floating":
            self._commit_selection()
        elif self.mode == "select":
            self._clear_selection()
        self.mode = new_mode
        self.drawing = False
        self.text_cursor_grid = None
        self._build_ui()  # reposition controls for new mode

    # -----------------------------------------------------------------------
    # Async computation helpers
    # -----------------------------------------------------------------------

    def _cancel_pending(self):
        """Cancel any in-progress background computation."""
        if self._pending_future is not None:
            self._pending_future.cancel()
            self._pending_future = None

    def _check_future(self):
        """Poll the background future. If done, apply the result."""
        if self._pending_future is None:
            return
        if self._pending_future.done():
            try:
                result = self._pending_future.result()
                self._bytes_to_grid(result)
                self.generation = self._pending_gen
            except Exception:
                pass  # computation was cancelled or failed
            self._pending_future = None

    @property
    def _is_computing(self):
        return self._pending_future is not None

    # -----------------------------------------------------------------------
    # History (undo support)
    # -----------------------------------------------------------------------

    def _push_history(self):
        self.history.append((copy_grid(self.grid), self.generation))
        if len(self.history) > HISTORY_MAX:
            self.history.pop(0)

    # -----------------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------------

    def _save_gen0(self):
        """Snapshot the current grid as generation 0 (if not already saved)."""
        if self.gen0_grid is None or self.generation == 0:
            self.gen0_grid = copy_grid(self.grid)
            self.gen0_rows = self.grid_rows
            self.gen0_cols = self.grid_cols

    def _do_reset(self):
        """Reset to the saved generation 0 state."""
        if self.gen0_grid is None:
            return
        self._cancel_pending()
        self.grid_rows = self.gen0_rows
        self.grid_cols = self.gen0_cols
        self.grid = copy_grid(self.gen0_grid)
        self.input_width.value = self.gen0_cols
        self.input_width.text = str(self.gen0_cols)
        self.input_height.value = self.gen0_rows
        self.input_height.text = str(self.gen0_rows)
        self.generation = 0
        self.history.clear()

    def _do_step(self):
        """Request a single step. Async for large grids, sync for small."""
        if self._is_computing:
            return  # already working on one

        # Save gen 0 on the very first step
        self._save_gen0()
        self._push_history()
        cells = self.grid_rows * self.grid_cols

        if cells <= ASYNC_THRESHOLD:
            # Small grid: compute inline (instant)
            self.grid = step_grid(self.grid, wrap=self.wrap)
            self.generation += 1
        else:
            # Large grid: offload to worker process
            data = self._grid_to_bytes()
            self._pending_gen = self.generation + 1
            self._pending_future = self._pool.submit(
                _step_worker, data, self.grid_rows, self.grid_cols, self.wrap)

    def _do_back(self):
        if not self.history:
            return
        self._cancel_pending()
        self.grid, self.generation = self.history.pop()

    def _do_clear(self):
        self._cancel_pending()
        self.grid = make_grid(self.grid_rows, self.grid_cols)
        self.generation = 0
        self.history.clear()
        self.gen0_grid = None  # nothing to reset to

    def _do_apply_size(self):
        self._cancel_pending()
        new_cols = max(4, min(512, self.input_width.value))
        new_rows = max(4, min(512, self.input_height.value))
        self.input_width.value = new_cols
        self.input_height.value = new_rows
        self.input_width.text = str(new_cols)
        self.input_height.text = str(new_rows)
        if new_cols == self.grid_cols and new_rows == self.grid_rows:
            return
        old = self.grid
        old_rows, old_cols = self.grid_rows, self.grid_cols
        self.grid_rows = new_rows
        self.grid_cols = new_cols
        self.grid = make_grid(new_rows, new_cols)
        for r in range(min(old_rows, new_rows)):
            for c in range(min(old_cols, new_cols)):
                self.grid[r][c] = old[r][c]
        self.generation = 0
        self.history.clear()

    # -----------------------------------------------------------------------
    # Swap operations
    # -----------------------------------------------------------------------

    def _do_swap_live(self):
        self._cancel_pending()
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                self.grid[r][c] = 1 - self.grid[r][c]
        self.alive_is_light = not self.alive_is_light

    def _do_swap_colors(self):
        self.alive_is_light = not self.alive_is_light

    # -----------------------------------------------------------------------
    # Text stamping
    # -----------------------------------------------------------------------

    def _stamp_text(self, row, col):
        """Place the current text onto the grid at (row, col)."""
        text = self.text_input.text
        if not text:
            return
        pixels = self._get_text_pixels(text)
        for dr, dc in pixels:
            gr = row + dr
            gc = col + dc
            if 0 <= gr < self.grid_rows and 0 <= gc < self.grid_cols:
                self.grid[gr][gc] = 1

    # -----------------------------------------------------------------------
    # Image loading & adjustments
    # -----------------------------------------------------------------------

    def _do_load_image(self):
        path = ""
        try:
            import subprocess
            result = subprocess.run(
                ["osascript", "-e",
                 'POSIX path of (choose file of type {"public.image"} '
                 'with prompt "Select an image")'],
                capture_output=True, text=True, timeout=60)
            path = result.stdout.strip()
        except Exception:
            pass

        if not path or not os.path.isfile(path):
            return

        try:
            self.source_image = Image.open(path)
        except Exception as e:
            print(f"Failed to open image: {e}")
            return

        self.img_rotation = 0
        self.dark_is_alive = True
        self.img_contrast = 1.0
        self.slider_contrast.value = 1.0

        self._apply_image_to_grid()

    def _process_source_image(self):
        if self.source_image is None:
            return None
        img = self.source_image.copy()

        if self.img_rotation:
            img = img.rotate(-self.img_rotation, expand=True)

        if abs(self.img_contrast - 1.0) > 0.01:
            img = ImageEnhance.Contrast(img).enhance(self.img_contrast)

        img = img.convert("L")

        w = max(4, min(512, self.input_width.value))
        h = max(4, min(512, self.input_height.value))
        img = img.resize((w, h), Image.LANCZOS)
        return img

    def _apply_image_to_grid(self):
        self._cancel_pending()
        img = self._process_source_image()
        if img is None:
            return
        w, h = img.size
        self.grid_cols = w
        self.grid_rows = h
        self.input_width.value = w
        self.input_width.text = str(w)
        self.input_height.value = h
        self.input_height.text = str(h)
        self.grid = make_grid(h, w)
        thresh = self.threshold
        for r in range(h):
            for c in range(w):
                px = img.getpixel((c, r))
                if self.dark_is_alive:
                    self.grid[r][c] = 1 if px < thresh else 0
                else:
                    self.grid[r][c] = 1 if px >= thresh else 0
        self.generation = 0
        self.history.clear()
        self.gen0_grid = copy_grid(self.grid)
        self.gen0_rows = self.grid_rows
        self.gen0_cols = self.grid_cols

    def _do_img_rotate(self):
        if self.source_image is None:
            return
        self.img_rotation = (self.img_rotation + 90) % 360
        self._apply_image_to_grid()

    def _do_reapply_image(self):
        if self.source_image is None:
            return
        self._apply_image_to_grid()

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self):
        running_app = True
        while running_app:
            dt = self.clock.tick(60)

            running_app = self._handle_events()

            # Check if background computation finished
            self._check_future()

            # Auto-advance when playing (forward or backward)
            if self.running and self.speed > 0 and not self._is_computing:
                interval = 1000.0 / self.speed
                self.accum_ms += dt
                if self.accum_ms >= interval:
                    self.accum_ms = 0.0
                    if self.play_direction == 1:
                        self._do_step()
                    else:
                        if self.history:
                            self._do_back()
                        else:
                            self.running = False  # nothing left to rewind

            self._draw()

        # Clean up worker pool
        self._cancel_pending()
        self._pool.shutdown(wait=False)
        pygame.quit()


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    app = LifeApp()
    app.run()


if __name__ == "__main__":
    main()
