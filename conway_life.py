#!/opt/homebrew/bin/python3.12
"""Conway's Game of Life with image import, pixel editing, and adjustable speed.

Large-grid computation runs asynchronously in a worker process so the UI
stays responsive at 60 fps regardless of grid size.
"""

import pygame
import pygame.freetype
from PIL import Image, ImageEnhance
import random
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

        self.btn_boundary = Button((px, y, pw, bh),
                                   "Boundary: Wrap" if self.wrap else "Boundary: Fixed",
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

        third = (pw - gap * 2) // 3
        self.btn_back = Button((px, y, third, bh), "< Back", BTN_BG, BTN_HOVER)
        self.btn_play = Button((px + third + gap, y, third, bh), "Play", BTN_GREEN, BTN_GREEN_H)
        self.btn_step = Button((px + (third + gap) * 2, y, third, bh), "Step >", BTN_BG, BTN_HOVER)
        y += bh + gap + 2

        self.slider_speed = Slider((px, y + 18, pw, 14), 1, 60, self.speed, 1, "Speed (gen/s)")
        y += 46

        self._sep2_y = y; y += 10

        # --- Edit ---
        self._edit_header_y = y; y += 22

        half = (pw - gap) // 2
        self.btn_swap_live = Button((px, y, half, bh), "Swap Live", BTN_BG, BTN_HOVER)
        self.btn_swap_colors = Button((px + half + gap, y, half, bh), "Swap Clr", BTN_BG, BTN_HOVER)
        y += bh + gap

        self._legend_y = y
        y += 18

        self.btn_clear = Button((px, y, pw, bh), "Clear", BTN_RED, BTN_RED_H)
        y += bh + gap
        self.btn_random = Button((px, y, pw, bh), "Random Fill", BTN_BG, BTN_HOVER)
        y += bh + gap
        self.btn_invert = Button((px, y, pw, bh), "Invert Grid", BTN_BG, BTN_HOVER)
        y += bh + gap

        # Reset button — positioned here but only shown when generation > 0
        self._reset_btn_y = y
        self.btn_reset = Button((px, y, pw, bh), "Reset to Gen 0", BTN_BLUE, BTN_BLUE_H)

        self.buttons = [self.btn_load, self.btn_apply, self.btn_play,
                        self.btn_step, self.btn_back, self.btn_clear,
                        self.btn_random, self.btn_invert,
                        self.btn_rotate, self.btn_reapply,
                        self.btn_swap_live, self.btn_swap_colors,
                        self.btn_boundary]
        self.sliders = [self.slider_thresh, self.slider_speed,
                        self.slider_contrast]
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
        cell = min(gw / cols, gh / rows)
        ox = gx + (gw - cell * cols) / 2
        oy = gy + (gh - cell * rows) / 2
        return cell, ox, oy

    def _pixel_at(self, mx, my):
        cell, ox, oy = self._cell_metrics()
        c = int((mx - ox) // cell)
        r = int((my - oy) // cell)
        if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
            return r, c
        return None

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
            for r in range(rows + 1):
                y = int(oy + r * cell)
                pygame.draw.line(self.screen, GRID_LINE,
                                 (int(ox), y), (int(ox + cols * cell), y))
            for c in range(cols + 1):
                x = int(ox + c * cell)
                pygame.draw.line(self.screen, GRID_LINE,
                                 (x, int(oy)), (x, int(oy + rows * cell)))

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

        # Update play button
        self.btn_play.text = "Pause" if self.running else "Play"
        self.btn_play.color = BTN_AMBER if self.running else BTN_GREEN
        self.btn_play.hover_color = BTN_AMBER_H if self.running else BTN_GREEN_H

        # Update boundary button text
        self.btn_boundary.text = "Boundary: Wrap" if self.wrap else "Boundary: Fixed"

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

        for b in self.buttons:
            b.draw(self.screen, self.font)
        for s in self.sliders:
            s.draw(self.screen, self.font_sm)
        for inp in self.inputs:
            inp.draw(self.screen, self.font_sm)

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

        # Hint — anchor below reset button if visible, otherwise below invert
        if show_reset:
            hint_y = self.btn_reset.rect.bottom + 10
        else:
            hint_y = self.btn_invert.rect.bottom + 10
        self.font_sm.render_to(self.screen, (x, hint_y),
                               "Click/drag grid to draw", TEXT_DIM)
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
                self._on_mousedown(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._on_mouseup(event.pos)
            elif event.type == pygame.MOUSEMOTION:
                self._on_mousemove(event.pos, event.buttons)
            elif event.type == pygame.KEYDOWN:
                handled = False
                for inp in self.inputs:
                    if inp.active:
                        handled = inp.handle_key(event)
                        break
                if not handled:
                    if event.key == pygame.K_SPACE:
                        self.running = not self.running
                    elif event.key == pygame.K_RIGHT:
                        self._do_step()
                    elif event.key == pygame.K_LEFT:
                        self._do_back()
                    elif event.key == pygame.K_c:
                        self._do_clear()
                    elif event.key == pygame.K_r:
                        self._do_random()
        return True

    def _on_mousedown(self, pos):
        clicked_input = False
        for inp in self.inputs:
            if inp.hit(pos):
                if not inp.active:
                    for other in self.inputs:
                        if other is not inp and other.active:
                            other.deactivate()
                    inp.activate()
                clicked_input = True
            else:
                if inp.active:
                    inp.deactivate()

        if clicked_input:
            return

        for s in self.sliders:
            if s.hit(pos):
                s.dragging = True
                s.update_drag(pos[0])
                self._sync_sliders()
                return

        if self.btn_load.hit(pos):
            self._do_load_image(); return
        if self.btn_apply.hit(pos):
            self._do_apply_size(); return
        if self.btn_boundary.hit(pos):
            self.wrap = not self.wrap; return
        if self.btn_play.hit(pos):
            self.running = not self.running; return
        if self.btn_step.hit(pos):
            self._do_step(); return
        if self.btn_back.hit(pos):
            self._do_back(); return
        if self.btn_clear.hit(pos):
            self._do_clear(); return
        if self.btn_random.hit(pos):
            self._do_random(); return
        if self.btn_invert.hit(pos):
            self._do_invert(); return
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

        # Grid drawing — cancel any pending computation since grid changed
        p = self._pixel_at(*pos)
        if p:
            self._cancel_pending()
            r, c = p
            self.draw_value = 0 if self.grid[r][c] else 1
            self.grid[r][c] = self.draw_value
            self.drawing = True

    def _on_mouseup(self, pos):
        self.drawing = False
        for s in self.sliders:
            s.dragging = False

    def _on_mousemove(self, pos, buttons):
        for b in self.buttons:
            b.update_hover(pos)
        for s in self.sliders:
            if s.dragging:
                s.update_drag(pos[0])
                self._sync_sliders()
        if self.drawing and buttons[0]:
            p = self._pixel_at(*pos)
            if p:
                r, c = p
                self.grid[r][c] = self.draw_value

    def _sync_sliders(self):
        self.speed = self.slider_speed.value
        self.threshold = self.slider_thresh.value
        self.img_contrast = self.slider_contrast.value

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

    def _do_random(self):
        self._cancel_pending()
        self.grid = [[random.randint(0, 1) for _ in range(self.grid_cols)]
                     for _ in range(self.grid_rows)]
        self.generation = 0
        self.history.clear()
        self.gen0_grid = copy_grid(self.grid)
        self.gen0_rows = self.grid_rows
        self.gen0_cols = self.grid_cols

    def _do_invert(self):
        self._cancel_pending()
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                self.grid[r][c] = 1 - self.grid[r][c]

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

            # Auto-advance when playing
            if self.running and self.speed > 0 and not self._is_computing:
                interval = 1000.0 / self.speed
                self.accum_ms += dt
                # Cap accumulator so we don't queue a burst of steps
                if self.accum_ms >= interval:
                    self.accum_ms = 0.0
                    self._do_step()

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
