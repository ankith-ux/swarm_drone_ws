#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
import json
import math
import time
import pygame
import threading

# Import the scenario loader to get ground truth map and POIs
from protoplasm_gazebo.scenario_loader import DisasterScenario, CELL_RUBBLE, CELL_HAZARD, CELL_HOT_DEBRIS, CELL_WIND_NOISE, CELL_CLEAR, CELL_SURVIVOR

# Modern Tactical Colors
C_BG = (15, 18, 22)
C_PANEL = (22, 26, 32)
C_GRID_LINE = (35, 40, 50)
C_TEXT = (200, 210, 220)
C_TEXT_DIM = (100, 110, 120)

C_GCS = (0, 200, 255)         # Cyan
C_DRONE = (0, 255, 180)       # Teal/Green
C_DRONE_RESCUE = (255, 215, 0) # Gold
C_LINK = (0, 100, 150)
C_LINK_GCS = (0, 150, 255)

C_POI_TRUE = (255, 100, 0)    # Orange (Ground Truth)
C_POI_FOUND = (50, 255, 50)   # Neon Green (GCS Confirmed)
C_POI_SWARM = (255, 215, 0)   # Gold (Swarm Known, not relayed to GCS yet)

# Regime colors
C_SENTINEL = (255, 60, 60)    # Red (awaiting verification)
C_VERIFIER = (255, 200, 50)   # Yellow (flying to verify)
C_RELAY = (0, 200, 255)       # Cyan (relaying to GCS)
C_RTB = (255, 140, 0)         # Orange (returning to base)

# Terrain Colors
C_TERRAIN = {
    CELL_CLEAR: (15, 18, 22),
    CELL_RUBBLE: (40, 35, 30),
    CELL_HAZARD: (60, 20, 20),
    CELL_HOT_DEBRIS: (80, 50, 0),
    CELL_WIND_NOISE: (20, 30, 60),
    CELL_SURVIVOR: (15, 18, 22) # Drawn separately
}

def draw_triangle(surface, color, x, y, angle, size=8):
    """Draw a tactical triangle pointing in the direction of travel."""
    p1 = (math.cos(angle) * size, math.sin(angle) * size)
    p2 = (math.cos(angle + 2.5) * size * 0.8, math.sin(angle + 2.5) * size * 0.8)
    p3 = (math.cos(angle - 2.5) * size * 0.8, math.sin(angle - 2.5) * size * 0.8)
    points = [
        (x + p1[0], y - p1[1]), # Invert Y for drawing
        (x + p2[0], y - p2[1]),
        (x + p3[0], y - p3[1])
    ]
    pygame.draw.polygon(surface, color, points)

class DashboardNode(Node):
    def __init__(self):
        super().__init__('protoplasm_dashboard')
        self.scenario = DisasterScenario(name='grand-challenge', cols=50, rows=50, world_size_x=1000.0, world_size_y=1000.0)
        
        self.ground_truth_pois = []
        for r in range(self.scenario.rows):
            for c in range(self.scenario.cols):
                if self.scenario.grid[r * self.scenario.cols + c] == CELL_SURVIVOR:
                    wx, wy = self.scenario.grid_to_world(c, r)
                    self.ground_truth_pois.append((wx, wy, c, r))

        self.drones = {}       
        self.found_pois = set()          # Swarm-known POIs (unverified detections)
        self.verified_pois = set()       # POIs verified by a second drone
        self.gcs_confirmed_pois = set()  # POIs physically relayed to GCS operators
        self.gcs_pos = (-575.0, 0.0)
        
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=50)
        self.sub_radio = self.create_subscription(String, '/swarm/radio', self._on_radio, qos)
        self.get_logger().info("Tactical Dashboard started.")

    def _on_radio(self, msg: String):
        try:
            data = json.loads(msg.data)
            msg_type = data.get('type')
            
            drone_id = data.get('drone_id')
            sender_x = data.get('sender_x')
            sender_y = data.get('sender_y')
            
            if drone_id is not None and sender_x is not None and sender_y is not None:
                heading = 0.0
                if drone_id in self.drones:
                    old_x = self.drones[drone_id]['x']
                    old_y = self.drones[drone_id]['y']
                    if math.hypot(sender_x - old_x, sender_y - old_y) > 0.1:
                        heading = math.atan2(sender_y - old_y, sender_x - old_x)
                    else:
                        heading = self.drones[drone_id]['heading']
                
                # Merge new data with existing state to keep confidence/uncertainty if missing
                state = self.drones.get(drone_id, {
                    'confidence': 0.0,
                    'uncertainty': 1.0,
                    'regime': 'UNKNOWN',
                    'callsign': f"D{drone_id:02d}"
                })
                
                self.drones[drone_id] = {
                    'x': sender_x,
                    'y': sender_y,
                    'heading': heading,
                    'callsign': data.get('callsign', state.get('callsign', f"D{drone_id:02d}")),
                    'regime': data.get('regime', state.get('regime', 'UNKNOWN')),
                    'confidence': data.get('confidence', state.get('confidence', 0.0)),
                    'uncertainty': data.get('uncertainty', state.get('uncertainty', 1.0)),
                    'speed': data.get('speed', state.get('speed', 0.0)),
                    'battery': data.get('battery', state.get('battery', 100.0)),
                    'sensors': data.get('sensors', state.get('sensors', {})),
                    'sentinel_cell': data.get('sentinel_cell'),
                    'sentinel_verified': data.get('sentinel_verified', False),
                    'verify_target': data.get('verify_target'),
                    'verify_for': data.get('verify_for'),
                    'last_seen': time.time()
                }

            if msg_type == 'SURVIVOR_FOUND':
                col = data.get('col')
                row = data.get('row')
                if col is not None and row is not None:
                    self.found_pois.add((col, row))

            elif msg_type == 'VERIFY_CONFIRM':
                col = data.get('col')
                row = data.get('row')
                if col is not None and row is not None:
                    self.verified_pois.add((col, row))
                    self.get_logger().info(
                        f'✅ PEER VERIFIED: POI at ({col},{row}) '
                        f'confirmed by {data.get("callsign", "??")}')

            elif msg_type == 'GCS_RELAY':
                col = data.get('col')
                row = data.get('row')
                if col is not None and row is not None:
                    self.gcs_confirmed_pois.add((col, row))
                    self.get_logger().info(
                        f'✅ GCS CONFIRMED: POI at ({col},{row}) — '
                        f'relayed by {data.get("callsign", "??")} '
                        f'(originally found by {data.get("finder", "??")})')
                    
        except json.JSONDecodeError:
            pass

def main():
    rclpy.init()
    node = DashboardNode()
    
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    pygame.init()
    
    # Initialize with default size, but resizable
    window_w, window_h = 1200, 800
    screen = pygame.display.set_mode((window_w, window_h), pygame.RESIZABLE)
    pygame.display.set_caption("Protoplasm Tactical Dashboard")
    clock = pygame.time.Clock()
    
    font_sm = pygame.font.SysFont("courier", 12)
    font_md = pygame.font.SysFont("courier", 16, bold=True)
    font_lg = pygame.font.SysFont("courier", 24, bold=True)

    running = True
    start_time = time.time()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                window_w, window_h = event.w, event.h
                screen = pygame.display.set_mode((window_w, window_h), pygame.RESIZABLE)

        # Dynamic layout calculations
        # Sidebar takes 350px, minimum 250px. Map takes the rest.
        sidebar_w = max(250, min(400, int(window_w * 0.3)))
        map_size = min(window_w - sidebar_w, window_h)
        
        # Center the map in its allocated area if window is wider
        map_offset_x = (window_w - sidebar_w - map_size) // 2
        map_offset_y = (window_h - map_size) // 2

        arena_size = 1200.0
        scale = map_size / arena_size
        center_px = map_size / 2

        def world_to_screen(x, y):
            sx = int(center_px + (x * scale)) + map_offset_x
            sy = int(center_px - (y * scale)) + map_offset_y
            return sx, sy

        now = time.time()
        screen.fill(C_BG)

        # ─── 1. DRAW GAZEBO TERRAIN MAP ───
        cell_size_x = node.scenario.world_size_x / node.scenario.cols
        cell_size_y = node.scenario.world_size_y / node.scenario.rows
        cell_w_px = cell_size_x * scale
        cell_h_px = cell_size_y * scale
        
        for r in range(node.scenario.rows):
            for c in range(node.scenario.cols):
                cell_type = node.scenario.grid[r * node.scenario.cols + c]
                if cell_type != CELL_CLEAR and cell_type != CELL_SURVIVOR:
                    color = C_TERRAIN.get(cell_type, C_BG)
                    wx, wy = node.scenario.grid_to_world(c, r)
                    wx_top_left = wx - (cell_size_x / 2)
                    wy_top_left = wy + (cell_size_y / 2)
                    sx, sy = world_to_screen(wx_top_left, wy_top_left)
                    pygame.draw.rect(screen, color, (sx, sy, int(cell_w_px)+1, int(cell_h_px)+1))

        # ─── 2. DRAW ARENA GRID ───
        for i in range(11):
            coord = -500.0 + (i * 100.0)
            x1, y1 = world_to_screen(coord, 500.0)
            x2, y2 = world_to_screen(coord, -500.0)
            pygame.draw.line(screen, C_GRID_LINE, (x1, y1), (x2, y2), 1)
            x3, y3 = world_to_screen(-500.0, coord)
            x4, y4 = world_to_screen(500.0, coord)
            pygame.draw.line(screen, C_GRID_LINE, (x3, y3), (x4, y4), 1)
        
        cx, cy = world_to_screen(0, 0)
        pygame.draw.line(screen, (80, 90, 100), (cx, map_offset_y), (cx, map_offset_y + map_size), 1)
        pygame.draw.line(screen, (80, 90, 100), (map_offset_x, cy), (map_offset_x + map_size, cy), 1)
        
        gcs_sx, gcs_sy = world_to_screen(node.gcs_pos[0], node.gcs_pos[1])
        pygame.draw.circle(screen, C_GRID_LINE, (gcs_sx, gcs_sy), int(100 * scale), 1)

        # ─── 3. DRAW GROUND TRUTH POIS ───
        pulse = (math.sin(now * 5) + 1) / 2
        for (wx, wy, c, r) in node.ground_truth_pois:
            sx, sy = world_to_screen(wx, wy)
            if (c, r) in node.gcs_confirmed_pois:
                # GCS CONFIRMED: Solid green with pulsing ring
                radius = 8 + (pulse * 4)
                pygame.draw.circle(screen, C_POI_FOUND, (sx, sy), int(radius), 2)
                pygame.draw.circle(screen, C_POI_FOUND, (sx, sy), 3)
            elif (c, r) in node.verified_pois:
                # PEER VERIFIED: Gold pulsing ring (waiting for GCS relay)
                radius = 7 + (pulse * 3)
                pygame.draw.circle(screen, C_POI_SWARM, (sx, sy), int(radius), 2)
                pygame.draw.circle(screen, C_POI_SWARM, (sx, sy), 3)
            elif (c, r) in node.found_pois:
                # UNVERIFIED: Orange pulsing ring (awaiting peer verification)
                radius = 5 + (pulse * 2)
                pygame.draw.circle(screen, C_POI_TRUE, (sx, sy), int(radius), 1)
            else:
                # UNKNOWN: Small dim dot (ground truth only)
                pygame.draw.circle(screen, (80, 50, 30), (sx, sy), 2)

        # ─── 4. DRAW DRONES & COMM LINKS ───
        active_drones = {d_id: d for d_id, d in node.drones.items() if now - d['last_seen'] < 5.0}
        drone_list = list(active_drones.values())
        
        links_active = 0
        for i in range(len(drone_list)):
            d1 = drone_list[i]
            sx1, sy1 = world_to_screen(d1['x'], d1['y'])
            if math.hypot(d1['x'] - node.gcs_pos[0], d1['y'] - node.gcs_pos[1]) <= 100.0:
                pygame.draw.aaline(screen, C_LINK_GCS, (sx1, sy1), (gcs_sx, gcs_sy))
                links_active += 1
            for j in range(i + 1, len(drone_list)):
                d2 = drone_list[j]
                if math.hypot(d1['x'] - d2['x'], d1['y'] - d2['y']) <= 100.0:
                    sx2, sy2 = world_to_screen(d2['x'], d2['y'])
                    pygame.draw.aaline(screen, C_LINK, (sx1, sy1), (sx2, sy2))
                    links_active += 1

        for d_id, d in active_drones.items():
            sx, sy = world_to_screen(d['x'], d['y'])
            regime = d.get('regime', 'SPREAD')
            if regime == 'SENTINEL':
                color = C_SENTINEL
            elif regime == 'VERIFIER':
                color = C_VERIFIER
            elif regime == 'RELAY':
                color = C_RELAY
            elif regime == 'RTB':
                color = C_RTB
            elif regime in ['SOLIDIFY', 'RESCUED']:
                color = C_DRONE_RESCUE
            else:
                color = C_DRONE
            draw_triangle(screen, color, sx, sy, d['heading'], size=10)
            pygame.draw.circle(screen, (10, 40, 50), (sx, sy), int(100 * scale), 1)
            lbl = font_sm.render(d['callsign'], True, C_TEXT)
            screen.blit(lbl, (sx + 8, sy - 14))
            
        pygame.draw.rect(screen, C_GCS, (gcs_sx - 6, gcs_sy - 6, 12, 12))
        screen.blit(font_sm.render("GCS", True, C_GCS), (gcs_sx + 10, gcs_sy - 6))

        # ─── 5. DRAW RIGHT SIDEBAR (DYNAMIC) ───
        sb_x = window_w - sidebar_w
        sidebar_rect = pygame.Rect(sb_x, 0, sidebar_w, window_h)
        pygame.draw.rect(screen, C_PANEL, sidebar_rect)
        pygame.draw.line(screen, C_GRID_LINE, (sb_x, 0), (sb_x, window_h), 2)
        
        screen.blit(font_lg.render("PROTOPLASM C2", True, C_GCS), (sb_x + 20, 20))
        screen.blit(font_sm.render("TACTICAL SWARM DASHBOARD", True, C_TEXT_DIM), (sb_x + 20, 50))
        pygame.draw.line(screen, C_GRID_LINE, (sb_x + 20, 70), (window_w - 20, 70), 1)
        
        elapsed = int(now - start_time)
        total_pois = len(node.ground_truth_pois)
        stats = [
            ("MISSION TIME", f"{elapsed//60:02d}:{elapsed%60:02d}"),
            ("ACTIVE DRONES", str(len(active_drones))),
            ("MESH LINKS", str(links_active)),
            ("DETECTED", f"{len(node.found_pois)} / {total_pois}"),
            ("VERIFIED", f"{len(node.verified_pois)} / {total_pois}"),
            ("GCS CONFIRMED", f"{len(node.gcs_confirmed_pois)} / {total_pois}"),
        ]
        
        y_offset = 90
        for label, val in stats:
            screen.blit(font_sm.render(label, True, C_TEXT_DIM), (sb_x + 20, y_offset))
            val_col = C_TEXT
            if label == "DETECTED" and len(node.found_pois) > 0:
                val_col = C_POI_TRUE
            if label == "VERIFIED" and len(node.verified_pois) > 0:
                val_col = C_POI_SWARM
            if label == "GCS CONFIRMED" and len(node.gcs_confirmed_pois) > 0:
                val_col = C_POI_FOUND
            screen.blit(font_md.render(val, True, val_col), (window_w - 100, y_offset))
            y_offset += 30

        pygame.draw.line(screen, C_GRID_LINE, (sb_x + 20, y_offset), (window_w - 20, y_offset), 1)
        y_offset += 10

        # ─── VERIFICATION ACTIVITY PANEL ───
        sentinels = {d_id: d for d_id, d in active_drones.items() if d.get('regime') == 'SENTINEL'}
        verifiers = {d_id: d for d_id, d in active_drones.items() if d.get('regime') == 'VERIFIER'}

        if sentinels or verifiers:
            screen.blit(font_md.render("VERIFY ACTIVITY", True, C_SENTINEL), (sb_x + 20, y_offset))
            y_offset += 20

            for d_id, d in sentinels.items():
                cell = d.get('sentinel_cell')
                verified = d.get('sentinel_verified', False)
                cell_str = f"({cell[0]},{cell[1]})" if cell else "?"
                status = "✅" if verified else "⏳"
                screen.blit(font_sm.render(
                    f"{status} {d['callsign']} → {cell_str}",
                    True, C_SENTINEL if not verified else C_POI_FOUND
                ), (sb_x + 20, y_offset))
                y_offset += 14

            for d_id, d in verifiers.items():
                target = d.get('verify_target')
                target_str = f"({target[0]},{target[1]})" if target else "?"
                # Find the sentinel callsign
                sentinel_id = d.get('verify_for')
                sentinel_name = "?"
                if sentinel_id is not None and sentinel_id in active_drones:
                    sentinel_name = active_drones[sentinel_id]['callsign']
                screen.blit(font_sm.render(
                    f"🔍 {d['callsign']} → {target_str} for {sentinel_name}",
                    True, C_VERIFIER
                ), (sb_x + 20, y_offset))
                y_offset += 14

            y_offset += 5
            pygame.draw.line(screen, C_GRID_LINE, (sb_x + 20, y_offset), (window_w - 20, y_offset), 1)
            y_offset += 10

        # ─── DRONE LIST ───
        screen.blit(font_md.render("DRONE STATUS", True, C_TEXT), (sb_x + 20, y_offset))
        y_offset += 20
        
        for d_id, d in sorted(active_drones.items()):
            # Stop drawing if we'd overflow into the legend area
            if y_offset > window_h - 130:
                remaining = len(active_drones) - len([did for did in active_drones if did <= d_id])
                if remaining > 0:
                    screen.blit(font_sm.render(f"... +{remaining} more", True, C_TEXT_DIM), (sb_x + 20, y_offset))
                break

            regime = d.get('regime', 'SPREAD')
            if regime == 'SENTINEL':
                color = C_SENTINEL
            elif regime == 'VERIFIER':
                color = C_VERIFIER
            elif regime == 'RELAY':
                color = C_RELAY
            elif regime == 'RTB':
                color = C_RTB
            else:
                color = C_DRONE
            conf = min(1.0, max(0.0, d.get('confidence', 0.0)))
            battery = d.get('battery', 100.0)
            
            # Single compact line: Callsign | Regime | Conf bar | Battery
            screen.blit(font_sm.render(f"[{d['callsign']}]", True, color), (sb_x + 20, y_offset))
            screen.blit(font_sm.render(f"{regime[:8]}", True, C_TEXT), (sb_x + 95, y_offset))
            pygame.draw.rect(screen, (40, 40, 40), (sb_x + 170, y_offset + 3, 40, 6))
            pygame.draw.rect(screen, C_POI_FOUND, (sb_x + 170, y_offset + 3, int(40 * conf), 6))
            bat_color = C_POI_FOUND if battery > 20 else (200, 80, 80)
            screen.blit(font_sm.render(f"{battery:.0f}%", True, bat_color), (sb_x + 215, y_offset))
            y_offset += 16

        legend_y = window_h - 120
        pygame.draw.line(screen, C_GRID_LINE, (sb_x + 20, legend_y), (window_w - 20, legend_y), 1)
        screen.blit(font_sm.render("TERRAIN LEGEND", True, C_TEXT_DIM), (sb_x + 20, legend_y + 15))
        
        legend_items = [
            (C_TERRAIN[CELL_RUBBLE], "Rubble"),
            (C_TERRAIN[CELL_HAZARD], "Hazard"),
            (C_TERRAIN[CELL_HOT_DEBRIS], "Hot Debris (False Pos)"),
            (C_TERRAIN[CELL_WIND_NOISE], "Wind Noise (False Pos)")
        ]
        
        ly = legend_y + 40
        for color, name in legend_items:
            pygame.draw.rect(screen, color, (sb_x + 20, ly, 12, 12))
            screen.blit(font_sm.render(name, True, C_TEXT), (sb_x + 40, ly))
            ly += 18

        pygame.display.flip()
        clock.tick(30)

    node.destroy_node()
    rclpy.shutdown()
    pygame.quit()

if __name__ == '__main__':
    main()
