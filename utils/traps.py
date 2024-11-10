from typing import List

import numpy as np
import supervision as sv

# TODO: incorporate perspective correction/surfaces

def create_sliding_zones(display_size=(1280,720), shift_size=100, zone_width=200):
    w, h = display_size
    n_x_shifts = (w-zone_width)//shift_size
    n_y_shifts = (h-zone_width)//shift_size
    zones = []
    for x in range(n_x_shifts):
        for y in range(n_y_shifts):
            x1 = x*shift_size
            y1 = y*shift_size
            x2 = x1+zone_width
            y2 = y1+zone_width
            polygon = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])
            zones.append(sv.PolygonZone(polygon=polygon))
    return zones


def get_trap_placements(detections, zones:List[sv.PolygonZone]):
    for zone in zones:
        zone.trigger(detections)
    return [zone for zone in sorted(zones, key=lambda x: x.current_count, reverse=True) if zone.current_count > 0]

def get_trap_annotators(zones:List[sv.PolygonZone], n_traps=3):
    return [sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.GREEN, thickness=4) for zone in zones[:n_traps]]