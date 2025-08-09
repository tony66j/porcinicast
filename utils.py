import math
import numpy as np

# ---------- Terreno: pendenza & aspect da piccolo campionamento ----------

def slope_aspect_from_elev_grid(grid, cell_size_m=30.0):
    """
    Calcola pendenza (gradi) e aspect (ottanti cardinali) da una griglia 3x3 o 5x5 di altezze (metri).
    Usa un semplice gradiente centrale su 3x3 interno.
    """
    g = np.array(grid, dtype=float)
    if g.ndim != 2 or min(g.shape) < 3:
        raise ValueError("grid must be 2D with size >= 3x3")
    # usa solamente il 3x3 centrale
    if g.shape[0] > 3 or g.shape[1] > 3:
        r0 = (g.shape[0] - 3)//2
        c0 = (g.shape[1] - 3)//2
        g = g[r0:r0+3, c0:c0+3]

    # Horn 1981 kernel approx
    dzdx = ((g[0,2] + 2*g[1,2] + g[2,2]) - (g[0,0] + 2*g[1,0] + g[2,0])) / (8*cell_size_m)
    dzdy = ((g[2,0] + 2*g[2,1] + g[2,2]) - (g[0,0] + 2*g[0,1] + g[0,2])) / (8*cell_size_m)

    slope_rad = math.atan(math.hypot(dzdx, dzdy))
    slope_deg = math.degrees(slope_rad)

    aspect_rad = math.atan2(dzdy, -dzdx)  # convenzione geospaziale
    aspect_deg = (math.degrees(aspect_rad) + 360.0) % 360.0
    aspect_octant = deg_to_octant(aspect_deg)
    return slope_deg, aspect_deg, aspect_octant

def deg_to_octant(deg):
    dirs = ['N','NE','E','SE','S','SW','W','NW']
    idx = int((deg + 22.5) // 45) % 8
    return dirs[idx]


# ---------- Scoring micologico ----------

def normalized_precip_14mm(p14_mm):
    # Normalizza 0..1 con soglie ragionevoli: 20..80 mm
    return max(0.0, min(1.0, (p14_mm - 20.0) / 60.0))

def normalized_temp_window(tmean7_c, t_opt=14.0, t_width=8.0):
    # Triangolare rispetto all'ottimo
    return max(0.0, min(1.0, 1.0 - abs(tmean7_c - t_opt)/t_width))

def normalized_altitude_window(alt_m, alt_opt=950.0, alt_width=500.0):
    return max(0.0, min(1.0, 1.0 - abs(alt_m - alt_opt)/alt_width))

def aspect_bonus(aspect_octant, season='warm'):
    if season == 'warm':
        if aspect_octant in ('N','NE','NW'): return 0.8
        if aspect_octant in ('E','W'): return 0.6
        return 0.5
    else:
        # stagione fredda: bonus leggermente più neutro
        if aspect_octant in ('S','SE','SW'): return 0.7
        return 0.6

def seasonal_altitude_adjust(lat, month):
    """
    Aggiusta l'altitudine ottimale per stagione/latitudine (euristica).
    """
    # estate (lug-ago): alt opt più alto; autunno (set-ott): medio; primavera: più basso
    if month in (7,8):
        return 1100.0
    if month in (9,10):
        return 950.0
    if month in (5,6):
        return 800.0
    return 900.0

def forest_compatibility_multiplier(forest_label):
    # compatibilità micorrizica: Fagus/Quercus/Castanea/Abies/Picea => alta
    if forest_label in ('Fagus sylvatica','Quercus spp.','Castanea sativa','Pinus/Abies/Picea'):
        return 1.1
    return 1.0

def composite_score(p14_mm, tmean7_c, alt_m, aspect_octant, forest_label, month, lat):
    p14n = normalized_precip_14mm(p14_mm)
    tn = normalized_temp_window(tmean7_c, t_opt=14.0, t_width=8.0)

    alt_opt = seasonal_altitude_adjust(lat, month)
    zn = normalized_altitude_window(alt_m, alt_opt=alt_opt, alt_width=500.0)

    asp = aspect_bonus(aspect_octant, season='warm' if month in (5,6,7,8,9) else 'cold')

    # placeholder suolo/NDVI in attesa di dati
    soil = 0.6
    ndvi = 0.6
    compat = forest_compatibility_multiplier(forest_label)

    weights = {'P':0.25, 'Pd':0.10, 'T':0.15, 'SM':0.15, 'Asp':0.07, 'Bos':0.15, 'Z':0.08, 'V':0.05}
    Pdist = 0.7  # TODO: ricalcolare da serie giornaliera reale

    score = 100.0*(weights['P']*p14n + weights['Pd']*Pdist + weights['T']*tn +
                   weights['SM']*soil + weights['Asp']*asp + weights['Bos']*(compat) +
                   weights['Z']*zn + weights['V']*ndvi)
    # clamp
    return max(0.0, min(100.0, score)), {
        'p14n': p14n, 'tn': tn, 'zn': zn, 'asp': asp,
        'compat': compat, 'soil': soil, 'ndvi': ndvi,
        'alt_opt': alt_opt
    }


def best_window_3day(avg_scores):
    """
    Trova la finestra di 3 giorni consecutivi con media più alta.
    Ritorna (start_idx, end_idx, media).
    """
    n = len(avg_scores)
    best_s, best_e, best_m = 0, min(2, n-1), -1
    for i in range(0, n-2):
        m = (avg_scores[i] + avg_scores[i+1] + avg_scores[i+2]) / 3.0
        if m > best_m:
            best_m, best_s, best_e = m, i, i+2
    return best_s, best_e, int(round(best_m))
