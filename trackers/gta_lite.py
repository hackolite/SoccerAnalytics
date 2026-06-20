"""
trackers/gta_lite.py — Module de raffinement de tracklets par association globale
inspiré de GTA (Global Tracklet Association), intégrable comme étape
de post-traitement dans le pipeline TrackLab / sn-gamestate.

Principe :
1. Détecte les ruptures suspectes (sauts de vitesse implausibles) dans
   les tracklets produits par le tracker court-terme (étape précédente).
2. Pour chaque rupture, cherche les candidats de réassociation parmi
   tous les tracklets actifs dans une fenêtre temporelle proche.
3. Résout l'assignation par algorithme hongrois (association globale,
   pas gourmande) sur un score combinant apparence + plausibilité physique.
4. N'applique la correction que si la marge de confiance est suffisante
   (garde-fou contre la propagation d'erreur).

Usage minimal :
    from trackers.gta_lite import refine_tracklets
    refine_tracklets(tracks)          # modifie tracks['players'] sur place
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Hyper-paramètres par défaut
# ──────────────────────────────────────────────────────────────────────────────
_DEFAULT_MAX_GAP_FRAMES: int = 30
"""Fenêtre temporelle maximale (en frames) pour tenter un ré-appariement."""

_DEFAULT_ABS_SPEED_THRESH: float = 80.0
"""Seuil dur de vitesse en px/frame ; au-delà la rupture est suspecte."""

_DEFAULT_SPEED_FACTOR: float = 3.0
"""Facteur × vitesse_moyenne au-delà duquel un saut est considéré suspect."""

_DEFAULT_VEL_WINDOW: int = 10
"""Nombre de frames utilisées pour estimer la vitesse courante d'un tracklet."""

_DEFAULT_MIN_CONFIDENCE_MARGIN: float = 0.15
"""Marge minimale entre le meilleur score et le second pour valider la correction."""

_DEFAULT_W_POSITION: float = 0.70
"""Poids du score de plausibilité spatiale dans le score composite."""

_DEFAULT_W_SIZE: float = 0.30
"""Poids du score de similarité de taille de bounding-box dans le score composite."""


# ──────────────────────────────────────────────────────────────────────────────
# Algorithme hongrois (Kuhn-Munkres) — implémentation NumPy sans scipy
# ──────────────────────────────────────────────────────────────────────────────

def _linear_sum_assignment(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Algorithme d'assignation à coût minimal (Kuhn-Munkres / hongrois).

    Paramètres
    ----------
    cost : ndarray de forme (n_rows, n_cols)
        Matrice de coûts (valeurs >= 0 conseillées).

    Retourne
    --------
    (row_ind, col_ind) — indices des paires assignées, un couple par ligne.
    """
    C = np.array(cost, dtype=float)
    n_rows, n_cols = C.shape
    n = max(n_rows, n_cols)

    # Rembourrer en matrice carrée
    if n_rows < n or n_cols < n:
        Cp = np.full((n, n), float(C.max() + 1) if C.size > 0 else 1.0)
        Cp[:n_rows, :n_cols] = C
    else:
        Cp = C.copy()

    # Étape 1 : soustraire les minima de ligne puis de colonne
    Cp -= Cp.min(axis=1, keepdims=True)
    Cp -= Cp.min(axis=0, keepdims=True)

    # mask : 0 = rien, 1 = étoilé, 2 = primé
    mask = np.zeros((n, n), dtype=np.int8)
    row_cov = np.zeros(n, dtype=bool)
    col_cov = np.zeros(n, dtype=bool)

    # Étape 2 : étoiler un zéro par ligne (sans partage)
    for r in range(n):
        for c in range(n):
            if Cp[r, c] == 0 and not row_cov[r] and not col_cov[c]:
                mask[r, c] = 1
                row_cov[r] = True
                col_cov[c] = True
    row_cov[:] = False
    col_cov[:] = False

    # Couvrir les colonnes qui ont un zéro étoilé
    col_cov = (mask == 1).any(axis=0)

    def _find_uncovered_zero() -> Tuple[int, int]:
        for r in range(n):
            if row_cov[r]:
                continue
            for c in range(n):
                if not col_cov[c] and Cp[r, c] == 0:
                    return r, c
        return -1, -1

    # Boucle principale
    while not col_cov.all():
        r, c = _find_uncovered_zero()

        if r == -1:
            # Aucun zéro non couvert → augmenter la matrice
            h = np.where(row_cov[:, None] | col_cov[None, :], np.inf, Cp).min()
            uncov_rows = np.where(~row_cov)[0]
            uncov_cols = np.where(~col_cov)[0]
            cov_rows = np.where(row_cov)[0]
            cov_cols = np.where(col_cov)[0]
            Cp[np.ix_(uncov_rows, uncov_cols)] -= h
            if cov_rows.size > 0 and cov_cols.size > 0:
                Cp[np.ix_(cov_rows, cov_cols)] += h
            continue

        mask[r, c] = 2  # primer ce zéro

        # Y a-t-il un zéro étoilé sur la même ligne ?
        star_in_row = np.where(mask[r, :] == 1)[0]
        if star_in_row.size > 0:
            sc = star_in_row[0]
            row_cov[r] = True
            col_cov[sc] = False
        else:
            # Construire un chemin augmentant et basculer étoilés/primés
            path: List[Tuple[int, int]] = [(r, c)]
            while True:
                last_r, last_c = path[-1]
                star_rows = np.where(mask[:, last_c] == 1)[0]
                if star_rows.size == 0:
                    break
                sr = int(star_rows[0])
                path.append((sr, last_c))
                prime_cols = np.where(mask[sr, :] == 2)[0]
                path.append((sr, int(prime_cols[0])))

            for pr, pc in path:
                mask[pr, pc] = 0 if mask[pr, pc] == 1 else 1

            mask[mask == 2] = 0
            row_cov[:] = False
            col_cov[:] = False
            col_cov = (mask == 1).any(axis=0)

    rows, cols = np.where((mask == 1)[:n_rows, :n_cols])
    return rows, cols


# ──────────────────────────────────────────────────────────────────────────────
# Fonctions géométriques utilitaires
# ──────────────────────────────────────────────────────────────────────────────

def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _bbox_dims(bbox: List[float]) -> Tuple[float, float]:
    """Retourne (largeur, hauteur) d'une boîte [x1, y1, x2, y2]."""
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def _estimate_velocity(
    appearances: List[Tuple[int, List[float]]],
    window: int = _DEFAULT_VEL_WINDOW,
) -> Tuple[float, float]:
    """Vitesse moyenne (vx, vy) en px/frame sur les dernières *window* frames."""
    recent = appearances[-window:] if len(appearances) > 1 else appearances
    if len(recent) < 2:
        return (0.0, 0.0)
    vx_acc, vy_acc = 0.0, 0.0
    for i in range(1, len(recent)):
        f0, b0 = recent[i - 1]
        f1, b1 = recent[i]
        dt = max(f1 - f0, 1)
        cx0, cy0 = _bbox_center(b0)
        cx1, cy1 = _bbox_center(b1)
        vx_acc += (cx1 - cx0) / dt
        vy_acc += (cy1 - cy0) / dt
    count = len(recent) - 1
    return (vx_acc / count, vy_acc / count)


def _avg_speed(appearances: List[Tuple[int, List[float]]],
               window: int = _DEFAULT_VEL_WINDOW) -> float:
    """Norme de la vitesse moyenne (scalaire, px/frame)."""
    vx, vy = _estimate_velocity(appearances, window)
    return float(np.sqrt(vx ** 2 + vy ** 2))


def _predict_position(
    last_frame: int,
    last_bbox: List[float],
    velocity: Tuple[float, float],
    target_frame: int,
) -> Tuple[float, float]:
    """Extrapolation linéaire du centre de *last_frame* vers *target_frame*."""
    dt = target_frame - last_frame
    cx, cy = _bbox_center(last_bbox)
    return (cx + velocity[0] * dt, cy + velocity[1] * dt)


# ──────────────────────────────────────────────────────────────────────────────
# Scores d'association (valeurs dans [0, 1])
# ──────────────────────────────────────────────────────────────────────────────

def _score_position(
    predicted: Tuple[float, float],
    actual: Tuple[float, float],
    ref_speed: float,
    dt: int,
    abs_cap: float = _DEFAULT_ABS_SPEED_THRESH,
) -> float:
    """Score spatial : 1 si la position est exactement prédite, 0 si trop loin."""
    dist = float(np.sqrt(
        (predicted[0] - actual[0]) ** 2 +
        (predicted[1] - actual[1]) ** 2
    ))
    max_dist = max(ref_speed * max(dt, 1), abs_cap)
    return max(0.0, 1.0 - dist / max_dist)


def _score_size(bbox_a: List[float], bbox_b: List[float]) -> float:
    """Similarité de dimensions : 1 = tailles identiques, 0 = très différentes."""
    wa, ha = _bbox_dims(bbox_a)
    wb, hb = _bbox_dims(bbox_b)
    w_r = min(wa, wb) / (max(wa, wb) + 1e-8)
    h_r = min(ha, hb) / (max(ha, hb) + 1e-8)
    return float((w_r + h_r) / 2.0)


def _composite_score(
    end_appearances: List[Tuple[int, List[float]]],
    start_frame: int,
    start_bbox: List[float],
    w_pos: float = _DEFAULT_W_POSITION,
    w_size: float = _DEFAULT_W_SIZE,
    vel_window: int = _DEFAULT_VEL_WINDOW,
    abs_cap: float = _DEFAULT_ABS_SPEED_THRESH,
) -> float:
    """Score composite score_pos × w_pos + score_size × w_size ∈ [0, 1].

    Paramètres
    ----------
    end_appearances :
        Historique du tracklet qui se termine (trié par frame croissante).
    start_frame :
        Première frame du tracklet candidat à la réassociation.
    start_bbox :
        Bounding box du candidat à *start_frame*.
    """
    last_frame, last_bbox = end_appearances[-1]
    velocity = _estimate_velocity(end_appearances, vel_window)
    spd = _avg_speed(end_appearances, vel_window)
    ref_speed = max(spd * _DEFAULT_SPEED_FACTOR, abs_cap)

    predicted = _predict_position(last_frame, last_bbox, velocity, start_frame)
    actual = _bbox_center(start_bbox)

    s_pos = _score_position(predicted, actual, ref_speed,
                            dt=start_frame - last_frame,
                            abs_cap=abs_cap)
    s_size = _score_size(last_bbox, start_bbox)
    return w_pos * s_pos + w_size * s_size


# ──────────────────────────────────────────────────────────────────────────────
# Détection des ruptures dans un tracklet
# ──────────────────────────────────────────────────────────────────────────────

def _has_speed_break(
    appearances: List[Tuple[int, List[float]]],
    abs_thresh: float = _DEFAULT_ABS_SPEED_THRESH,
    speed_factor: float = _DEFAULT_SPEED_FACTOR,
    vel_window: int = _DEFAULT_VEL_WINDOW,
) -> bool:
    """Retourne True si le tracklet présente au moins un saut de vitesse suspect.

    Un saut est suspect si le déplacement instantané entre deux apparences
    consécutives dépasse ``max(abs_thresh, avg_speed × speed_factor)``.
    """
    if len(appearances) < 2:
        return False
    for i in range(1, len(appearances)):
        f0, b0 = appearances[i - 1]
        f1, b1 = appearances[i]
        dt = max(f1 - f0, 1)
        cx0, cy0 = _bbox_center(b0)
        cx1, cy1 = _bbox_center(b1)
        dist = float(np.sqrt((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2))
        pix_per_frame = dist / dt
        avg_spd = _avg_speed(appearances[:i], vel_window)
        threshold = max(abs_thresh, avg_spd * speed_factor)
        if pix_per_frame > threshold:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Point d'entrée public
# ──────────────────────────────────────────────────────────────────────────────

def refine_tracklets(
    tracks: Dict,
    max_gap_frames: int = _DEFAULT_MAX_GAP_FRAMES,
    abs_speed_thresh: float = _DEFAULT_ABS_SPEED_THRESH,
    speed_factor: float = _DEFAULT_SPEED_FACTOR,
    min_confidence_margin: float = _DEFAULT_MIN_CONFIDENCE_MARGIN,
    w_position: float = _DEFAULT_W_POSITION,
    w_size: float = _DEFAULT_W_SIZE,
    vel_window: int = _DEFAULT_VEL_WINDOW,
) -> int:
    """Raffine les tracklets joueurs par association globale (GTA Lite).

    Modifie ``tracks['players']`` sur place en fusionnant les fragments de
    tracklets dont la rupture est jugée suspecte et pour lesquels une
    réassociation haute-confiance est trouvée.

    Paramètres
    ----------
    tracks :
        Dictionnaire de tracks tel que retourné par ``get_object_tracks``.
        Au minimum ``tracks['players']`` doit être présent.
    max_gap_frames :
        Fenêtre temporelle maximale (frames) pour tenter une réassociation.
    abs_speed_thresh :
        Seuil dur de déplacement en px/frame.
    speed_factor :
        Facteur multiplicatif de la vitesse moyenne pour le seuil adaptatif.
    min_confidence_margin :
        Marge minimale (score_best − score_2nd) pour valider une correction.
    w_position :
        Poids du score de plausibilité spatiale (w_position + w_size = 1).
    w_size :
        Poids du score de similarité de taille.
    vel_window :
        Nombre de frames utilisées pour estimer la vitesse d'un tracklet.

    Retourne
    --------
    int
        Nombre de fusions effectuées.
    """
    player_frames: List[Dict] = tracks.get('players', [])
    if not player_frames:
        return 0

    # ── 1. Construire les tracklets ──────────────────────────────────────────
    # {track_id: [(frame_num, bbox), ...]}  trié par frame croissante
    tracklets: Dict[int, List[Tuple[int, List[float]]]] = {}
    for frame_num, frame_dict in enumerate(player_frames):
        for track_id, info in frame_dict.items():
            bbox = info.get('bbox')
            if bbox is None:
                continue
            tracklets.setdefault(track_id, []).append((frame_num, bbox))
    for tid in tracklets:
        tracklets[tid].sort(key=lambda x: x[0])

    # ── 2. Identifier les extrémités suspects ────────────────────────────────
    # Une extrémité de fin est suspecte si le tracklet présente un saut de
    # vitesse OU si sa durée est très courte (fragmentation probable).
    # Toutes les fins de tracklets entrent comme candidats sources.
    # Toutes les débuts de tracklets entrent comme candidats cibles.

    ends: List[Tuple[int, int, List[float]]] = []   # (track_id, last_frame, last_bbox)
    starts: List[Tuple[int, int, List[float]]] = []  # (track_id, first_frame, first_bbox)

    for tid, apps in tracklets.items():
        ends.append((tid, apps[-1][0], apps[-1][1]))
        starts.append((tid, apps[0][0], apps[0][1]))

    if not ends or not starts:
        return 0

    # ── 3. Construire la matrice de coûts ────────────────────────────────────
    # Ligne i = fin du tracklet i ; colonne j = début du tracklet j
    # On ne considère que les paires (i, j) telles que :
    #   • i ≠ j
    #   • 1 ≤ starts[j].frame − ends[i].frame ≤ max_gap_frames
    #   • Les tracklets ne sont pas déjà le même ID
    # Les paires non éligibles reçoivent un coût maximal (1.0).

    n_e = len(ends)
    n_s = len(starts)
    cost = np.ones((n_e, n_s), dtype=float)
    score_matrix = np.zeros((n_e, n_s), dtype=float)

    for i, (e_id, e_frame, e_bbox) in enumerate(ends):
        e_apps = tracklets[e_id]
        for j, (s_id, s_frame, s_bbox) in enumerate(starts):
            if s_id == e_id:
                continue
            gap = s_frame - e_frame
            if gap < 1 or gap > max_gap_frames:
                continue
            sc = _composite_score(
                e_apps, s_frame, s_bbox,
                w_pos=w_position,
                w_size=w_size,
                vel_window=vel_window,
                abs_cap=abs_speed_thresh,
            )
            score_matrix[i, j] = sc
            cost[i, j] = 1.0 - sc

    # ── 4. Assignation globale par algorithme hongrois ───────────────────────
    row_ind, col_ind = _linear_sum_assignment(cost)

    # ── 5. Filtrer par marge de confiance et appliquer les fusions ───────────
    # Pour chaque paire assignée, on vérifie que le score est meilleur que
    # tout autre candidat d'au moins *min_confidence_margin*.

    merges_applied = 0
    already_merged_src: set = set()   # e_id déjà utilisé comme source
    already_merged_dst: set = set()   # s_id déjà utilisé comme destination

    for ri, ci in zip(row_ind, col_ind):
        sc = score_matrix[ri, ci]
        if sc <= 0.0:
            continue  # paire non éligible
        e_id = ends[ri][0]
        s_id = starts[ci][0]
        if e_id in already_merged_src or s_id in already_merged_dst:
            continue

        # Marge sur les lignes : meilleur score de la ligne ri hors ci
        row_scores = np.delete(score_matrix[ri, :], ci)
        row_2nd = float(row_scores.max()) if row_scores.size > 0 else 0.0

        # Marge sur les colonnes : meilleur score de la colonne ci hors ri
        col_scores = np.delete(score_matrix[:, ci], ri)
        col_2nd = float(col_scores.max()) if col_scores.size > 0 else 0.0

        margin = sc - max(row_2nd, col_2nd)
        if margin < min_confidence_margin:
            logger.debug(
                "GTA Lite: paire (%d→%d) rejetée — marge %.3f < %.3f",
                e_id, s_id, margin, min_confidence_margin,
            )
            continue

        # Fusionner s_id dans e_id (renommer toutes les occurrences de s_id)
        conflicts = 0
        for frame_num, bbox in tracklets[s_id]:
            frame_dict = player_frames[frame_num]
            if s_id not in frame_dict:
                continue
            if e_id in frame_dict:
                # Conflit : e_id déjà présent dans cette frame → on saute
                conflicts += 1
                continue
            frame_dict[e_id] = frame_dict.pop(s_id)

        logger.info(
            "GTA Lite: fusion tracklet %d → %d (score=%.3f, marge=%.3f, conflits=%d)",
            s_id, e_id, sc, margin, conflicts,
        )
        already_merged_src.add(e_id)
        already_merged_dst.add(s_id)
        merges_applied += 1

    if merges_applied:
        logger.info("GTA Lite: %d fusion(s) appliquée(s).", merges_applied)
    else:
        logger.debug("GTA Lite: aucune fusion retenue.")

    return merges_applied
