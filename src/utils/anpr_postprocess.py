import re
import cv2
from Levenshtein import distance
from collections import defaultdict

def correct_anpr_format(plate_text):
    curr = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
    if len(curr) < 9 or len(curr) > 10:
        return curr

    num_to_alpha = {'0':'O', '1':'I', '2':'Z', '4':'A', '5':'S', '8':'B'}
    alpha_to_num = {'O':'0', 'I':'1', 'Z':'2', 'A':'4', 'S':'5', 'B':'8'}

    fixed = list(curr)

    for i in [0, 1]:
        if fixed[i] in num_to_alpha:
            fixed[i] = num_to_alpha[fixed[i]]

    for i in [2, 3]:
        if fixed[i] in alpha_to_num:
            fixed[i] = alpha_to_num[fixed[i]]

    for i in range(len(fixed)-4, len(fixed)):
        if fixed[i] in alpha_to_num:
            fixed[i] = alpha_to_num[fixed[i]]

    return "".join(fixed)

import Levenshtein

def is_similar_plate(p1, p2, max_diff=1):
    if len(p1) != len(p2):
        return False
    return distance(p1, p2) <= max_diff

def final_voting(predictions_list):
    clusters = []

    for text, conf, frame_idx in predictions_list:
        placed = False
        for cluster in clusters:
            rep = cluster[0][0]
            if Levenshtein.distance(text, rep) <= 2:
                cluster.append((text, conf, frame_idx))
                placed = True
                break
        
        if not placed:
            clusters.append([(text, conf, frame_idx)])

    # Merge clusters
    merged = []
    for cluster in clusters:
        added = False
        for m in merged:
            if Levenshtein.distance(cluster[0][0], m[0][0]) <= 2:
                m.extend(cluster)
                added = True
                break
        if not added:
            merged.append(cluster)

    # Weighted scoring
    best_cluster = None
    best_score = 0

    for cluster in merged:
        total_score = 0
        total_frames = len(cluster)

        for text, conf, frame_idx in cluster:
            weight = conf * (1 + frame_idx / max(1, total_frames))
            total_score += weight

        if total_score > best_score:
            best_score = total_score
            best_cluster = cluster

    if not best_cluster:
        return None, 0.0

    best_text = max(best_cluster, key=lambda x: x[1])[0]
    avg_conf = sum(c for _, c, _ in best_cluster) / len(best_cluster)

    if avg_conf < 0.7 or len(best_cluster) < 3:
        return None, avg_conf

    return best_text, avg_conf

def adaptive_conf_filter(conf, history):
    if len(history) < 3:
        return conf > 0.6
    avg_conf = sum(c for _, c in history) / len(history)
    return conf > max(0.6, avg_conf - 0.1)

def is_fast_motion(prev_box, curr_box, threshold=40):
    dx = abs(curr_box[0] - prev_box[0])
    dy = abs(curr_box[1] - prev_box[1])
    return (dx + dy) > threshold

def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var < threshold
