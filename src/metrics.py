#!/usr/bin/env python3
'''
Compute pipeline metrics.
Localization: IoU
Segmentation: character-count agreement (GT has no per-char bboxes)
OCR: Levenshtein similarity + confusion matrix
Pipeline: E2E accuracy (exact match)
'''
import csv
import json
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.abspath(os.path.join(project_root, "data"))
output_dir_default = os.path.join(data_path, "evaluate_output")


def load_ground_truth(validate_dir):
    '''
    Returns a dict mapping image filename -> list of ground truth bboxes.
    Reads per-image JSON sidecars: {"filename": "...", "bbox": [x, y, w, h]}.
    '''
    filename_to_bboxes = {}
    for root, dirs, files in os.walk(validate_dir):
        for fname in files:
            if not fname.endswith('.json'):
                continue
            with open(os.path.join(root, fname), 'r') as f:
                data = json.load(f)
            img_name = data.get('filename')
            bbox = data.get('bbox')
            if img_name and bbox:
                filename_to_bboxes.setdefault(img_name, []).append(bbox)
    return filename_to_bboxes


def iou(a, b):
    '''Both a and b are [x, y, w, h]. Returns IoU in [0, 1].'''
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def best_iou(predicted_boxes, gt_boxes):
    '''Max IoU between any predicted box and any GT box.'''
    if not predicted_boxes:
        return 0.0
    if not gt_boxes:
        return 0.0
    return max(iou(p, g) for p in predicted_boxes for g in gt_boxes)


def levenshtein(a, b):
    '''Edit distance between two strings.'''
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(curr[j - 1] + 1,
                            prev[j] + 1,
                            prev[j - 1] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def levenshtein_similarity(pred, gt):
    '''1 - edit_distance / max(len). Returns [0, 1].'''
    if not pred and not gt:
        return 1.0
    denom = max(len(pred), len(gt))
    return 1.0 - levenshtein(pred, gt) / denom if denom else 1.0


def aligned_pairs(gt, pred):
    '''
    Levenshtein-alignment between gt and pred.
    Returns list of (gt_char|None, pred_char|None) covering substitutions,
    insertions, and deletions.
    '''
    m, n = len(gt), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if gt[i - 1] == pred[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,
                           dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)
    pairs = []
    i, j = m, n
    while i > 0 and j > 0:
        cost = 0 if gt[i - 1] == pred[j - 1] else 1
        if dp[i][j] == dp[i - 1][j - 1] + cost:
            pairs.append((gt[i - 1], pred[j - 1]))
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((gt[i - 1], None))
            i -= 1
        else:
            pairs.append((None, pred[j - 1]))
            j -= 1
    while i > 0:
        pairs.append((gt[i - 1], None))
        i -= 1
    while j > 0:
        pairs.append((None, pred[j - 1]))
        j -= 1
    return list(reversed(pairs))


def collect_predictions(output_dir):
    '''
    For each image subdir in output_dir, read recognition/Rec.
    Returns {image_name: prediction_text}. Missing rec dir or "No characters
    found" sentinel -> empty string.
    '''
    preds = {}
    for entry in os.scandir(output_dir):
        if not entry.is_dir():
            continue
        rec_file = os.path.join(entry.path, 'recognition', 'Rec')
        text = ''
        if os.path.exists(rec_file):
            with open(rec_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            if text == 'No characters found':
                text = ''
        preds[entry.name] = text
    return preds


def collect_localization_iou(output_dir):
    '''{image_name: iou_float} from per-image iou.txt.'''
    ious = {}
    for entry in os.scandir(output_dir):
        if not entry.is_dir():
            continue
        iou_file = os.path.join(entry.path, 'iou.txt')
        if not os.path.exists(iou_file):
            continue
        with open(iou_file, 'r') as f:
            try:
                ious[entry.name] = float(f.read().strip())
            except ValueError:
                pass
    return ious


def collect_segmentation_counts(output_dir):
    '''{image_name: number of segmented character images}.'''
    counts = {}
    for entry in os.scandir(output_dir):
        if not entry.is_dir():
            continue
        char_dir = os.path.join(entry.path, 'characters')
        if os.path.isdir(char_dir):
            counts[entry.name] = sum(1 for f in os.listdir(char_dir)
                                     if f.lower().endswith('.jpg'))
        else:
            counts[entry.name] = 0
    return counts


def regenerate_csv(csv_path, output_dir):
    '''
    Read csv_path (filename, ground_truth, predicted), replace prediction
    with current pipeline output from output_dir/<image>/recognition/Rec,
    write back to csv_path. Returns the rows.
    '''
    preds = collect_predictions(output_dir)
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if not row:
                continue
            filename = row[0]
            gt = row[1] if len(row) > 1 else ''
            rows.append([filename, gt, preds.get(filename, '')])
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'ground_truth', 'predicted'])
        writer.writerows(rows)
    return rows


def compute_metrics(rows, output_dir):
    ious = collect_localization_iou(output_dir)
    seg_counts = collect_segmentation_counts(output_dir)

    iou_vals = []
    lev_vals = []
    seg_agreement = []
    e2e_correct = 0
    e2e_total = 0
    confusion = {}

    for filename, gt, pred in rows:
        gt = gt.strip().upper()
        pred = pred.strip().upper()

        if filename in ious:
            iou_vals.append(ious[filename])

        if not gt:
            continue

        lev_vals.append(levenshtein_similarity(pred, gt))
        e2e_total += 1
        if pred == gt:
            e2e_correct += 1

        seg_n = seg_counts.get(filename, 0)
        gt_n = len(gt)
        denom = max(seg_n, gt_n)
        if denom > 0:
            seg_agreement.append(min(seg_n, gt_n) / denom)

        for g, p in aligned_pairs(gt, pred):
            confusion[(g, p)] = confusion.get((g, p), 0) + 1

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        'localization_iou_mean': avg(iou_vals),
        'localization_iou_count': len(iou_vals),
        'segmentation_count_agreement_mean': avg(seg_agreement),
        'segmentation_count_agreement_count': len(seg_agreement),
        'ocr_levenshtein_similarity_mean': avg(lev_vals),
        'ocr_levenshtein_similarity_count': len(lev_vals),
        'e2e_accuracy': e2e_correct / e2e_total if e2e_total else 0.0,
        'e2e_correct': e2e_correct,
        'e2e_total': e2e_total,
        'confusion_matrix': confusion,
    }


def write_summary(metrics, path):
    with open(path, 'w', encoding='utf-8') as f:
        for k, v in metrics.items():
            if k == 'confusion_matrix':
                f.write('confusion_matrix (gt -> pred : count, ∅ = gap):\n')
                for (g, p), c in sorted(v.items(), key=lambda x: -x[1]):
                    f.write(f'  {g or "∅"} -> {p or "∅"}: {c}\n')
            else:
                f.write(f'{k}: {v}\n')


if __name__ == '__main__':
    csv_path = os.path.join(output_dir_default, 'comparison_new')
    rows = regenerate_csv(csv_path, output_dir_default)
    metrics = compute_metrics(rows, output_dir_default)
    summary_path = os.path.join(output_dir_default, 'metrics_summary.txt')
    write_summary(metrics, summary_path)

    print(f'Rewrote {csv_path} ({len(rows)} rows)')
    print(f'Summary written to {summary_path}')
    print()
    for k, v in metrics.items():
        if k == 'confusion_matrix':
            print(f'{k}: {len(v)} unique pairs (see summary file)')
        else:
            print(f'{k}: {v}')
