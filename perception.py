import cv2


def detect(frame, cfg):
    p = cfg['perception']
    x0, x1 = p['crop_x_start'], p['crop_x_end']
    y0, y1 = p['crop_y_start'], p['crop_y_end']
    thr = p['threshold']
    min_area = p['min_contour_area']
    ground_y = p['ground_line_y']
    tol = p['ground_tolerance']
    dino_right = p['dino_right_edge']
    dmx0 = p['dino_mask_x_start']
    dmx1 = p['dino_mask_x_end']

    crop = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    nearest = None
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        abs_x = x + x0
        abs_y = y + y0
        abs_bot = abs_y + h
        # drop the dino's own contour: ground-touching and fully inside its x band
        if abs_bot >= ground_y - tol and abs_x >= dmx0 and abs_x + w <= dmx1:
            continue
        kind = 'ground' if abs_bot >= ground_y - tol else 'flying'
        dist = abs_x - dino_right
        if nearest is None or dist < nearest['distance']:
            nearest = {'present': True, 'distance': int(dist),
                       'type': kind, 'height': int(abs_y)}

    if nearest is None:
        return {'present': False, 'distance': None, 'type': None, 'height': None}
    return nearest
