import cv2 as cv
import numpy as np


def process_and_scan(image_file, debug=False):
    bytes_data = image_file.getvalue()
    nparr = np.frombuffer(bytes_data, np.uint8)
    img_raw = cv.imdecode(nparr, cv.IMREAD_COLOR)

    if img_raw is None:
        raise ValueError("Image could not be loaded!")

    debug_steps = {}

    dim_limit = 1128
    max_dim = max(img_raw.shape)

    if max_dim > dim_limit:
        resize_ratio = dim_limit / max_dim
        image_work = cv.resize(img_raw, None, fx=resize_ratio, fy=resize_ratio)
        img_copy_res = image_work.copy()
    else:
        image_work = img_raw.copy()
        img_copy_res = image_work.copy()

    if debug:
        debug_steps["1_resized"] = image_work.copy()

    kernl = np.ones((5, 5), np.uint8)
    image_work = cv.morphologyEx(image_work, cv.MORPH_CLOSE, kernl)

    def remove_background(img):
        mask = np.zeros(img.shape[:2], np.uint8)
        h, w = img.shape[:2]
        rect = (10, 10, w - 20, h - 20)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        return img * mask2[:, :, np.newaxis]

    image_work = remove_background(image_work)

    if debug:
        debug_steps["2_background_removed"] = image_work.copy()

    def finding_the_edge(img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5, 5), 0)
        edge = cv.Canny(gray, 150, 300)
        edge = cv.dilate(edge, cv.getStructuringElement(cv.MORPH_RECT, (5, 5)))
        if debug:
            debug_steps["3_edges"] = edge.copy()
        contours, _ = cv.findContours(edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        return sorted(contours, key=cv.contourArea, reverse=True)[:4]

    sort_contours = finding_the_edge(image_work)

    if debug:
        contours_img = image_work.copy()
        cv.drawContours(contours_img, sort_contours, -1, (0, 255, 0), 3)
        debug_steps["4_candidate_contours"] = contours_img

    def get_corner(pts):
        pts = pts.reshape(4, 2)
        rec = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rec[0] = pts[np.argmin(s)]
        rec[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rec[1] = pts[np.argmin(diff)]
        rec[3] = pts[np.argmax(diff)]
        return rec

    found = False
    corners = None

    for k in sort_contours:
        peri = cv.arcLength(k, True)
        approx = cv.approxPolyDP(k, 0.02 * peri, True)
        if len(approx) == 4:
            corners = approx
            found = True
            break
        elif len(approx) > 4:
            hull = cv.convexHull(k)
            peri_hull = cv.arcLength(hull, True)
            approx_hull = cv.approxPolyDP(hull, 0.02 * peri_hull, True)
            if len(approx_hull) == 4:
                corners = approx_hull
                found = True
                break

    if not found:
        rect_fallback = cv.minAreaRect(sort_contours[0])
        corners = cv.boxPoints(rect_fallback).astype(np.int64)

    corners = get_corner(corners)

    if debug:
        corners_img = image_work.copy()
        for i, (x, y) in enumerate(corners):
            cv.circle(corners_img, (int(x), int(y)), 10, (0, 0, 255), -1)
            cv.putText(corners_img, str(i), (int(x) + 15, int(y)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv.polylines(corners_img, [pts], True, (255, 0, 0), 3)
        debug_steps["5_detected_corners"] = corners_img
        debug_steps["_corner_detection_method"] = "approxPolyDP" if found else "minAreaRect fallback"

    (tl, tr, br, bl) = corners
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    destination_corners = np.array([[0, 0], [maxWidth - 1, 0],
                                     [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    homography = cv.getPerspectiveTransform(corners, destination_corners)
    final = cv.warpPerspective(img_copy_res, homography, (maxWidth, maxHeight))

    if debug:
        return final, debug_steps
    return final
