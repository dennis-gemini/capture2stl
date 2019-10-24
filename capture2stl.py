# capture2stl.py
# Convert the 2D image captured by webcam into the grayscaled lithophane STL file
#
# Written by Dennis Chen (Oct 2017)
#
# Installation:
#   Windows:
#   1. Download python installer: 
#   2. Download numpy installer as newer as possible:
#   3. Download opencv self-extractor, copy cv2.pyd to python's DLLs directory
#   4. In the cmd console, goto python's Scripts directory and execute "pip install numpy-stl" if network is available
#
#   Linux/MacOS
#   1. pip install numpy
#   2. pip install opencv
#   3. pip install numpy-stl
#

import sys
import cv2
import numpy as np
import math
from stl import mesh

def _get_option(options, key, default):
    if key in options:
        return options[key]
    return default


def _white_balanced(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0]) # equalize the histogram of the Y channel
    # convert the YUV image back to RGB format
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)


_bg_filter = None
def _exclude_background(img):
    global _bg_filter

    if _bg_filter is None:
        _bg_filter = cv2.createBackgroundSubtractorMOG2()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = _bg_filter.apply(img)
    mask = cv2.dilate(mask, kernel, iterations = 2)
    mask = cv2.morphologyEx(mask , cv2.MORPH_OPEN, kernel, iterations = 3)
    mask = cv2.dilate(mask, kernel, iterations = 1)

    left, top, right, bottom = sys.maxint, sys.maxint, 0, 0
    _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key = cv2.contourArea, reverse = True)[:10]:
        x, y, w, h = cv2.boundingRect(c)
        left, top, right, bottom = min(left, x), min(top, y), max(right, x + w), max(bottom, y + h)

    mask = np.zeros_like(img)
    if len(mask.shape) == 2:
        cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)
    else:
        cv2.rectangle(mask, (left, top), (right, bottom), (255,) * mask.shape[2], -1)

    return cv2.bitwise_and(img, mask)


def _filter_white(img):
    hls   = cv2.cvtColor(img, cv2.COLOR_BGR2HLS) 
    lower = np.uint8([  0, 160,   0])
    upper = np.uint8([255, 255, 255])
    mask_hls = cv2.inRange(hls, lower, upper)

    lower = np.uint8([127, 127, 127])
    upper = np.uint8([255, 255, 255])
    mask_rgb = cv2.inRange(img, lower, upper)

    return cv2.bitwise_and(img, img, mask = cv2.bitwise_or(mask_hls, mask_rgb))


def _detect_edges(img, kernel_size = 5, low_threshold = 50, high_threshold = 150):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    img = cv2.Canny(img, low_threshold, high_threshold)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations = 2)
    return img


def _find_contours(img, edges, draw_color = (0, 0, 255), draw_width = 5):
    _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    matched = None
    candidates = []
    for c in sorted(contours, key = cv2.contourArea, reverse = True)[:2]:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)

        if len(approx) == 4:
            box = np.int0(cv2.boxPoints(cv2.minAreaRect(c)))
            area1 = cv2.contourArea(approx)
            area2 = cv2.contourArea(box)

            if area1 / area2 >= 0.7:
                candidates.append((approx, box))

    if candidates:
        approx1, box1 = candidates.pop(0)
        x1, y1, w1, h1 = cv2.boundingRect(box1)

        #cv2.drawContours(img, [approx1], 0, (255, 0, 0), draw_width)

        if candidates:
            left, top, right, bottom = x1, y1, x1 + w1, y1 + h1
            for approx2, box2 in candidates:
                #cv2.drawContours(img, [approx2], 0, (255, 0, 0), draw_width)

                x2, y2, w2, h2 = cv2.boundingRect(box2)
                if (x2 > right) or (x2 + w2 < left) or (y2 > bottom) or (y2 + h2 < top):
                    continue
                if (x2 >= left) and (x2 + w2 <= right) and (y2 >= top) and (y2 + h2 <= bottom):
                    area1 = cv2.contourArea(approx1)
                    area2 = cv2.contourArea(approx2)
                    if area2/area1 >= 0.9:
                        matched = approx2, box2
                    else:
                        matched = approx1, box1
                    break
        else:
             matched = approx1, box1

    if matched:
        approx, box = matched
        if draw_color:
            cv2.drawContours(img, [approx], 0, draw_color, draw_width)
            x1, y1, w1, h1 = cv2.boundingRect(box)
            x2, y2, w2, h2 = cv2.boundingRect(approx)
            x = min((x1 + x1)/2, min([a[0][0] for a in approx]))
            y = min((y1 + y2)/2, min([a[0][1] for a in approx]))
            cv2.putText(img, "Detected", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, draw_color, draw_width / 2, cv2.LINE_AA)
        return np.vstack(approx).squeeze()

    return None


def _imclearborder(img, detected, binary_threshold, radius = 5):
    src = cv2.bitwise_not(img)
    src = cv2.threshold(src, binary_threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #src = cv2.threshold(src, binary_threshold, 255, cv2.THRESH_BINARY)[1]

    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #src = cv2.GaussianBlur(src, (5, 5), 0)
    #src = cv2.Canny(src, 50, 150)
    #src = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel, iterations = 2)

    _, contours, _ = cv2.findContours(src, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rows, cols = src.shape[:2]

    contour_list = []
    for i in xrange(len(contours)):
        for point in contours[i]:
            x, y = point[0]
            if (y >= 0 and y < radius) or (y >= rows - 1 - radius and y < rows) or \
               (x >= 0 and x < radius) or (x >= cols - 1 - radius and x < cols):
                contour_list.append(contours[i])
                break

    mask = np.zeros_like(img, dtype = np.uint8)
    cv2.drawContours(mask, contour_list, -1, (255, 255, 255), -1)
    cv2.drawContours(detected, contour_list, -1, (0, 255, 255), 1)

    nr_black = np.count_nonzero(mask == 0)
    nr_white = np.count_nonzero(mask != 0)
    
    if nr_white > nr_black:
        return cv2.bitwise_and(img, mask)
    return cv2.bitwise_or(img, mask)


def _normalize_crop(img, poly):
    src_rect = np.zeros((4, 2), dtype = "float32")
    a = np.array(poly)
    s = a.sum(axis = 1)
    src_rect[0] = a[np.argmin(s)]    # top-left: smallest sum
    src_rect[2] = a[np.argmax(s)]    # bottom-right: largest sum
    d = np.diff(a, axis = 1)
    src_rect[1] = a[np.argmin(d)] # top-right: smallest difference
    src_rect[3] = a[np.argmax(d)] # bottom-left: largest difference

    (tl, tr, br, bl) = src_rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst_rect = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1],
    ], dtype = "float32")

    M = cv2.getPerspectiveTransform(src_rect, dst_rect)
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))


def _grid_images(img_list):
    row = None
    img = None
    n   = len(img_list)
    h   = int(round(math.sqrt(n)))
    w   = int(math.ceil(float(n) / h))
    d   = max(h, w)
    x   = 0

    grid_width  = max([src.shape[1] for _, src in img_list]) / d
    grid_height = max([src.shape[0] for _, src in img_list]) / d
    grid_ratio  = grid_width / grid_height

    for title, src in img_list:
        height, width = src.shape[:2]

        if src.dtype != np.uint8:
            src = cv2.convertScaleAbs(src)
        if len(src.shape) == 2:
            src = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)

        if width / height < grid_ratio:
            width = min(grid_height * width / height, grid_width)
            height = grid_height
        else:
            height = min(grid_width * height / width, grid_height)
            width = grid_width

        src = cv2.resize(src, (width, height))

        if src.shape[0] != grid_height or src.shape[1] != grid_width:
            canvas = np.zeros([grid_height, grid_width, 3], dtype = np.uint8)
            canvas[:src.shape[0],:src.shape[1],:3] = src
            src = canvas

        cv2.putText(src, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2, cv2.LINE_AA)

        if row is None:
            row = src
        else:
            row = np.hstack((row, src))

        x += 1
        if x >= w:
            if img is None:
                img = row
            else:
                img = np.vstack((img, row))
            row = None
            x = 0

    if row is not None:
        if img is None:
            img = row
        else:
            if x < w:
                _, width = img.shape[:2]
                height, width0  = row.shape[:2]
                blank = np.zeros([height, width - width0, 3], dtype = np.uint8)
                blank[:] = (128, 128, 128)
                row = np.hstack((row, blank))

            img = np.vstack((img, row))
        row = None
    return img


def _get_message_pane(text, width, height, font_size = 1.0, stroke_width = 2, fgcolor = (0, 255, 0), bgcolor = (128, 128, 128), x = -1, y = -1, highlight = False, line_gap = 4):
    canvas = np.zeros([height, width, 3], dtype = np.uint8)

    for i, c in enumerate(bgcolor):
        canvas[:,:,i] = c

    lines = [l.strip() for l in text.split("\n")]
    textsizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_size, stroke_width) for line in lines]
    boxes = np.array([(b[0], b[1] + d) for b, d in textsizes])
    text_width  = boxes.max(axis = 0)[0] + line_gap * 2
    text_height = boxes.sum(axis = 0)[1] + (line_gap * (len(lines) + 1))
 
    if x < 0:
        x = (width - text_width) / 2
    if y < 0:
        y = (height - text_height) / 2
        if y < boxes[0][1] + line_gap:
            y = boxes[0][1] + line_gap

    if highlight:
        cv2.rectangle(canvas, (x - line_gap, y - textsizes[0][0][1] - line_gap), (x  - line_gap + text_width, y - line_gap - textsizes[0][0][1] + text_height), fgcolor, -1)
        font_color = bgcolor
    else:
        font_color = fgcolor

    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, stroke_width, cv2.LINE_AA)
        y += boxes[i][1] + line_gap

    return canvas


def capture(**options):
    cam_index     = _get_option(options, "camera", 0)
    resolution    = _get_option(options, "resolution", None)
    mirror        = _get_option(options, "mirror", True)
    white_balance = _get_option(options, "white_blance", True)
    cam           = cv2.VideoCapture(cam_index)
    original      = None
    cropped       = None
    label         = None
    blinking      = 0
    highlight     = False

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    cv2.moveWindow("result", 0, 0)
    while True:
        ret, img = cam.read()

        if mirror:
            img = cv2.flip(img, 1)

        current = img.copy()

        if white_balance:
            img = _white_balanced(img)

        effective = _filter_white(_exclude_background(img))
        edges = _detect_edges(effective)
        region = _find_contours(img, edges)

        img_list = [("Camera", img)]
        img_list.append(("Detecting Edges", edges))

        if region is not None:
            original = current
            cropped  = _normalize_crop(current, region)
            if mirror:
                original = cv2.flip(original, 1)
                cropped  = cv2.flip(cropped, 1)

        if blinking == 0:
            highlight = not highlight

        if cropped is not None:
            img_list.append(("Last Detected", cropped))
            img_list.append(("", _get_message_pane("Press [Enter] to confirm", img.shape[1], img.shape[0], font_size = 3, stroke_width = 3, highlight = highlight)))
        else:
            img_list.append(("", _get_message_pane("Image is undetected", img.shape[1], img.shape[0], font_size = 3, stroke_width = 3, highlight = highlight)))

        blinking = (blinking + 1) % 4

        cv2.imshow('result', _grid_images(img_list))

        keystroke = cv2.waitKey(100)
        #Press ESC to exit
        if keystroke == 27:
            cropped = None
            break
        #Press ENTER to confirm the cropped image
        elif keystroke in (10, 13):
            if cropped is None:
                original = current
                cropped = current

                if mirror:
                    original = cv2.flip(original, 1)
                    cropped  = cv2.flip(cropped, 1)
            break

    #cv2.destroyAllWindows()
    #cam.release()

    if cropped is None:
        return None

    return original, cropped, label


def _image2mesh(img, base_height = 2, dx = 0.3, dy = 0.3, dz = 6):
    rows, cols = img.shape[:2]

    result = []
    # bottom: draw the triangles clockwise for pointing the normals downloadwards
    result.append([[0, 0, 0], [cols * dx, rows * dy, 0], [cols * dx, 0, 0]])
    result.append([[0, 0, 0], [0, rows * dy, 0], [cols * dx, rows * dy, 0]])

    last_row = None 
    for j in xrange(rows):
        y = (rows - j - 1) * dy
        z, zn, znw, zw = 0, 0, 0, 0
        cur_row = []
        for i in xrange(cols):
            x = i * dx
            z = img[j, i] / 255.0 * dz + base_height
            cur_row.append(z)

            # back face
            if j == 0:
                if i > 0:
                    zw  = cur_row[i - 1]
                    result.append([[x, y, z], [x, y, 0], [x - dx, y, 0]])
                    result.append([[x, y, z], [x - dx, y, 0], [x - dx, y, zw]])
                continue

            # left face
            if i == 0:
                zn  = last_row[i]
                result.append([[x, y, z], [x, y + dy, zn], [x, y + dy, 0]])
                result.append([[x, y, z], [x, y + dy, 0], [x, y, 0]])
                continue

            zn  = last_row[i]
            znw = last_row[i - 1]
            zw  = cur_row[i - 1]

            # top face
            result.append([[x, y, z], [x, y + dy, zn], [x - dx, y + dy, znw]])
            result.append([[x, y, z], [x - dx, y + dy, znw], [x - dx, y, zw]])

        # right face
        result.append([[x, y, z], [x, y, 0], [x, y + dy, 0]])
        result.append([[x, y, z], [x, y + dy, 0], [x, y + dy, zn]])

        # front face
        if j == rows - 1:
            for i in xrange(1, cols):
                x = i * dx
                z = cur_row[i]
                zw = cur_row[i - 1]
                result.append([[x, y, z], [x - dx, y, zw], [x - dx, y, 0]])
                result.append([[x, y, z], [x - dx, y, 0], [x, y, 0]])

        last_row = cur_row

    data = np.zeros(len(result) * 3, dtype=mesh.Mesh.dtype)
    for i, face in enumerate(result):
        data['vectors'][i] = face

    return mesh.Mesh(data, remove_empty_areas = True, remove_duplicate_polygons = 1)


# FIXME: too slow
#def _mesh_to_3dimage(label, m):
#    from mpl_toolkits import mplot3d
#    from matplotlib import pyplot
#    figure = pyplot.figure()
#    axes = mplot3d.Axes3D(figure)
#    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(m.vectors))
#    scale = m.points.flatten(-1)
#    axes.auto_scale_xyz(scale, scale, scale)
#    figure.canvas.draw()
#    #cols, rows = figure.canvas.get_width_height()
#    #buf, size = figure.canvas.print_to_buffer()
#    #return np.fromstring(buf, dtype = np.uint8).reshape(rows, cols, 3)
#    figure.savefig("%s-thumb.png" % label)
#    return cv2.imread("%s-thumb.png" % label)


def _process_image(cropped, inverse, binary_threshold, smoothed, kernel_size, auto_scale, scale_max_width, scale_max_height, scale_target):
    detected = cropped.copy()
    img = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)

    if binary_threshold is not None:
        if binary_threshold < 0:
            binary_threshold = 127.0
        img = _imclearborder(img, detected, binary_threshold)

    if inverse:
        img = cv2.bitwise_not(img)

    if binary_threshold is not None:
        img = cv2.threshold(img, binary_threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        #img = cv2.threshold(img, binary_threshold, 255, cv2.THRESH_BINARY)[1]
        #img[img < binary_threshold] = 0
        #img[img >= binary_threshold] = 255
        #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, binary_threshold)

    if smoothed:
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 2)

    if scale_target == 0 and auto_scale:
        h, w = img.shape[:2]
        aspect = min(float(scale_max_width) / w, float(scale_max_height) / h)
        effective = cv2.resize(img, (int(w * aspect), int(h * aspect)), interpolation=cv2.INTER_AREA if aspect < 1.0 else cv2.INTER_CUBIC)
    else:
        effective = img
        img = None

    return effective, img, detected


def build_stl(original, cropped, label, **options):
    inverse              = _get_option(options, "inverse", True)
    binary               = _get_option(options, "binary", False)
    def_binary_threshold = _get_option(options, "def_binary_threshold", 127)
    binary_threshold     = def_binary_threshold if binary else None
    kernel_size          = _get_option(options, "kernel_size", 5)
    smoothed             = _get_option(options, "smoothed", True)
    auto_scale           = _get_option(options, "auto_scale", True)
    scale_max_width      = _get_option(options, "scale_max_width", 32)
    scale_max_height     = _get_option(options, "scale_max_height", 18)
    scale_target         = _get_option(options, "scale_target", 0)
    base_height          = _get_option(options, "base_height", 2)
    dx                   = _get_option(options, "dx", 0.3)
    dy                   = _get_option(options, "dy", 0.3)
    dz                   = _get_option(options, "dz", 6)

    cv2.moveWindow("result", 0, 0)
    if label is None:
        label = ""

    exchange_counter = 0
    rotate_image = 0
    while True:
        effective, unscaled, detected = _process_image(cropped, inverse, binary_threshold, smoothed, kernel_size, auto_scale, scale_max_width / dx, scale_max_height / dy, scale_target)
        exchange_counter = (exchange_counter + 1) % 2
        if exchange_counter == 0:
            if unscaled is not None:
                rotate_image = (rotate_image + 1) % 3
            else:
                rotate_image = (rotate_image + 1) % 2

        img_list = [
            ("%s-original.png" % label if label else "Original", original),
        ]
        if rotate_image == 0:
            img_list.append(("%s-effective.png" % label if label else "Filtered", effective))
        elif rotate_image == 1:
            img_list.append(("%s-cropped.png" % label if label else "Detected", detected))
        else:
            img_list.append(("%s-unscaled.png" % label if label else "Unscaled", unscaled))

        input_dialog_img = _grid_images(img_list)

        msg = [
            "<ESC> to cancel",
            "<Space> to toggle grayscaled/binary mode",
            #"[" or "]" to -/+ binary threshold",
            "<Tab> to toggle image inversion", 
            "-------------------------------------",
            "Mode: %s" % ("Grayscaled" if binary_threshold is None else "Binary"),
            #"Threshold: %s" % ("N/A" if binary_threshold is None else str(binary_threshold)),
            "Inversed: %s" % ("True (Embossed)" if inverse else "False (Debossed)"),
            "Enter name: [%s]" % (label if label else "").ljust(5),
        ]
        cv2.imshow('result', np.vstack((input_dialog_img, _get_message_pane("\n".join(msg), input_dialog_img.shape[1], input_dialog_img.shape[0], font_size = 1, stroke_width = 2))))

        keystroke = cv2.waitKey(200)
        if keystroke <= 0:
            continue
        if keystroke == 27:
            return None, None, None, None
        if keystroke in (10, 13) and label:
            break

        if keystroke in (8, 127):
            label = label[:-1]
        elif chr(keystroke) in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_@()':
            if len(label) < 40:
                label += chr(keystroke)
        elif keystroke == 32:
            if binary_threshold is None: 
                binary_threshold = def_binary_threshold
            else:
                binary_threshold = None
        elif keystroke == 9: # tab
            inverse = not inverse
        elif keystroke == 91: # [
            if binary_threshold is not None and binary_threshold > 0:
                binary_threshold -= 1
        elif keystroke == 93: # ]
            if binary_threshold is not None and binary_threshold < 255:
                binary_threshold += 1
    input_dialog_img = None

    img_list.append(("", _get_message_pane("Creating 3D model...\nWaiting for completion", original.shape[1], original.shape[0], font_size = 3, stroke_width = 3)))
    cv2.imshow('result', _grid_images(img_list))
    cv2.waitKey(100)

    if scale_target == 1 and auto_scale:
        h, w = effective.shape[:2]
        aspect = min(float(scale_max_width) / dx / w, float(scale_max_height) / dy / h)
        dx *= aspect
        dy *= aspect

    m = _image2mesh(effective, base_height = base_height, dx = dx, dy = dy, dz = dz)

    #cv2.destroyAllWindows()
    return m, label, effective, unscaled


def save_result(original, cropped, effective, unscaled, label, m):
    if not label:
        label = "result"

    import os
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    prefix = "%s-%s" % (label, timestamp)

    try:
        if not os.path.exists("result"):
            os.mkdir("result")
    except BaseException:
        pass

    cv2.moveWindow("result", 0, 0)
    img_list = [
        ("%s-original.png" % prefix, original),
        ("%s-cropped.png" % prefix, cropped),
        None,
    ]

    try:
        cv2.imwrite(os.path.join("result", "%s-original.png" % prefix), original)
        cv2.imwrite(os.path.join("result", "%s-cropped.png" % prefix), cropped)
        cv2.imwrite(os.path.join("result", "%s-effective.png" % prefix), effective)

        if unscaled is not None:
            cv2.imwrite(os.path.join("result", "%s-unscaled.png" % prefix), unscaled)

        m.save(os.path.join("result", "%s.stl" % prefix))
        img_list[2] = ("", _get_message_pane("Saved successfully\nPress any key to return", original.shape[1], original.shape[0], font_size = 3, stroke_width = 3))
    except BaseException:
        img_list[2] = ("", _get_message_pane("Failed to save\nPress any key to return", original.shape[1], original.shape[0], font_size = 3, stroke_width = 3))

    cv2.imshow('result', _grid_images(img_list))
    cv2.waitKey(-1)
    #cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_options = {
        "camera": 0,                    # select the camera by index
        "resolution": (1280, 720),      # force the webcam resolution
        "mirror": True,                 # mirror the display
        "white_balance": True,
    }
    stl_options = {
        "inverse": True,                # inverse the output: blacker is higher; whiter is lower
        "binary": True,                 # whether to convert the grayscaled image into the binary image
        "def_binary_threshold": 127,    # convert the grayscaled image with the binary threshold; 127 can be a possible value
        "kernel_size": 5,               # used by gaussian blur
        "smoothed": True,               # gaussian-blurred edges
        "auto_scale": True,             # enable auto-scale
        "scale_max_width": 50.0,        # max width in milimeter of the output
        "scale_max_height": 17.0,       # max height in milimeter of the output
        "scale_target": 0,              # 0: scale the image for low poly, 1: scale the stl for high quality
        "base_height": 0.4,             # the base height in milimeter of the lithophane
        "dx": 0.3,                      # delta x in milimeter for each pixel
        "dy": 0.3,                      # delta y in milimeter for each pixel
        "dz": 0.8,                      # max height in milimeter for each pixel as per value 255
    }

    while True:
        result = capture(**capture_options)
        if result is None: 
            break

        original, cropped, label = result
        m, label, effective, unscaled = build_stl(original, cropped, label, **stl_options)
        if m is not None:
            save_result(original, cropped, effective, unscaled, label, m)

    cv2.destroyAllWindows()

# :vim set sw=4 ts=4 et :

