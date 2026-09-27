"""Frozen A (1b311e8) geometry; preserve template offsets and black crop borders."""
from __future__ import annotations
import cv2
import numpy as np
from skimage import transform as trans

def clamp_cut_values(startX, endX, startY, endY, image):
    if startX < 0:
        startX = 0
    if endX > image.shape[1]:
        endX = image.shape[1]
    if startY < 0:
        startY = 0
    if endY > image.shape[0]:
        endY = image.shape[0]
    return startX, endX, startY, endY



def face_offset_top(face: Face, offset):
    face["bbox"][1] += offset
    face["bbox"][3] += offset
    lm106 = face.landmark_2d_106
    add = np.full_like(lm106, [0, offset])
    face["landmark_2d_106"] = lm106 + add
    return face


def resize_image_keep_content(image, new_width=512, new_height=512):
    dim = None
    (h, w) = image.shape[:2]
    if h > w:
        r = new_height / float(h)
        dim = (int(w * r), new_height)
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = new_width / float(w)
        dim = (new_width, int(h * r))
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    (h, w) = image.shape[:2]
    if h == new_height and w == new_width:
        return image
    resize_img = np.zeros(shape=(new_height, new_width, 3), dtype=image.dtype)
    offs = (new_width - w) if h == new_height else (new_height - h)
    startoffs = int(offs // 2) if offs % 2 == 0 else int(offs // 2) + 1
    offs = int(offs // 2)

    if h == new_height:
        resize_img[0:new_height, startoffs : new_width - offs] = image
    else:
        resize_img[startoffs : new_height - offs, 0:new_width] = image
    return resize_img


def rotate_image_90(image, rotate=True):
    if rotate:
        return np.rot90(image)
    else:
        return np.rot90(image, 1, (1, 0))


def rotate_anticlockwise(frame):
    return rotate_image_90(frame)


def rotate_clockwise(frame):
    return rotate_image_90(frame, False)


def rotate_image_180(image):
    return np.flip(image, 0)


# alignment code from insightface https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py

arcface_dst = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


""" def estimate_norm(lmk, image_size=112):
    assert lmk.shape == (5, 2)
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    elif image_size % 128 == 0:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    elif image_size % 512 == 0:
        ratio = float(image_size) / 512.0
        diff_x = 32.0 * ratio

    dst = arcface_dst * ratio
    dst[:, 0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M
 """

def estimate_norm(lmk, image_size=112):
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x

    if image_size == 160:
        dst[:,0] += 0.1
        dst[:,1] += 0.1
    elif image_size == 256:
        dst[:,0] += 0.5
        dst[:,1] += 0.5
    elif image_size == 320:
        dst[:,0] += 0.75
        dst[:,1] += 0.75
    elif image_size == 512:
        dst[:,0] += 1.5
        dst[:,1] += 1.5

    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M



# aligned, M = norm_crop2(f[1], face.kps, 512)
def align_crop(img, landmark, image_size=112, mode="arcface"):
    M = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped, M


def square_crop(im, S):
    if im.shape[0] > im.shape[1]:
        height = S
        width = int(float(im.shape[1]) / im.shape[0] * S)
        scale = float(S) / im.shape[0]
    else:
        width = S
        height = int(float(im.shape[0]) / im.shape[1] * S)
        scale = float(S) / im.shape[1]
    resized_im = cv2.resize(im, (width, height))
    det_im = np.zeros((S, S, 3), dtype=np.uint8)
    det_im[: resized_im.shape[0], : resized_im.shape[1], :] = resized_im
    return det_im, scale


def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    # translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data, M, (output_size, output_size), borderValue=0.0)
    return cropped, M


def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        # print('new_pt', new_pt.shape, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    # print(scale)
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        # print('new_pt', new_pt.shape, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale

    return new_pts


def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)

def create_blank_image(width, height):
    img = np.zeros((height, width, 4), dtype=np.uint8)
    img[:] = [0,0,0,0]
    return img
