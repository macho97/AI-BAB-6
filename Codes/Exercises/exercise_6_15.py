# Exercise 6.15
# Modify the previous Python program (Example 6.16), trying different photos for the
# face swap. Comment on the results.
import sys
import os
import numpy as np
import cv2
import face_recognition


def readPoints(path):
    points = []
    image = cv2.imread(path)
    if image is None:
        return points
    rgb_frame = image[:, :, ::-1]
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            for pt in face_landmarks[facial_feature]:
                points.append(pt)
    return points


def applyAffineTransform(src, srcTri, dstTri, size):
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None,
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst


def rectContains(rect, point):
    return 0 <= point[0] < rect[2] and 0 <= point[1] < rect[3]


def calculateDelaunayTriangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)
    triangleList = subdiv.getTriangleList()
    delaunayTri = []
    for t in triangleList:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            for j in range(3):
                pt_j = [pt1, pt2, pt3][j]
                for k in range(len(points)):
                    if abs(pt_j[0] - points[k][0]) < 1.0 and abs(pt_j[1] - points[k][1]) < 1.0:
                        ind.append(k)
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))
    return delaunayTri


def warpTriangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    t1Rect = [((t1[i][0] - r1[0]), (t1[i][1] - r1[1])) for i in range(3)]
    t2Rect = [((t2[i][0] - r2[0]), (t2[i][1] - r2[1])) for i in range(3)]
    t2RectInt = [((t2[i][0] - r2[0]), (t2[i][1] - r2[1])) for i in range(3)]

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    size = (r2[2], r2[3])
    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    img2Rect = img2Rect * mask

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = (
        img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * ((1.0, 1.0, 1.0) - mask) + img2Rect
    )


def swap_faces(src_file, dst_file, output_file="face_swapped.jpg"):
    print(f"\nMelakukan swap wajah dari {src_file} ke {dst_file}...")
    img1 = cv2.imread(src_file)
    img2 = cv2.imread(dst_file)
    if img1 is None or img2 is None:
        print("File gambar tidak ditemukan.")
        return

    img1Warped = np.copy(img2)
    points1 = readPoints(src_file)
    points2 = readPoints(dst_file)

    if len(points1) == 0 or len(points2) == 0:
        print("Gagal mendeteksi landmark wajah pada salah satu gambar.")
        return

    hull1 = []
    hull2 = []
    hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)
    for i in range(len(hullIndex)):
        hull1.append(points1[int(hullIndex[i])])
        hull2.append(points2[int(hullIndex[i])])

    sizeImg2 = img2.shape
    rect = (0, 0, sizeImg2[1], sizeImg2[0])
    dt = calculateDelaunayTriangles(rect, hull2)

    for i in range(len(dt)):
        t1 = [hull1[dt[i][j]] for j in range(3)]
        t2 = [hull2[dt[i][j]] for j in range(3)]
        warpTriangle(img1, img1Warped, t1, t2)

    hull8U = [(hull2[i][0], hull2[i][1]) for i in range(len(hull2))]
    mask = np.zeros(img2.shape, dtype=img2.dtype)
    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    r = cv2.boundingRect(np.float32([hull2]))
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)

    cv2.imwrite(output_file, output)
    cv2.imshow(f"Face Swap: {src_file} -> {dst_file}", output)
    cv2.waitKey(0)


if __name__ == '__main__':
    # Menguji pasangan foto berbeda
    test_pairs = [
        ('face0.jpg', 'face1.jpeg'),
        ('face1.jpeg', 'face2.jpeg'),
    ]
    for src, dst in test_pairs:
        if os.path.exists(src) and os.path.exists(dst):
            swap_faces(src, dst)
    cv2.destroyAllWindows()
