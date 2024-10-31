import cv2
import math
import numpy as np
from helpers import distance, get_head_direction, process_frame


class FaceMeshProcessor:
    def __init__(self, face_mesh):
        self.face_mesh = face_mesh
        self.right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]]  # right eye landmark positions
        self.left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]]  # left eye landmark positions
        self.mouth = [[61, 291], [39, 181], [0, 17], [269, 405]]  # mouth landmark coordinates
        self.img_h = None
        self.img_w = None
        self.mesh_coords = None
        self.nose_2d = None
        self.nose_3d = None
        self.face_2d = None
        self.face_3d = None
        self.angles = None
        self.tip = None
        self.face_detected = False
        self.image = None


    def process_image(self, image):
        self.image, results = process_frame(image, self.face_mesh)
        self.img_h, self.img_w, _ = self.image.shape

        if results.multi_face_landmarks:
            self.face_detected = True
            self.mesh_coords = [(int(lm.x * self.img_w), int(lm.y * self.img_h)) for lm in
                                results.multi_face_landmarks[0].landmark]
            ear = self._eye_feature()
            mar = self._mouth_feature()
            puc = self._pupil_feature()
            moe = mar / ear

            self.face_2d, self.face_3d, self.nose_2d, self.nose_3d = self._get_face_coordinates(results)
            self.angles, rot_vec, trans_vec, cam_matrix, dist_matrix = self._estimate_head_pose()
            _, x, y, z = get_head_direction(self.angles)
            self.tip = (int(self.nose_2d[0] + y * 10), int(self.nose_2d[1] - x * 10))
        else:
            ear = mar = puc = moe = -1000
            self.tip = (-1000, -1000)
            self.face_detected = False

        return ear, mar, puc, moe, self.tip, self.image



    def _eye_feature(self):
        """Calculate eye aspect ratio for both eyes."""
        return (self._eye_aspect_ratio(self.left_eye) + self._eye_aspect_ratio(self.right_eye)) / 2

    def _eye_aspect_ratio(self, eye):
        n1 = distance(self.mesh_coords[eye[1][0]], self.mesh_coords[eye[1][1]])
        n2 = distance(self.mesh_coords[eye[2][0]], self.mesh_coords[eye[2][1]])
        n3 = distance(self.mesh_coords[eye[3][0]], self.mesh_coords[eye[3][1]])
        d = distance(self.mesh_coords[eye[0][0]], self.mesh_coords[eye[0][1]])
        return (n1 + n2 + n3) / (3 * d)

    def _mouth_feature(self):
        """Calculate mouth aspect ratio."""
        n1 = distance(self.mesh_coords[self.mouth[1][0]], self.mesh_coords[self.mouth[1][1]])
        n2 = distance(self.mesh_coords[self.mouth[2][0]], self.mesh_coords[self.mouth[2][1]])
        n3 = distance(self.mesh_coords[self.mouth[3][0]], self.mesh_coords[self.mouth[3][1]])
        d = distance(self.mesh_coords[self.mouth[0][0]], self.mesh_coords[self.mouth[0][1]])
        return (n1 + n2 + n3) / (3 * d)

    def _pupil_circularity(self, eye):
        """Calculate the pupil circularity feature for a given eye."""
        perimeter = sum(
            distance(self.mesh_coords[eye[i][0]], self.mesh_coords[eye[(i + 1) % 4][0]]) for i in range(4))
        area = math.pi * ((distance(self.mesh_coords[eye[1][0]], self.mesh_coords[eye[3][1]]) * 0.5) ** 2)
        return (4 * math.pi * area) / (perimeter ** 2)

    def _pupil_feature(self):
        """Calculate average pupil circularity for both eyes."""
        return (self._pupil_circularity(self.left_eye) + self._pupil_circularity(self.right_eye)) / 2

    def _get_face_coordinates(self, results):
        """Extract the 2D and 3D face coordinates from landmarks."""
        face_2d, face_3d = [], []
        for idx, lm in enumerate(results.multi_face_landmarks[0].landmark):
            x, y = int(lm.x * self.img_w), int(lm.y * self.img_h)
            if idx == 1:  # Nose tip
                self.nose_2d = (lm.x * self.img_w, lm.y * self.img_h)
                self.nose_3d = (lm.x * self.img_w, lm.y * self.img_h, lm.z * 3000)
            if idx in [1, 33, 61, 199, 263, 291]:
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])
        return np.array(face_2d, dtype=np.float64), np.array(face_3d, dtype=np.float64), self.nose_2d, self.nose_3d

    def _estimate_head_pose(self):
        """Estimate head pose using 2D-3D coordinate matching."""
        focal_length = 1 * self.img_w
        cam_matrix = np.array([[focal_length, 0, self.img_h / 2],
                               [0, focal_length, self.img_w / 2],
                               [0, 0, 1]])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        success, rot_vec, trans_vec = cv2.solvePnP(self.face_3d, self.face_2d, cam_matrix, dist_matrix)
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return angles, rot_vec, trans_vec, cam_matrix, dist_matrix

    def draw_head_pose(self, image):
        """Draw the head pose direction line on the image."""
        _, x, y, z = get_head_direction(self.angles)
        p1 = (int(self.nose_2d[0]), int(self.nose_2d[1]))
        p2 = (int(self.nose_2d[0] + y * 10), int(self.nose_2d[1] - x * 10))
        cv2.line(image, p1, p2, (255, 0, 0), 1)
        # cv2.circle(image, p1, 2, (255, 0, 0), -1)
        cv2.circle(image, p2, 4, (255, 0, 0), -1)

    def draw_head_pose_points(self, image):
        """Draw head pose points on the image."""
        face_index = [1, 33, 61, 199, 263, 291]
        for point in face_index:
            x, y = self.mesh_coords[point]
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)


