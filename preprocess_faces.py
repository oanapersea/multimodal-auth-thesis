import cv2
import numpy as np
import face_recognition

from config import RAW_FACE_DIR, PROC_FACE_DIR, OUTPUT_SIZE, MARGIN_FRAC, DETECTION_MODEL

class FacePreprocessor:

    def __init__(self, size=OUTPUT_SIZE, margin=MARGIN_FRAC, model=DETECTION_MODEL):
        self.size   = size
        self.margin = margin
        self.model  = model

    def detect(self, rgb_img):
        boxes = face_recognition.face_locations(rgb_img, model=self.model)
        if not boxes:
            return None, None
        # lm is a dictionary of facial feature coordinates, for the first detected images_raw
        lm = face_recognition.face_landmarks(rgb_img, boxes)[0]
        return boxes[0], lm

    def align(self, img, le, re):
        # compute horizontal and vertical distances between eyes
        dy, dx = re[1] - le[1], re[0] - le[0]
        # computes the eye line relative to the horizontal
        angle  = np.degrees(np.arctan2(dy, dx))
        # create a matrix to rotate the image, the left eye is the rotation point
        M      = cv2.getRotationMatrix2D(tuple(le), angle, 1.0)
        # take the width and the height of the image
        h, w   = img.shape[:2]
        # rotate the image
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
        return rotated, M

    def crop(self, img, box):
        top, right, bottom, left = box
        #get height and width
        h, w     = bottom - top, right - left
        #compute padding such that the cropped image will be a square
        pad      = int(self.margin * max(h, w))
        #avoid getting outside of the original image
        y1 = max(0, top - pad)
        y2 = min(img.shape[0], bottom + pad)
        x1 = max(0, left - pad)
        x2 = min(img.shape[1], right + pad)
        #crop (the image from y1 to y2 and from x1 to x2)
        return img[y1:y2, x1:x2]

    def resize(self, img):
        return cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)

    def process_folder(self, user: str, raw_root=RAW_FACE_DIR, proc_root=PROC_FACE_DIR):
        #go only in the directory of the current user
        for person in sorted(raw_root.iterdir()):
            if not person.is_dir() or person.name != user:
                continue
            dst = proc_root / person.name
            #create the directory
            dst.mkdir(parents=True, exist_ok=True)
            #iterate through photos
            for src in person.glob("*.jpg"):
                img = cv2.imread(str(src))
                #skip if the image is unreadable
                if img is None:
                    continue
                #convert from bgr 2 rgb for compatibility with face_recognition
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #detect box and landmarks
                box, lm = self.detect(rgb)
                if box is None:
                    continue
                #get coordinates of eyes
                le = np.mean(lm["left_eye"], axis=0)
                re = np.mean(lm["right_eye"], axis=0)
                #align the image
                aligned,M = self.align(rgb, le, re)
                #create an array compatible with transform, containing the coordinates of the box
                pts = np.array([[[box[3], box[0]], [box[1], box[0]], [box[1], box[2]], [box[3], box[2]]]], dtype=np.float32)
                #rotate the box to match the rotated image
                pts_w = cv2.transform(pts, M)[0]  # remove dimension, extract only points
                #extract x and y points
                ys, xs = pts_w[:,1], pts_w[:,0]
                #save the coordinates of the new box
                new_box = (int(ys.min()), int(xs.max()), int(ys.max()), int(xs.min()))
                #crop the picture
                face = self.crop(aligned, new_box)
                #convert back to bgr
                face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                #resize
                out = self.resize(face_bgr)
                #declare the filename's path
                fn = dst / src.name
                #save the image to that path
                ok = cv2.imwrite(str(fn), out)
                status = "Success" if ok else "Failed"
                print(f"{status}: Writing preprocessed images_raw to {fn}")


if __name__ == "__main__":
    import sys
    user = sys.argv[1]
    FacePreprocessor().process_folder(user)


