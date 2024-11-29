import cv2
import numpy as np
from tkinter import Tk, Button, filedialog


def get_contours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = np.array([])
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest


def reorder(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 2), dtype=np.int32)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points


def warp_perspective(img, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    width, height = 420, 596  # A4 size dimensions
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, matrix, (width, height))
    return img_warp


def stack_images(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    for x in range(rows):
        for y in range(cols):
            imgArray[x][y] = cv2.resize(imgArray[x][y], (width, height))
            if len(imgArray[x][y].shape) == 2:
                imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
    hor = [np.hstack(imgArray[x]) for x in range(rows)]
    ver = np.vstack(hor)
    return ver


def process_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 50, 150)
    img_contour = img.copy()

    biggest = get_contours(img_canny)
    if biggest.size != 0:
        cv2.drawContours(img_contour, [biggest], -1, (0, 255, 0), 10)
        img_warp = warp_perspective(img, biggest)
        img_warp_gray = cv2.cvtColor(img_warp, cv2.COLOR_BGR2GRAY)
        img_adaptive_thresh = cv2.adaptiveThreshold(img_warp_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        img_warp = img.copy()
        img_warp_gray = img_gray.copy()
        img_adaptive_thresh = img_gray.copy()

    img_thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)[1]

    # Stack images for display
    image_array = [
        [img, img_gray, img_thresh, img_contour],
        [img_warp, img_warp_gray, img_adaptive_thresh, img_canny]
    ]
    return stack_images(0.6, image_array)


def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
    if file_path:
        img = cv2.imread(file_path)
        img = cv2.resize(img, (640, 480))  # Resize for consistency
        stacked_images = process_image(img)
        cv2.imshow("Document Scanner - Uploaded Image", stacked_images)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def start_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.resize(img, (640, 480))  # Resize image for consistent display
        stacked_images = process_image(img)

        cv2.imshow("Document Scanner - Webcam", stacked_images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main_menu():
    root = Tk()
    root.title("Document Scanner")
    root.geometry("300x150")

    btn_webcam = Button(root, text="Use Webcam", command=lambda: [root.destroy(), start_webcam()])
    btn_upload = Button(root, text="Upload Image", command=lambda: [root.destroy(), upload_image()])
    btn_exit = Button(root, text="Exit", command=root.destroy)

    btn_webcam.pack(pady=10)
    btn_upload.pack(pady=10)
    btn_exit.pack(pady=10)

    root.mainloop()


# Run the main menu
main_menu()
