# **Documentation: Document Scanner Using OpenCV and Tkinter**

This Python application provides a simple Document Scanner tool using OpenCV and Tkinter. It allows users to scan documents either via webcam or by uploading an image. The program processes the document to showcase different transformations, including grayscale, thresholding, contours, and perspective correction.

---

## **Features**
1. **Webcam Integration**: 
   - Live video feed for real-time document scanning.
   - Press **`q`** to quit the webcam mode.

2. **Image Upload**:
   - Allows users to upload an image from their local device for processing.

3. **Document Processing**:
   - **Original Image**: Displays the original input image.
   - **Grayscale**: Converts the image to grayscale.
   - **Thresholding**: Applies binary thresholding.
   - **Contours**: Detects and highlights contours.
   - **Warp Perspective**: Warps the perspective of the detected document to A4 size.
   - **Warp Gray**: Grayscale version of the warped perspective.
   - **Adaptive Thresholding**: Applies adaptive thresholding to the warped grayscale image.

4. **User-Friendly Menu**:
   - A graphical menu implemented using `tkinter`, enabling users to:
     - Use the webcam for scanning.
     - Upload an image.
     - Exit the application.

---

## **How It Works**

### **1. Main Menu**
The application starts with a graphical menu implemented using `tkinter`. It offers the following options:
- **Use Webcam**: Starts the webcam and processes the video feed in real time.
- **Upload Image**: Opens a file dialog for selecting an image.
- **Exit**: Exits the application.

Code:
```python
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
```

---

### **2. Document Processing**
The `process_image` function processes the input image by applying the following transformations:
- **Grayscale Conversion**:
  ```python
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ```
- **Blurring and Edge Detection**:
  - Gaussian blur smooths the image.
  - Canny edge detection detects edges in the image.
  ```python
  img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
  img_canny = cv2.Canny(img_blur, 50, 150)
  ```
- **Contour Detection**:
  - Finds contours in the edge-detected image and identifies the largest quadrilateral, presumed to be the document.
  ```python
  contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  ```
- **Perspective Warping**:
  - Warps the perspective of the detected document to A4 dimensions.
  ```python
  matrix = cv2.getPerspectiveTransform(pts1, pts2)
  img_warp = cv2.warpPerspective(img, matrix, (width, height))
  ```
- **Adaptive Thresholding**:
  - Applies adaptive thresholding for enhanced readability of text.
  ```python
  img_adaptive_thresh = cv2.adaptiveThreshold(
      img_warp_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
  )
  ```

Stacked images are displayed for easy visualization using the `stack_images` function.

---

### **3. Webcam Integration**
The `start_webcam` function uses OpenCV's `VideoCapture` to process frames in real-time. Users can press **`q`** to exit the webcam mode.

Code:
```python
def start_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.resize(img, (640, 480))
        stacked_images = process_image(img)

        cv2.imshow("Document Scanner - Webcam", stacked_images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
```

---

### **4. Image Upload**
The `upload_image` function uses `tkinter.filedialog` to open a file dialog, allowing users to select an image for processing.

Code:
```python
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
    if file_path:
        img = cv2.imread(file_path)
        img = cv2.resize(img, (640, 480))
        stacked_images = process_image(img)
        cv2.imshow("Document Scanner - Uploaded Image", stacked_images)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
```

---

## **Helper Functions**

### **1. Contour Detection**
Identifies the largest quadrilateral in the image, which is presumed to be the document.

Code:
```python
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
```

### **2. Perspective Warping**
Transforms the document's perspective to make it appear like a scanned document.

Code:
```python
def warp_perspective(img, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    width, height = 420, 596  # A4 size dimensions
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, matrix, (width, height))
    return img_warp
```

---

## **How to Run the Application**
1. Install required dependencies:
   ```bash
   pip install opencv-python
   ```
2. Save the script as a `.py` file (e.g., `document_scanner.py`).
3. Run the script:
   ```bash
   python document_scanner.py
   ```
4. Choose an option from the menu:
   - Use Webcam
   - Upload Image
   - Exit

---

## **Dependencies**
- **OpenCV**: For image processing.
- **NumPy**: For numerical computations.
- **Tkinter**: For GUI and file dialog.

---

## **Keyboard Controls**
- **`q`**: Quit the webcam mode.

---

## **Limitations**
1. The program assumes the document is a quadrilateral.
2. Webcam quality may affect contour detection.
3. Image resizing may distort some images with non-standard aspect ratios.

---

