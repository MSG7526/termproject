import cv2
import numpy as np
import os

def extract_frames(video_path, output_dir, interval=1):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    saved_count = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % (fps * interval) == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_count}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Frames saved in {output_dir}")
    
def segment_road_and_sidewalk(image):

    resized = cv2.resize(image, (640, 360)) 
    roi = resized[180:, :] 

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)  
    enhanced = cv2.merge((l, a, b))
    roi_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    Z = roi_enhanced.reshape((-1, 3))
    Z = np.float32(Z)

    k = 5  
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    segmented_image = segmented.reshape(roi.shape)

    hsv = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2HSV)

    road_lower = np.array([0, 0, 30])  # 더 넓은 범위로 설정
    road_upper = np.array([180, 70, 220])
    road_mask = cv2.inRange(hsv, road_lower, road_upper)

    road_segmented = cv2.bitwise_and(segmented_image, segmented_image, mask=road_mask)

    result = cv2.addWeighted(roi, 0.7, road_segmented, 0.3, 0)
    return result


def detect_crosswalk(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crosswalk_detected = image.copy()

    for contour in contours:
        area = cv2.contourArea(contour)
        if 1000 < area < 5000: 
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(crosswalk_detected, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return crosswalk_detected

def main(video_path, output_dir):
    extract_frames(video_path, output_dir)

    for frame_file in sorted(os.listdir(output_dir)):
        frame_path = os.path.join(output_dir, frame_file)
        frame = cv2.imread(frame_path)

        road_segmented = segment_road_and_sidewalk(frame)
        cv2.imshow("Road and Sidewalk Segmentation (Improved)", road_segmented)

        crosswalk = detect_crosswalk(frame)
        cv2.imshow("Crosswalk Detection", crosswalk)

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "/Users/zeonons/Documents/opencv.mp4" 
    output_dir = "frames"     
    main(video_path, output_dir)
