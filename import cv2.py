import cv2
import numpy as np
import os

# 1. 동영상을 1초마다 캡처하기//first
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

        # 1초 간격으로 저장
        if frame_count % (fps * interval) == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_count}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Frames saved in {output_dir}")

# 2. 도로/인도 구분//2에 추가 ㅇㅇ 
def segment_road_and_sidewalk(image):
    """
    도로와 인도를 구분하는 함수.
    ROI 설정, k-means 기반 분할, HSV 필터링, 대비 강화.
    """
    # Step 1: 이미지 크기 조정 및 ROI 설정
    resized = cv2.resize(image, (640, 360))  # 처리 속도를 위해 크기 조정
    roi = resized[180:, :]  # 하단 절반만 관심 영역으로 설정 (도로 영역 예상)

    # Step 2: 대비 조정 및 밝기 강화
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)  # 명암 대비 개선
    enhanced = cv2.merge((l, a, b))
    roi_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Step 3: k-means 기반 이미지 분할
    Z = roi_enhanced.reshape((-1, 3))
    Z = np.float32(Z)

    # K-means 설정
    k = 5  # 클러스터 수
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 클러스터링 결과를 이미지로 변환
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    segmented_image = segmented.reshape(roi.shape)

    # Step 4: HSV 색상 기반 도로 영역 필터링
    hsv = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2HSV)

    # 도로로 예상되는 HSV 색상 범위
    road_lower = np.array([0, 0, 30])  # 더 넓은 범위로 설정
    road_upper = np.array([180, 70, 220])
    road_mask = cv2.inRange(hsv, road_lower, road_upper)

    # Step 5: 도로 마스크 적용
    road_segmented = cv2.bitwise_and(segmented_image, segmented_image, mask=road_mask)

    # Step 6: 최종 결과 반환
    result = cv2.addWeighted(roi, 0.7, road_segmented, 0.3, 0)
    return result


# 4. 횡단보도 찾기//3에 추가 ㅇㅇ 
def detect_crosswalk(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crosswalk_detected = image.copy()

    for contour in contours:
        area = cv2.contourArea(contour)
        if 1000 < area < 5000:  # 적절한 크기의 영역 필터링
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(crosswalk_detected, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return crosswalk_detected

# 메인 함수
def main(video_path, output_dir):
    extract_frames(video_path, output_dir)

    for frame_file in sorted(os.listdir(output_dir)):
        frame_path = os.path.join(output_dir, frame_file)
        frame = cv2.imread(frame_path)

        # 도로/인도 구분
        road_segmented = segment_road_and_sidewalk(frame)
        cv2.imshow("Road and Sidewalk Segmentation (Improved)", road_segmented)

        # 횡단보도 찾기
        crosswalk = detect_crosswalk(frame)
        cv2.imshow("Crosswalk Detection", crosswalk)

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "/Users/zeonons/Documents/opencv.mp4"  # 분석할 동영상 파일
    output_dir = "frames"         # 캡처한 프레임 저장 디렉토리
    main(video_path, output_dir)
