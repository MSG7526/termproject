import cv2
import numpy as np

def process_frame(frame):
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    
    lower_road = np.array([0, 0, 0])
    upper_road = np.array([180, 255, 50])
    road_mask = cv2.inRange(hsv, lower_road, upper_road)
    
    
    lower_sidewalk1 = np.array([0, 100, 100])
    upper_sidewalk1 = np.array([10, 255, 255])
    lower_sidewalk2 = np.array([160, 100, 100])
    upper_sidewalk2 = np.array([180, 255, 255])
    sidewalk_mask1 = cv2.inRange(hsv, lower_sidewalk1, upper_sidewalk1)
    sidewalk_mask2 = cv2.inRange(hsv, lower_sidewalk2, upper_sidewalk2)
    sidewalk_mask = cv2.bitwise_or(sidewalk_mask1, sidewalk_mask2)

    
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    bright_white = cv2.bitwise_and(frame, frame, mask=white_mask)
    bright_white[np.where((bright_white != [0, 0, 0]).all(axis=2))] = [255, 255, 255]

   
    road_sidewalk_mask = cv2.bitwise_or(road_mask, sidewalk_mask)
    result = cv2.bitwise_and(frame, frame, mask=road_sidewalk_mask)

   
    result = cv2.addWeighted(result, 1, bright_white, 1, 0)

    
    edges = cv2.Canny(white_mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

    
    contours, _ = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if len(approx) >= 4 and area > 500: 
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.5 < aspect_ratio < 2.0:  
                cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return result

def main():
   
    video_path = input("비디오 파일 경로를 입력하세요: ")
    
    
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

       
        result = process_frame(frame)

        
        cv2.imshow('Original', frame)
        cv2.imshow('Road and Sidewalk Detection', result)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

