import cv2

def draw_box(frame, box, label: str = ""):
    x1, y1, x2, y2, score = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if label:
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
