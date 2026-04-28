from ultralytics import YOLO
import cv2
import sys

def main():
    source = r".\testSnimke\video5.mp4"
    conf = 0.3
    display_size = [800, 600]
    output = "anotated.mp4"


    # učitavanje modela
    model = YOLO("yolo11n20cl.pt")

    # otvaranje video izvora
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Nije moguće otvoriti video izvor {source}")
        sys.exit(1)

    # proporcije videa
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()


    # priprema za spremanje obrađenog videa
    out = None
    if output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output, fourcc, fps, (width, height))

    # Model za praćenje

    results = model.track(source=source, stream=True, show=False, conf=conf, persist=True)

    # Boje
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLACK = (0, 0, 0)

    # Prikaz prozora
    display_width, display_height = display_size
    cv2.namedWindow("YOLO Aircraft Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Aircraft Detection", display_width, display_height)

    try:
        for r in results:
            frame = r.orig_img.copy()

            if r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if class_name.lower() == "civil":
                        box_color = GREEN
                        top_label = "Civil"
                        bottom_label = ""
                    else:
                        box_color = RED
                        top_label = "Military"
                        bottom_label = class_name

                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    # Font
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 2
                    thickness = 2

                    if top_label:
                        (text_width, text_height), baseline = cv2.getTextSize(
                            top_label, font, font_scale, thickness
                        )
                        text_y = max(y1 - 5, text_height + 5)
                        cv2.putText(frame, top_label, (x1, text_y), font,
                                    font_scale, BLACK, thickness)

                    if bottom_label:
                        (text_width, text_height), baseline = cv2.getTextSize(
                            bottom_label, font, font_scale, thickness
                        )
                        text_y = y2 + text_height + 5
                        if text_y + text_height < frame.shape[0]:  # within frame height
                            cv2.putText(frame, bottom_label, (x1, text_y), font,
                                        font_scale, BLACK, thickness)

                    if box.id is not None:
                        track_id = int(box.id[0])
                        id_text = f"ID:{track_id}"
                        (tw, th), _ = cv2.getTextSize(id_text, font, 1, 1)
                        cv2.putText(frame, id_text, (x1, y1 - text_height - 10),
                                    font, 1, BLACK, 1)

            cv2.putText(frame, f"Confidence threshold: {conf}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, BLACK, 2)


            # Spremanje videa
            if out is not None:
                out.write(frame)

            # Display
            display_frame = cv2.resize(frame, (display_width, display_height))
            cv2.imshow("YOLO Aircraft Detection", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Cleanup
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()