import warnings

import cv2
import numpy
import torch

from nets import nn
from utils import util

warnings.filterwarnings("ignore")


def draw_line(image, x1, y1, x2, y2, index):
    w = 10
    h = 10
    color = (200, 0, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
    # Top left corner
    cv2.line(image, (x1, y1), (x1 + w, y1), color, 3)
    cv2.line(image, (x1, y1), (x1, y1 + h), color, 3)

    # Top right corner
    cv2.line(image, (x2, y1), (x2 - w, y1), color, 3)
    cv2.line(image, (x2, y1), (x2, y1 + h), color, 3)

    # Bottom right corner
    cv2.line(image, (x2, y2), (x2 - w, y2), color, 3)
    cv2.line(image, (x2, y2), (x2, y2 - h), color, 3)

    # Bottom left corner
    cv2.line(image, (x1, y2), (x1 + w, y2), color, 3)
    cv2.line(image, (x1, y2), (x1, y2 - h), color, 3)

    text = f'ID:{str(index)}'
    cv2.putText(image, text,
                (x1, y1 - 2),
                0, 1 / 2, (0, 255, 0),
                thickness=1, lineType=cv2.FILLED)


def main():
    size = 640
    model = torch.load('./weights/v8_n.pt', map_location='cuda')['model'].float()
    model.eval()
    model.half()
    reader = cv2.VideoCapture('./demo/demo.mp4')

    # Check if camera opened successfully
    if not reader.isOpened():
        print("Error opening video stream or file")
    fps = int(reader.get(cv2.CAP_PROP_FPS))
    bytetrack = nn.BYTETracker(fps)
    # Read until video is completed
    while reader.isOpened():
        # Capture frame-by-frame
        success, frame = reader.read()
        if success:
            boxes = []
            confidences = []
            object_classes = []

            image = frame.copy()
            shape = image.shape[:2]

            r = size / max(shape[0], shape[1])
            if r != 1:
                h, w = shape
                image = cv2.resize(image,
                                   dsize=(int(w * r), int(h * r)),
                                   interpolation=cv2.INTER_LINEAR)

            h, w = image.shape[:2]
            image, ratio, pad = util.resize(image, size)
            shapes = shape, ((h / shape[0], w / shape[1]), pad)
            # Convert HWC to CHW, BGR to RGB
            sample = image.transpose((2, 0, 1))[::-1]
            sample = numpy.ascontiguousarray(sample)
            sample = torch.unsqueeze(torch.from_numpy(sample), dim=0)

            sample = sample.cuda()
            sample = sample.half()  # uint8 to fp16/32
            sample = sample / 255  # 0 - 255 to 0.0 - 1.0

            # Inference
            with torch.no_grad():
                outputs = model(sample)

            # NMS
            outputs = util.non_max_suppression(outputs, 0.001, 0.7)
            for i, output in enumerate(outputs):
                detections = output.clone()
                util.scale(detections[:, :4], sample[i].shape[1:], shapes[0], shapes[1])
                detections = detections.cpu().numpy()
                for detection in detections:
                    x1, y1, x2, y2 = list(map(int, detection[:4]))
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(detection[4])
                    object_classes.append(detection[5])
            outputs = bytetrack.update(numpy.array(boxes),
                                       numpy.array(confidences),
                                       numpy.array(object_classes))
            if len(outputs) > 0:
                boxes = outputs[:, :4]
                identities = outputs[:, 4]
                object_classes = outputs[:, 6]
                for i, box in enumerate(boxes):
                    if object_classes[i] != 0:  # 0 is for person class (COCO)
                        continue
                    x1, y1, x2, y2 = list(map(int, box))
                    # get ID of object
                    index = int(identities[i]) if identities is not None else 0

                    draw_line(frame, x1, y1, x2, y2, index)
            cv2.imshow('Frame', frame.astype('uint8'))
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    reader.release()
    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
