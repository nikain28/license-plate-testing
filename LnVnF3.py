import argparse
import threading
import time
from pathlib import Path
import blobconverter
import cv2
import depthai as dai
import numpy as np
from depthai_sdk.fps import FPSHandler
import pytesseract
import  imutils
import re

b=[]
count = 0

#plateCascade = cv2.CascadeClassifier("sg2.xml")
#plateC = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
#fire_cascade = cv2.CascadeClassifier('cascade.xml')

minArea = 200

parser = argparse.ArgumentParser()
parser.add_argument('-nd', '--no-debug', action="store_true", help="Prevent debug output")
parser.add_argument('-cam', '--camera', action="store_true",
                    help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)")
parser.add_argument('-vid', '--video', type=str,
                    help="Path to video file to be used for inference (conflicts with -cam)")
args = parser.parse_args()

if not args.camera and not args.video:
    raise RuntimeError(
        "No source selected. Use either \"-cam\" to run on RGB camera as a source or \"-vid <path>\" to run on video"
    )

debug = not args.no_debug
shaves = 6 if args.camera else 8


def frame_norm(frame, bbox):
    return (np.clip(np.array(bbox), 0, 1) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)

def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

# def create_pipeline():
#     print("Creating pipeline...")
#     pipeline = dai.Pipeline()
#
#     if args.camera:
#         print("Creating Color Camera...")
#         cam = pipeline.create(dai.node.ColorCamera)
#         cam.setPreviewSize(672, 384)
#         cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
#         cam.setInterleaved(False)
#         cam.setBoardSocket(dai.CameraBoardSocket.RGB)
#         cam_xout = pipeline.create(dai.node.XLinkOut)
#         cam_xout.setStreamName("cam_out")
#         cam.preview.link(cam_xout.input)
#
#     # NeuralNetwork
#     print("Creating Vehicle Detection Neural Network...")
#     veh_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
#     veh_nn.setConfidenceThreshold(0.5)
#     veh_nn.setBlobPath(blobconverter.from_zoo(name="vehicle-detection-adas-0002", shaves=shaves))
#     veh_nn.input.setQueueSize(1)
#     veh_nn.input.setBlocking(False)
#     veh_nn_xout = pipeline.create(dai.node.XLinkOut)
#     veh_nn_xout.setStreamName("veh_nn")
#     veh_nn.out.link(veh_nn_xout.input)
#
#     if args.camera:
#         cam.preview.link(veh_nn.input)
#     else:
#         veh_xin = pipeline.create(dai.node.XLinkIn)
#         veh_xin.setStreamName("veh_in")
#         veh_xin.out.link(veh_nn.input)
#
#     attr_nn = pipeline.create(dai.node.NeuralNetwork)
#     attr_nn.setBlobPath(blobconverter.from_zoo(name="vehicle-attributes-recognition-barrier-0039", shaves=shaves))
#     attr_nn.input.setBlocking(False)
#     attr_nn.input.setQueueSize(1)
#     attr_xout = pipeline.create(dai.node.XLinkOut)
#     attr_xout.setStreamName("attr_nn")
#     attr_nn.out.link(attr_xout.input)
#     attr_pass = pipeline.create(dai.node.XLinkOut)
#     attr_pass.setStreamName("attr_pass")
#     attr_nn.passthrough.link(attr_pass.input)
#     attr_xin = pipeline.create(dai.node.XLinkIn)
#     attr_xin.setStreamName("attr_in")
#     attr_xin.out.link(attr_nn.input)
#
#     print("Pipeline created.")
#     return pipeline
#
#running = True
license_detections = []
vehicle_detections = []
rec_results = []
attr_results = []
fire_results = []
frame_det_seq = 0
frame_seq_map = {}
veh_last_seq = 0
lic_last_seq = 0
decoded_text = []
#fire_stacked = None

if args.camera:
    fps = FPSHandler()
else:
    cap = cv2.VideoCapture(str(Path(args.video).resolve().absolute()))
    width = int(cap.get(3))
    height = int(cap.get(4))
    new_width = 640
    new_height = 480
    ret, frame = cap.read()
    resized_frame = cv2.resize(frame, (new_width, new_height))
    fps = FPSHandler(cap)


# def veh_thread(det_queue, attr_queue):
#     global vehicle_detections, veh_last_seq
#
#     while running:
#         try:
#             in_dets = det_queue.get()
#             vehicle_detections = in_dets.detections
#
#             orig_frame = frame_seq_map.get(in_dets.getSequenceNum(), None)
#             if orig_frame is None:
#                 continue
#
#             veh_last_seq = in_dets.getSequenceNum()
#
#             for detection in vehicle_detections:
#                 bbox = frame_norm(orig_frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
#                 cropped_frame = orig_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
#
#                 tstamp = time.monotonic()
#                 img = dai.ImgFrame()
#                 img.setTimestamp(tstamp)
#                 img.setType(dai.RawImgFrame.Type.BGR888p)
#                 img.setData(to_planar(cropped_frame, (72, 72)))
#                 img.setWidth(72)
#                 img.setHeight(72)
#                 attr_queue.send(img)
#
#             fps.tick('veh')
#         except RuntimeError:
#             continue
#
# def lic_thread(img):
#     numberPlates = plateCascade.detectMultiScale(img, 1.1, 5)
#     NPlate = plateC.detectMultiScale(img, 1.1, 4)
#
#     for (x, y, w, h) in numberPlates:
#         area = w * h
#
#         if area > minArea:
#
#             imgRoi = img[y:y + h, x:x + w]
#
#             imgRoi = imutils.resize(imgRoi, width=500)
#
#             gray = cv2.cvtColor(imgRoi, cv2.COLOR_BGR2GRAY)
#
#             img_inv = cv2.bitwise_not(gray)
#
#             # Perform OCR on the dilated image
#             text = pytesseract.image_to_string(img_inv,
#                                                config=f'--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
#                                                lang='eng')
#
#             if text is not None:
#                 with open('abc.txt', 'w') as f:
#                     f.write(text[:-1])
#
#                 with open('abc.txt', 'r') as file:
#                     lines = file.readlines()
#                     non_empty_lines = [line for line in lines if line.strip()]
#                     num_lines = len(non_empty_lines)
#
#             if num_lines > 1:
#
#                 h = round(1.3 * h)
#                 y = round(0.95 * y)
#
#                 imgRoi = img[y:y + h, x:x + w]
#
#                 imgRoi = imutils.resize(imgRoi, width=500)
#
#                 gray = cv2.cvtColor(imgRoi, cv2.COLOR_BGR2GRAY)
#
#                 img_inv = cv2.bitwise_not(gray)
#
#                 decoded_text = pytesseract.image_to_string(img_inv,
#                                                            config=f'--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
#                                                            lang='eng')
#
#                 pattern = r"^(?=.*[a-zA-Z])(?=.*\d).+$"
#
#                 if decoded_text is not None:
#                     decoded_text = re.sub(r'\W+', ' ', decoded_text)
#                     decoded_text = decoded_text.replace(" ", "")
#                     if decoded_text.isalnum() or decoded_text.isspace():
#                         if re.search(pattern, decoded_text):
#                             cv2.putText(img, decoded_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                                         (0, 0, 255), 2)
#                             decoded_text = decoded_text
#                             b.append("ROI2")
#                             b.append(decoded_text)
#                             cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#
#                 with open('number.xlsx', 'w') as f:
#                     if b is not None:
#                         for s in b:
#                             f.write(s)
#                             f.write("\n")
#             else:
#                 for (x2, y2, w2, h2) in NPlate:
#                     area = w * h
#
#                     w2 = round(1.1 * w2)
#
#                     if area > minArea:
#
#
#                         imgRoi = img[y2:y2 + h2, x2:x2 + w2]
#
#                         imgRoi = imutils.resize(imgRoi, width=500)
#
#                         gray = cv2.cvtColor(imgRoi, cv2.COLOR_BGR2GRAY)
#
#                         img_inv = cv2.bitwise_not(gray)
#
#                         decoded_text = pytesseract.image_to_string(img_inv,
#                                                                    config=f'--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
#                                                                    lang='eng')
#
#                         pattern = r"^(?=.*[a-zA-Z])(?=.*\d).+$"
#
#                         if decoded_text is not None:
#                             decoded_text = re.sub(r'\W+', ' ', decoded_text)
#                             decoded_text = decoded_text.replace(" ", "")
#                             if decoded_text.isalnum() or decoded_text.isspace():
#                                 if re.search(pattern, decoded_text):
#                                     cv2.putText(img, decoded_text, (x2, y2 - 10),
#                                                 cv2.FONT_HERSHEY_SIMPLEX, 1,
#                                                 (0, 0, 255), 2)
#                                     decoded_text = decoded_text
#                                     b.append("ROI")
#                                     b.append(decoded_text)
#                                     cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)
#
#
#                         with open('number.xlsx', 'w') as f:
#                             if b is not None:
#                                 for s in b:
#                                     f.write(s)
#                                     f.write("\n")
#             fps.tick('lic')
#
# def fire_thread(frame):
#     fire = fire_cascade.detectMultiScale(frame, 12, 3)  # test for fire detection
#     for (x, y, w, h) in fire:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255),
#                       2)  # highlight the area of image with fire
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         roi_gray = gray[y:y + h, x:x + w]
#         roi_color = frame[y:y + h, x:x + w]
#
#         # Convert to HSV color space
#         hsv_image = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
#
#         # Define the range of fire color in HSV
#         lower = np.array([0, 50, 50])
#         upper = np.array([10, 255, 255])
#
#         # Create a binary image where the fire color is white and everything else is black
#         mask = cv2.inRange(hsv_image, lower, upper)
#
#         # Apply morphological operations to remove noise and fill gaps
#         kernel = np.ones((5, 5), np.uint8)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#
#         # Calculate the percentage of white pixels in the image
#         total_pixels = mask.shape[0] * mask.shape[1]
#         white_pixels = np.sum(mask == 255)
#         white_percentage = (white_pixels / total_pixels) * 100
#
#         # Determine if the image has fire color based on the threshold percentage
#         if white_percentage >= 15:
#             cv2.putText(frame, "Fire Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                         (0, 255, 0), 2)
#         else:
#             cv2.putText(frame, "Smoke Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                         (0, 255, 0), 2)
#
#         fps.tick('fire')


# def attr_thread(q_attr, q_pass):
    # global attr_results, decoded_text
    #
    # while running:
    #     try:
    #         attr_data = q_attr.get()
    #     except RuntimeError:
    #         continue
    #
    #     colors = ["white", "gray", "yellow", "red", "green", "blue", "black"]
    #     types = ["car", "bus", "truck", "van"]
    #
    #     in_color = np.array(attr_data.getLayerFp16("color"))
    #     in_type = np.array(attr_data.getLayerFp16("type"))
    #
    #     color = colors[in_color.argmax()]
    #     color_prob = float(in_color.max())
    #     type = types[in_type.argmax()]
    #     type_prob = float(in_type.max())
    #
    #     attr_results = [(color, type, color_prob, type_prob)] + attr_results[:9]
    #
    #     fps.tick('attr')

def show_result(frame):
    cv2.putText(frame, 'Fire Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Object Detection', frame)
    cv2.waitKey(5000)


print("Starting pipeline...")
with dai.Device(create_pipeline()) as device:
    if args.camera:
        cam_out = device.getOutputQueue("cam_out", 1, True)
    else:
        veh_in = device.getInputQueue("veh_in")

    attr_in = device.getInputQueue("attr_in")
    veh_nn = device.getOutputQueue("veh_nn", 1, False)
    attr_nn = device.getOutputQueue("attr_nn", 1, False)
    attr_pass = device.getOutputQueue("attr_pass", 1, False)


    veh_t = threading.Thread(target=veh_thread, args=(veh_nn, attr_in))
    veh_t.start()
    attr_t = threading.Thread(target=attr_thread, args=(attr_nn, attr_pass))
    attr_t.start()


    def should_run():
        return cap.isOpened() if args.video else True


    def get_frame():
        global frame_det_seq

        if args.video:
            read_correctly, frame = cap.read()
            if read_correctly:
                frame_seq_map[frame_det_seq] = frame
                frame_det_seq += 1
            return read_correctly, frame
        else:
            in_rgb = cam_out.get()
            frame = in_rgb.getCvFrame()
            frame_seq_map[in_rgb.getSequenceNum()] = frame

            return True, frame


    # try:
        # while should_run():
        #     read_correctly, frame = get_frame()
        #
        #     if not read_correctly:
        #         break
        #
        #     for map_key in list(filter(lambda item: item <= min(lic_last_seq, veh_last_seq), frame_seq_map.keys())):
        #         del frame_seq_map[map_key]
        #
        #     fps.nextIter()
        #
        #     if not args.camera:
        #         tstamp = time.monotonic()
        #         veh_frame = dai.ImgFrame()
        #         veh_frame.setData(to_planar(frame, (300, 300)))
        #         veh_frame.setTimestamp(tstamp)
        #         veh_frame.setSequenceNum(frame_det_seq)
        #         veh_frame.setType(dai.RawImgFrame.Type.BGR888p)
        #         veh_frame.setWidth(300)
        #         veh_frame.setHeight(300)
        #         veh_frame.setData(to_planar(frame, (672, 384)))
        #         veh_frame.setWidth(672)
        #         veh_frame.setHeight(384)
        #         veh_in.send(veh_frame)
        #
        #     if debug:
        #         if count % 5 == 0:
        #             debug_frame = frame.copy()
        #             for detection in vehicle_detections:
        #                 bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        #                 cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        #
        #                 for attr_color, attr_type, color_prob, type_prob in attr_results:
        #                     cv2.putText(frame, f'{attr_color} {attr_type}', (bbox[0], bbox[1] - 5),
        #                                 cv2.FONT_HERSHEY_TRIPLEX, 0.5,
        #                                 (0, 255, 0))
        #                     # cv2.putText(debug_frame, attr_type, (15, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5,(0, 255, 0))
        #                     # cv2.putText(debug_frame, f"{int(color_prob * 100)}%", (150, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        #                     # cv2.putText(debug_frame, f"{int(type_prob * 100)}%", (150, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        #
        #             lic_thread(frame)
        #
        #             fire_thread(frame)

                # count += 1
                #     # Exit if 'q' is pressed
                # cv2.imshow("rgb", frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

    # except KeyboardInterrupt:
    #     pass

    running = False

    attr_t.join()
    veh_t.join()

print("FPS: {:.2f}".format(fps.fps()))
if not args.camera:
    cap.release()
    cv2.destroyAllWindows()
