import zmq
import cv2
import numpy as np
import msgpack
import time

def get_ros_timestamp():
    now = time.time()
    secs = int(now)
    nsecs = int((now - secs) * 1e9)
    return secs, nsecs

def image_reciever_process(img_send_conn):
    context = zmq.Context()
    image_socket = context.socket(zmq.REQ)
    image_socket.connect("tcp://192.168.1.133:5556")

    print("[IMAGE RECIEVER] Ready to fetch and forward frames.")

    try:
        while True:
            image_socket.send(b"REQUEST_IMAGE")
            response = image_socket.recv()

            if response != b"No image available":
                image_data = msgpack.unpackb(response, raw=False)

                secs, nsecs = get_ros_timestamp()
                timestamp = {"secs": secs, "nsecs": nsecs}

                # Decode color image
                color_img = cv2.imdecode(np.frombuffer(image_data["color"], np.uint8), cv2.IMREAD_COLOR)

                # Check and decode depth image
                depth_img = None
                if "depth" in image_data:
                    depth_img = cv2.imdecode(np.frombuffer(image_data["depth"], np.uint8), cv2.IMREAD_UNCHANGED)

                # Prepare dictionary with both images and metadata
                data_to_send = {
                    "timestamp": timestamp,
                    "color_image": color_img,
                    "depth_image": depth_img,
                    "height": color_img.shape[0],
                    "width": color_img.shape[1]
                }

                img_send_conn.send(data_to_send)

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("[IMAGE RECIEVER] KeyboardInterrupt received. Exiting.")

    image_socket.close()
