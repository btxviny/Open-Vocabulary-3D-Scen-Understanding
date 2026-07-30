import zmq
import pickle
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

def get_ros_time():
    """
    Returns a ROS-like timestamp as (secs, nsecs), using time.time().
    """
    now = time.time()
    secs = int(now)
    nsecs = int((now - secs) * 1e9)
    return secs, nsecs

def create_extrinsic_matrix(pose):
    """
    Create a 4x4 extrinsic matrix from the pose dictionary.
    """
    quat = [pose['qx'], pose['qy'], pose['qz'], pose['qw']]
    r = R.from_quat(quat)
    R_mat = r.as_matrix()
    t = np.array([[pose['x']], [pose['y']], [pose['z']]])
    extrinsic = np.vstack((np.hstack((R_mat, t)), [0, 0, 0, 1]))
    return extrinsic

def lidar_reciever_process(pipe):
    context = zmq.Context()
    lidar_socket = context.socket(zmq.REQ)
    lidar_socket.connect("tcp://192.168.1.133:5557")

    try:
        while True:
            lidar_socket.send(b"REQUEST_LIDAR")
            lidar_response = lidar_socket.recv()

            if lidar_response == b"No LiDAR data available":
                continue

            lidar_data = pickle.loads(lidar_response)
            pose = lidar_data["pose"]
            points = lidar_data["pointcloud"]

            if points is None or points.shape[0] == 0:
                continue

            points = points.astype(np.float64)
            if points.shape[1] != 3:
                points = points.reshape(-1, 3)

            secs, nsecs = get_ros_time()
            extrinsic = create_extrinsic_matrix(pose)

            msg = {
                "timestamp": {"secs": secs, "nsecs": nsecs},
                "pointcloud": points,
                "extrinsic_matrix": extrinsic.tolist()
            }

            pipe.send(msg)
            # print(f"[LIDAR] Sent message at ({secs}, {nsecs})")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("[LIDAR] Shutting down.")

    finally:
        lidar_socket.close()
