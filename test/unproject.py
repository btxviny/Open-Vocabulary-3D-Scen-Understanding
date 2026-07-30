import time
import numpy as np

def unproject_process(conn_from_sam, conn_from_depth, conn_to_reconstruct):

    print("[UNPROJECT] Listening for SAM and depth data...")

    fx = 386.99176025390625
    fy = 386.99176025390625
    ppx = 320.9659118652344
    ppy = 241.470703125

    transformation_matrix = np.array([
        [0.894078, 0.012117, 0.447747, 0.008695],
        [-0.446418, -0.057450, 0.892979, 0.127623],
        [0.036543, -0.998275, -0.045955, 0.761847],
        [0.000000, 0.000000, 0.000000, 1.000000]
    ])

    def unproject_to_3d(x, y, depth, fx, fy, ppx, ppy):
        x_norm = (x - ppx) / fx
        y_norm = (y - ppy) / fy
        z = depth
        return [x_norm * z, y_norm * z, z]

    def apply_transformation(points, matrix):
        points = np.asarray(points)
        ones = np.ones((points.shape[0], 1))
        homogeneous = np.hstack((points, ones))
        transformed = homogeneous @ matrix.T
        return transformed[:, :3]

    depth_dict = {}
    latest_matched_ts = None
    stride = 500

    votes = {}
    last_seen_ids = set()

    while True:
        if conn_from_depth.poll():
            depth_msg = conn_from_depth.recv()
            ts = (int(depth_msg['timestamp']['secs']), int(depth_msg['timestamp']['nsecs']))
            depth_dict[ts] = depth_msg

        if conn_from_sam.poll():
            sam_msg = conn_from_sam.recv()
            ts_sam = (int(sam_msg['timestamp']['secs']), int(sam_msg['timestamp']['nsecs']))

            if ts_sam in depth_dict:
                matched_depth = depth_dict.pop(ts_sam)
                latest_matched_ts = ts_sam
                depth_dict = {ts: msg for ts, msg in depth_dict.items() if ts > latest_matched_ts}

                depth_np = np.array(matched_depth['depth_image'], dtype=np.float32).reshape(
                    matched_depth['height'], matched_depth['width']
                ) / 1000.0

                pred_mask = np.array(sam_msg['mask'], dtype=np.uint8).reshape(
                    matched_depth['height'], matched_depth['width']
                )

                object_ids = np.unique(pred_mask)
                object_ids = object_ids[object_ids != 0]

                current_seen_ids = set(object_ids)
                obj_pointclouds = []

                for obj_id in object_ids:
                    mask = (pred_mask == obj_id)
                    ys, xs = np.where(mask)

                    points_local = []

                    for i in range(0, len(xs), stride):
                        y = ys[i]
                        x = xs[i]
                        z = depth_np[y, x]
                        if 0 < z <= 6:
                            pt = unproject_to_3d(x, y, z, fx, fy, ppx, ppy)
                            points_local.append(pt)

                    if not points_local:
                        continue

                    points_global = apply_transformation(points_local, transformation_matrix)

                    # Voting logic
                    if obj_id in last_seen_ids:
                        votes[obj_id] = votes.get(obj_id, 0) + 1
                    else:
                        votes[obj_id] = 1

                    obj_pointclouds.append({
                        "object_id": int(obj_id),
                        "bbox": votes[obj_id] > 3,
                        "3d_points": points_global.tolist()
                    })

                last_seen_ids = current_seen_ids

                if obj_pointclouds:
                    conn_to_reconstruct.send({
                        "timestamp": ts_sam,
                        "objects": obj_pointclouds
                    })

        time.sleep(0.01)
