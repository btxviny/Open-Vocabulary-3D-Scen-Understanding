from multiprocessing import Process, Pipe
from src.ros_data.image_reciever import image_reciever_process
from src.ros_data.lidar_reciever import lidar_reciever_process
from src.ros_data.movement import movement_process
from src.ros_data.match_and_store import match_and_store_process

if __name__ == "__main__":
    # Create pipes: sender_conn → receiver_conn
    img_send_conn, img_recv_conn = Pipe()         # image_reciever → match_and_store
    lidar_send_conn, lidar_recv_conn = Pipe()     # lidar_reciever → match_and_store

    # Start processes
    move = Process(target=movement_process)
    image_recv = Process(target=image_reciever_process, args=(img_send_conn,))
    lidar_proc = Process(target=lidar_reciever_process, args=(lidar_send_conn,))
    match_and_store = Process(target=match_and_store_process, args=(lidar_recv_conn, img_recv_conn))

    move.start()
    image_recv.start()
    lidar_proc.start()
    match_and_store.start()

    move.join()
    image_recv.join()
    lidar_proc.join()
    match_and_store.join()
