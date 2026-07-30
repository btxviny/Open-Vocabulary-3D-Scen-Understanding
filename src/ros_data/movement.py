import pygame
import zmq
import pickle
import time

DEADZONE = 0.1
MAX_VELOCITY = 300

def init_joystick():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("[MOVEMENT] No joystick detected.")
        return None
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    return joystick

def get_controller_input(joystick):
    pygame.event.pump()
    y_axis = joystick.get_axis(1)
    x_axis = joystick.get_axis(0)
    if abs(y_axis) < DEADZONE: y_axis = 0
    if abs(x_axis) < DEADZONE: x_axis = 0
    return {
        "left_speed": int((y_axis + x_axis) * MAX_VELOCITY),
        "right_speed": int((y_axis - x_axis) * MAX_VELOCITY)
    }

def movement_process():
    joystick = init_joystick()
    if joystick is None:
        return

    context = zmq.Context()
    movement_socket = context.socket(zmq.REQ)
    movement_socket.connect("tcp://192.168.1.133:5555")

    print("[MOVEMENT] Controller process started.")

    try:
        while True:
            command_data = get_controller_input(joystick)
            serialized_command = pickle.dumps(command_data)
            movement_socket.send(serialized_command)
            movement_socket.recv()  # Wait for ACK
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("[MOVEMENT] Stopped by user.")
    finally:
        pygame.quit()
        movement_socket.close()
