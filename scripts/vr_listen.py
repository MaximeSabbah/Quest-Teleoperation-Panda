import struct
import socket
from dataclasses import dataclass
import numpy as np


@dataclass
class ControllerPose:
    matrix: np.ndarray  # shape (4,4)

@dataclass
class ControllerInput:
    joystick: tuple[float, float]
    index_trigger: float
    hand_trigger: float
    buttons: dict[str, bool]

@dataclass
class VRFrame:
    head_pose: ControllerPose
    left_pose: ControllerPose
    right_pose: ControllerPose
    left_input: ControllerInput
    right_input: ControllerInput


def decode_packet(data: bytes) -> VRFrame:
    floats = struct.unpack('<66f', data)
    f = iter(floats)

    def next_mat():
        return np.array([ [next(f), next(f), next(f), next(f)],
                          [next(f), next(f), next(f), next(f)],
                          [next(f), next(f), next(f), next(f)],
                          [next(f), next(f), next(f), next(f)] ], dtype=np.float32)

    head_pose  = ControllerPose(next_mat())
    left_pose  = ControllerPose(next_mat())
    right_pose = ControllerPose(next_mat())

    # Analog values
    left_joy   = (next(f), next(f))
    right_joy  = (next(f), next(f))
    left_index, right_index, left_grip, right_grip = next(f), next(f), next(f), next(f)

    # Buttons
    btnA, btnB, btnX, btnY, thumbL, thumbR, trigL, trigR, gripL, gripR = [int(next(f)) for _ in range(10)]

    left_buttons = {
        "X": bool(btnX),
        "Y": bool(btnY),
        "Thumbstick": bool(thumbL),
        "TriggerButton": bool(trigL),
        "GripButton": bool(gripL),
    }
    right_buttons = {
        "A": bool(btnA),
        "B": bool(btnB),
        "Thumbstick": bool(thumbR),
        "TriggerButton": bool(trigR),
        "GripButton": bool(gripR),
    }

    left_input  = ControllerInput(left_joy, left_index, left_grip, left_buttons)
    right_input = ControllerInput(right_joy, right_index, right_grip, right_buttons)

    return VRFrame(head_pose, left_pose, right_pose, left_input, right_input)


# Example UDP receiver loop
UDP_IP = "0.0.0.0"
UDP_PORT = 5000
print(socket.AF_INET, socket.SOCK_DGRAM)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
print(f"Listening on {UDP_PORT}...")

while True:
    print("IN Here")
    data, addr = sock.recvfrom(4096)
    print(f"Received message: {data} from {addr}")
    frame = decode_packet(data)
    print("Head position:", frame.head_pose.matrix[:3, 3])
    print("Left joystick:", frame.left_input.joystick)
    print("Right A button:", frame.right_input.buttons['A'])
