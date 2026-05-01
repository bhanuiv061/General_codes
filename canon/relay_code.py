# ############################## watch dog working ########################

# import time
# import requests
# import pyhid_usb_relay

# FLASK_URL = "http://172.16.36.67:5000/health"
# # FLASK_URL = "http://192.168.4.53:5000/health"
# CHECK_INTERVAL = 2

# relay_board = pyhid_usb_relay.find()
# current_state = None

# def set_relay(state):
#     global current_state
#     if state == current_state:
#         return

#     if state == "ON":
#         relay_board.set_state(1, True)  # turn ON channel 1
#         print("[RELAY STATUS] ON")
#     else:
#         relay_board.set_state(1, False) # turn OFF channel 1
#         print("[RELAY STATUS] OFF")

#     current_state = state

# def flask_is_running():
#     try:
#         r = requests.get(FLASK_URL, timeout=2)
#         return r.status_code == 200
#     except requests.RequestException:
#         return False

# if __name__ == "__main__":
#     set_relay("OFF")
#     while True:
#         if flask_is_running():
#             set_relay("ON")
#         else:
#             set_relay("OFF")
#         time.sleep(CHECK_INTERVAL)


import pyhid_usb_relay

relay = pyhid_usb_relay.find()

print("Manual Relay Control")
print("Type: on | off | exit")

while True:
    cmd = input(">> ").strip().lower()

    if cmd == "on":
        relay.set_state(1, True)  # CH1 ON
        print("Relay CH1: ON")

    elif cmd == "off":
        relay.set_state(1, False)  # CH1 OFF
        print("Relay CH1: OFF")

    elif cmd == "exit":
        relay.set_state(1, False)
        print("Exiting...")
        break

    else:
        print("Invalid command. Use: on | off | exit")


# import hid

# h = hid.device()
# h.open(0x16C0, 0x05DF)  # VID, PID (example)

# # Turn Relay 1 ON
# h.write([0x00, 0xFF, 0x01, 0x01])

# # Turn Relay 1 OFF
# h.write([0x00, 0xFD, 0x01, 0x00])

# h.close()


# import serial
# import time

# set = serial.Serial('COM5', 9600, timeout=1)
# time.sleep(2)

# # Relay ON (example command)
# set.write(b'\xA0\x01\x01\xA2')  # Relay 1 ON
# time.sleep(2)

# # Relay OFF
# set.write(b'\xA0\x01\x00\xA1')  # Relay 1 OFF

# set.close()
