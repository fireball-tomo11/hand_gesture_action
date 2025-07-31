import argparse
import sys
import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set GPIO port to BCM coding mode
GPIO.setmode(GPIO.BCM)

# Ignore the warning message
GPIO.setwarnings(False)

# ピンの設定
IN1, IN2, IN3, IN4 = 20, 21, 19, 26
ENA, ENB = 16, 13
EchoPin, TrigPin = 0, 1
TrackSensorPin1 = 5

# グローバル変数
pwm_ENA = None
pwm_ENB = None

def motor_init():
    """モーターの初期化を行う"""
    global pwm_ENA, pwm_ENB
    for pin in [ENA, IN1, IN2, ENB, IN3, IN4]:
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
    GPIO.output(ENA, GPIO.HIGH)
    GPIO.output(ENB, GPIO.HIGH)
    pwm_ENA = GPIO.PWM(ENA, 2000)
    pwm_ENB = GPIO.PWM(ENB, 2000)
    pwm_ENA.start(0)
    pwm_ENB.start(0)

def measure_distance():
    """距離を測定する関数"""
    GPIO.output(TrigPin, GPIO.LOW)
    time.sleep(0.000002)
    GPIO.output(TrigPin, GPIO.HIGH)
    time.sleep(0.000015)
    GPIO.output(TrigPin, GPIO.LOW)

    start_time = time.time()
    while not GPIO.input(EchoPin):
        if time.time() - start_time > 0.03:
            return -1
    t1 = time.time()
    while GPIO.input(EchoPin):
        if time.time() - t1 > 0.03:
            return -1
    t2 = time.time()
    time.sleep(0.01)
    return ((t2 - t1) * 340 / 2) * 100

def motor_control(in1, in2, in3, in4, delaytime, power):
    """モーター制御の一般化された関数"""
    GPIO.output(IN1, in1)
    GPIO.output(IN2, in2)
    GPIO.output(IN3, in3)
    GPIO.output(IN4, in4)
    pwm_ENA.ChangeDutyCycle(power)
    pwm_ENB.ChangeDutyCycle(power)
    time.sleep(delaytime)

# 以下の関数を motor_control を使用して簡略化
def run(delaytime, power): motor_control(GPIO.HIGH, GPIO.LOW, GPIO.HIGH, GPIO.LOW, delaytime, power)
def back(delaytime): motor_control(GPIO.LOW, GPIO.HIGH, GPIO.LOW, GPIO.HIGH, delaytime, power=60)
def left(delaytime): motor_control(GPIO.LOW, GPIO.LOW, GPIO.HIGH, GPIO.LOW, delaytime, power=60)
def right(delaytime): motor_control(GPIO.HIGH, GPIO.LOW, GPIO.LOW, GPIO.LOW, delaytime, power=60)
def spin_left(delaytime): motor_control(GPIO.LOW, GPIO.HIGH, GPIO.HIGH, GPIO.LOW, delaytime, power=60)
def spin_right(delaytime): motor_control(GPIO.HIGH, GPIO.LOW, GPIO.LOW, GPIO.HIGH, delaytime, power=60)
def brake(delaytime): motor_control(GPIO.LOW, GPIO.LOW, GPIO.LOW, GPIO.LOW, delaytime, power=60)

def figure_eight():
    """8の字旋回の関数"""
    for _ in range(1): 
        spin_left(1)
        run(1, 30)
        brake(1)
        time.sleep(0.5)
        spin_right(1)
        run(1, 30)  # ゆっくり回転するために1秒待機
        brake(1)
        time.sleep(0.5)  # ブレーキをかけて0.5秒待機

def detect_gesture(model: str, num_hands: int,
        min_hand_detection_confidence: float,
        min_hand_presence_confidence: float, min_tracking_confidence: float,
        camera_id: int, width: int, height: int, timeout: float = 5.0) -> str:
    """ジェスチャーを検出する関数"""
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    current_gesture = None
    start_time = time.time()

    def save_result(result: vision.GestureRecognizerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        nonlocal current_gesture
        if result.gestures:
            gesture = result.gestures[0][0]
            current_gesture = gesture.category_name

    try:
        base_options = python.BaseOptions(model_asset_path=model)
        options = vision.GestureRecognizerOptions(base_options=base_options,
                                                running_mode=vision.RunningMode.LIVE_STREAM,
                                                num_hands=num_hands,
                                                min_hand_detection_confidence=min_hand_detection_confidence,
                                                min_hand_presence_confidence=min_hand_presence_confidence,
                                                min_tracking_confidence=min_tracking_confidence,
                                                result_callback=save_result)
        recognizer = vision.GestureRecognizer.create_from_options(options)

        while cap.isOpened() and current_gesture is None:
            if time.time() - start_time > timeout:
                print("Gesture detection timed out")
                break

            success, image = cap.read()
            if not success:
                print("Failed to capture image from camera")
                break

            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            recognizer.recognize_async(mp_image, time.time_ns() // 1_000_000)

            if cv2.waitKey(1) == 27:  # ESCキーで終了
                print("Detection canceled by user")
                break

    except Exception as e:
        print(f"An error occurred during gesture detection: {e}")
    finally:
        recognizer.close()
        cap.release()
        cv2.destroyAllWindows()

    return current_gesture

def detect(current_command):
    """検出したジェスチャーに基づいて動作を行う関数"""
    try:
        TrackSensorValue = GPIO.input(TrackSensorPin1)

        if TrackSensorValue == 0:
            brake(1)
            print('anyone has been seated')
            return
        
        if current_command == "Thumb_Up":
            distance = measure_distance()
            if distance > 20:
                run(1, 40)
                brake(1)
                time.sleep(1.0) 
                print('run')
            else:
                brake(1)
                print('stop')

        elif current_command == "Open_Palm":
            brake(1)
            print('stop')

        elif current_command == "Pointing_Up":
            spin_left(1)
            brake(1)
            time.sleep(1.0)
            print('spin_left')

        elif current_command == "Closed_Fist":
            spin_right(1)
            brake(1)
            time.sleep(1.0)
            print('spin_left')

        elif current_command == "Victory":
            figure_eight()
            print('8round')

        else:
            brake(1)
            print('brake_mainloop')
    except Exception as e:
        print(f"An error occurred during detection: {e}")

def main():
    """メイン関数"""
    print('start')
    try:
        motor_init()
        GPIO.setup(TrackSensorPin1, GPIO.IN)
        GPIO.setup(EchoPin, GPIO.IN)
        GPIO.setup(TrigPin, GPIO.OUT)

        while True:
            gesture = detect_gesture('gesture_recognizer.task', 1, 0.5,
                                    0.5, 0.5, 0, 1280, 720, timeout=10.0)

            if gesture:
                print(f"Detected gesture: {gesture}")
                detect(gesture)
            else:
                print("No gesture detected")

            # 適切な待機時間を設定して、ループの速度を調整する
            time.sleep(1)

    except KeyboardInterrupt:
        print("Program terminated by user")
    except Exception as e:
        print(f"An error occurred in main: {e}")
    finally:
        if pwm_ENA:
            pwm_ENA.stop()
        if pwm_ENB:
            pwm_ENB.stop()
        GPIO.cleanup()

if __name__ == '__main__':
    main()
