
import cv2
import json
import numpy as np
import chess
import chess.engine
import chess.svg
from PIL import Image
import io
import random
import os
import sys
import cairosvg
import time
from pyniryo import *
import itertools
import math
import string

from ultralytics import YOLO

# ----------------------------
# 1. Load YOLO model
# ----------------------------
model = YOLO("my_model.pt")

# ----------------------------
# 2. Function: Extract centers
# ----------------------------
def extract_centers(results):
    centers = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = round((x1 + x2) / 2)
            cy = round((y1 + y2) / 2)
            cls = int(box.cls[0])
            centers.append((cx, cy, cls))
    return centers

# ----------------------------
# 3. Function: Match pieces between frames
# ----------------------------
def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def match_pieces(list1, list2, tolerance=20):
    matches = []
    unmatched1 = list1.copy()
    unmatched2 = list2.copy()
    used2 = set()
    
    for i, (cx1, cy1, cls1) in enumerate(list1):
        best_j = None
        best_d = float('inf')
        for j, (cx2, cy2, cls2) in enumerate(list2):
            if j in used2 or cls1 != cls2:
                continue
            d = dist((cx1, cy1), (cx2, cy2))
            if d < best_d:
                best_d = d
                best_j = j
        if best_j is not None and best_d <= tolerance:
            matches.append(((cx1, cy1, cls1), (list2[best_j][0], list2[best_j][1], cls1)))
            used2.add(best_j)
            unmatched1.remove(list1[i])
            unmatched2.remove(list2[best_j])
    
    return matches, unmatched1, unmatched2

# ----------------------------
# 4. Function: Map pixel to square using fixed board corners
# ----------------------------
def pixel_to_square_rotated(cx, cy, min_x, max_x, min_y, max_y):
    square_size_x = (max_x - min_x)/7
    square_size_y = (max_y - min_y)/7

    file_index = 7 - round((cx - min_x) / square_size_x)
    rank_index = round((cy - min_y) / square_size_y)
    
    file_index = max(0, min(7, file_index))
    rank_index = max(0, min(7, rank_index))
    
    file_char = string.ascii_lowercase[file_index]
    rank_char = str(rank_index + 1)
    return f"{file_char}{rank_char}"

##robot
tool_used = ToolID.GRIPPER_1
robot_ip_address = "192.168.1.35"

case_dim = 0.035

PoseA1 = PoseObject(x=0.176, y=0.127, z=0.115, roll=-3.105, pitch=1.525, yaw=-3.109)

def pick(robot, offsetX, offsetY):
    robot.pick_from_pose([PoseA1.x+offsetX, PoseA1.y-offsetY, PoseA1.z-0.085, PoseA1.roll, PoseA1.pitch, PoseA1.yaw])

def pose(robot, offsetX, offsetY):
    robot.place_from_pose([PoseA1.x+offsetX, PoseA1.y-offsetY, PoseA1.z-0.085, PoseA1.roll, PoseA1.pitch, PoseA1.yaw])

def home(robot):
    robot.move_to_home_pose()

def observation_pose(robot):
    robot.move_pose(0.2315,-0.0025,0.2774,-3.1416,1.2678,-3.1416)
    time.sleep(1)
    #mtx, dist = robot.get_camera_intrinsics()
    img_compressed = robot.get_img_compressed()
    img_raw = uncompress_image(img_compressed)
    results_first = model.predict(img_raw, conf=0.63)
    centers_first = extract_centers(results_first)

    #matches, unmatched1, unmatched2 = match_pieces(centers_first, centers_next)
    ref_frame = img_raw.copy()
    robot.play_sound("ready.wav")
    # Display results
    results_first[0].show()



    print("[DEBUG] Frame available display. robot played")
    return ref_frame, results_first, centers_first


ENGINE_PATH = r"stockfish-windows-x86-64-avx2.exe"
DEBUG_MODE = False  # Tekan 'd' untuk toggle ON/OFF

# === ENGINE ===
if not os.path.exists(ENGINE_PATH):
    print(f"[ERROR] File engine: {ENGINE_PATH}")
    sys.exit(1)

engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
print(f"[INFO] Stockfish  {ENGINE_PATH}")


# === ORIENTATION ===
files = 'abcdefgh'
ranks = '12345678'


# === HELPERS ===

def overlay_poly(frame, poly_pts, color, alpha=0.45):
    overlay = frame.copy()
    pts = np.array(poly_pts, np.int32)
    cv2.fillPoly(overlay, [pts], color)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def show_board(board, last_move=None):
    svg = chess.svg.board(board=board, lastmove=last_move, coordinates=True, size=450)
    png_data = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
    img = Image.open(io.BytesIO(png_data))
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.imshow("Board State", img_cv)
    cv2.waitKey(1)

def invertir_escaque(escaque):
    columnas = 'abcdefgh'
    filas = '12345678'
    col, fila = escaque[0], escaque[1]
    col_invertida = columnas[::-1][columnas.index(col)]
    fila_invertida = filas[::-1][filas.index(fila)]
    return col_invertida + fila_invertida

def invertir_movimiento(movimiento):
    origen = movimiento[:2]
    destino = movimiento[2:]
    return invertir_escaque(origen) + invertir_escaque(destino)

def square_points(cx, cy, d):
    return [
        (cx - d, cy - d),
        (cx + d, cy - d),
        (cx + d, cy + d),
        (cx - d, cy + d),
    ]



board = chess.Board()
ref_frame = None
last_move = None
comp_turn = False
move_history = []
contador = 0
show_board(board)

try:
    robot = NiryoRobot(robot_ip_address)
    robot.calibrate_auto()
    robot.update_tool()
    #robot.open_gripper(200,200)
    #robot.close_gripper(200, 5)  # tiny torque to reduce opening

    robot.set_brightness(85/100)
    robot.set_contrast(100 / 100)
    robot.set_saturation(50 / 100)

    ref_frame, results_first, centers_first  = observation_pose(robot)
    img_raw = ref_frame

    xs = [c[0] for c in centers_first]
    ys = [c[1] for c in centers_first]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    while not board.is_game_over():

        cv2.imshow("Chess Tracker", img_raw.copy())
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            
            if ref_frame is None:
                #mtx, dist = robot.get_camera_intrinsics()
                robot.move_pose(0.2315,-0.0025,0.2774,-3.1416,1.2678,-3.1416)
                img_compressed = robot.get_img_compressed()
                img_raw = uncompress_image(img_compressed)
                results_first = model.predict(img_raw, conf=0.63)
                centers_first = extract_centers(results_first)

                
                
                ref_frame = img_raw.copy()
                print("[DEBUG] Frame available display.")
            else:
                #mtx, dist = robot.get_camera_intrinsics()
                robot.move_pose(0.2315,-0.0025,0.2774,-3.1416,1.2678,-3.1416)
                img_compressed = robot.get_img_compressed()
                img_raw = uncompress_image(img_compressed)
                results_next = model.predict(img_raw, conf=0.63)
                centers_next = extract_centers(results_next)
                matches, unmatched1, unmatched2 = match_pieces(centers_first, centers_next)
            
                frame_raw = img_raw.copy()
                #cv2.imshow('ref',ref_frame)
                #cv2.imshow('raw',frame_raw)
                print("[DEBUG] Frame finally shot, processed...")
                # versi lebih universal (untuk putih gading dan hitam/coklat)
                if unmatched1:
                    from_sq_list = []
                    print("\nRemoved pieces:")
                    for t in unmatched1:
                        print(f"Piece class {t[2]} at {pixel_to_square_rotated(t[0], t[1], min_x, max_x, min_y, max_y)}")
                        if t[2] == 0:
                            print("black piece moved")
                            to_sq_possible = pixel_to_square_rotated(t[0], t[1], min_x, max_x, min_y, max_y)
                        else:
                            from_sq = pixel_to_square_rotated(t[0], t[1], min_x, max_x, min_y, max_y)
                            from_sq_list.append(pixel_to_square_rotated(t[0], t[1], min_x, max_x, min_y, max_y))
                            center_of_square_removed = square_points(t[0], t[1], d=20)
                            print("white piece moved")

                        
                if unmatched2:
                    to_sq_list = []
                    print("\nNew pieces:")
                    for t in unmatched2:
                        print(f"Piece class {t[2]} at {pixel_to_square_rotated(t[0], t[1], min_x, max_x, min_y, max_y)}")
                        to_sq_list.append(pixel_to_square_rotated(t[0], t[1], min_x, max_x, min_y, max_y))
                    if len(to_sq_list) > 1:
                        if "g1" in to_sq_list:
                            print("short castle")
                            castle=True
                            to_sq = "g1"
                            from_sq = "e1"
                            center_of_square_populated = square_points(t[0], t[1], d=20)
                        elif "c1" in to_sq_list:
                            print("long castle")
                            castle=True
                            to_sq = "c1"
                            from_sq = "e1"
                            center_of_square_populated = square_points(t[0], t[1], d=20)
                        else:
                            print("detection not sure, entering resilient mode for to_sq")
                            #to_sq = to_sq_list[0]
                            for to_sq_tmp in to_sq_list:
                                move_tmp = from_sq + to_sq_tmp
                                mv_test = chess.Move.from_uci(move_tmp)
                                if mv_test in board.legal_moves:
                                    to_sq = to_sq_tmp
                                    break
                                else: 
                                    print("move not recognized")
                                    to_sq = ""
                            if to_sq == "":
                                print("error on detection, please retry move")
                            else:
                                center_of_square_populated = square_points(t[0], t[1], d=20)

                            
                    else:
                        to_sq = to_sq_list[0]
                        center_of_square_populated = square_points(t[0], t[1], d=20)

                else:
                    to_sq = to_sq_possible

                #print("detection not sure, entering resilient mode")
                #to_sq = to_sq_list[0]
                if len(from_sq_list) > 1 and (not castle):
                    print("entering resilient mode for from_sq")
                    for from_sq_tmp in from_sq_list:
                        move_tmp = from_sq_tmp + to_sq
                        mv_test = chess.Move.from_uci(move_tmp)
                        if mv_test in board.legal_moves:
                            from_sq = from_sq_tmp
                            break
                        else: 
                            print("move not recognized")
                            from_sq = ""
                    if from_sq == "":
                        print("error on detection, please retry move")
                    else:
                        print("correct detection on resilient mode from_sq")

                if from_sq and to_sq:
                    move = from_sq + to_sq
                    try:
                        mv = chess.Move.from_uci(move)
                        if mv in board.legal_moves:
                            board.push(mv)
                            move_history.append(mv)
                            last_move = mv
                            print(f"[YOU] player main: {move}")
                            show_board(board, last_move)

                            # === HIGHLIGHT ===
                            try:
                                frame_high = overlay_poly(frame_raw.copy(), center_of_square_removed, (0, 255, 0), 0.5)
                                frame_high = overlay_poly(frame_high, center_of_square_populated, (0, 0, 255), 0.5)
                                #frame_high = draw_board_labels(frame_high)
                                cv2.imshow("Chess Tracker", frame_high)
                                cv2.waitKey(700) 
                            except Exception as e:
                                    print(f"[DEBUG]: {e}")

                            comp_turn = True
                        else:
                            print(f"[!] Invalid move: {move}")
                    except Exception as e:
                        print(f"[!] Error except: {e}")

                ref_frame = None

        # === Undo  ===
        if key == ord('u'):
            if move_history:
                mv = move_history.pop()
                board.pop()
                print(f"[UNDO] : {mv}")
                show_board(board)
            else:
                print("[INFO] ")

        if key == ord('U'):
            if len(move_history) >= 2:
                mv2 = move_history.pop()
                mv1 = move_history.pop()
                board.pop()
                board.pop()
                print(f"[UNDO] : {mv1}, {mv2}")
                show_board(board)
            else:
                print("[INFO]")

        # === COMPUTER TURN ===
        if comp_turn:
            result = engine.play(board, chess.engine.Limit(time=random.uniform(0.4, 0.9)))
            mv = result.move
            
            if board.is_capture(mv):
                print("Capture move:", mv)
                board.push(mv)
                move_history.append(mv)
                last_move = mv
                print(f"[AI] Computer main: {mv.uci()}")

                salida = invertir_movimiento(mv.uci())
                print(salida)  # Imprime: d2d4

                ######robot manipula pieza del rival
                #pick
                a = salida[2:4]  # or simply s[:2]

                l1 = ord(a[0])
                offsetY = (l1 - 96.6) * case_dim
                offsetX = (int(a[1]) - 1.2) * case_dim

                pick(robot, offsetX, offsetY)
                robot.move_pose(0.23,0.0,0.17,-3.1416,1.2678,-3.1416)


                ##drop
                #b = salida[2:4]
                l2 = ord('a')
                offsetY = (l2 - 96.6 - 6) * case_dim
                offsetX = (1 - 1.2) * case_dim

                pose(robot, offsetX, offsetY)

                ######robot manipula su propia pieza 
                #pick
                a = salida[0:2]  # or simply s[:2]

                l1 = ord(a[0])
                offsetY = (l1 - 96.6) * case_dim
                offsetX = (int(a[1]) - 1.2) * case_dim

                pick(robot, offsetX, offsetY)


                ##drop
                b = salida[2:4]
                l2 = ord(b[0])
                offsetY = (l2 - 96.6) * case_dim
                offsetX = (int(b[1]) - 1.2) * case_dim

                pose(robot, offsetX, offsetY)
                show_board(board, last_move)
                ref_frame, results_first, centers_first = observation_pose(robot)
            elif board.is_castling(mv):
                board.push(mv)
                move_history.append(mv)
                last_move = mv
                print(f"[AI] Computer main: {mv.uci()}")

                salida = invertir_movimiento(mv.uci())
                print(salida) 
                if chess.square_file(mv.to_square) == 6:
                    print("Kingside castling")
                    ######robot manipula su propia pieza rey
                    #pick
                    a = salida[0:2]  # or simply s[:2]

                    l1 = ord(a[0])
                    offsetY = (l1 - 96.6) * case_dim
                    offsetX = (int(a[1]) - 1.2) * case_dim

                    pick(robot, offsetX, offsetY)
                    


                    ##drop
                    b = salida[2:4]
                    l2 = ord(b[0])
                    offsetY = (l2 - 96.6) * case_dim
                    offsetX = (int(b[1]) - 1.2) * case_dim

                    pose(robot, offsetX, offsetY)

                    ######robot manipula su propia pieza torre
                    #pick
                    a = "a1"  # or simply s[:2]

                    l1 = ord(a[0])
                    offsetY = (l1 - 96.6) * case_dim
                    offsetX = (int(a[1]) - 1.2) * case_dim

                    pick(robot, offsetX, offsetY)
                    robot.move_pose(0.13,0.20,0.22,-3.1416,1.2678,-3.1416)


                    ##drop
                    b = "c1"
                    l2 = ord(b[0])
                    offsetY = (l2 - 96.6) * case_dim
                    offsetX = (int(b[1]) - 1.2) * case_dim

                    pose(robot, offsetX, offsetY)
                    show_board(board, last_move)
                    ref_frame, results_first, centers_first = observation_pose(robot)

                else:
                    print("Queenside castling")
                    ######robot manipula su propia pieza rey
                    #pick
                    a = salida[0:2]  # or simply s[:2]

                    l1 = ord(a[0])
                    offsetY = (l1 - 96.6) * case_dim
                    offsetX = (int(a[1]) - 1.2) * case_dim

                    pick(robot, offsetX, offsetY)


                    ##drop
                    b = salida[2:4]
                    l2 = ord(b[0])
                    offsetY = (l2 - 96.6) * case_dim
                    offsetX = (int(b[1]) - 1.2) * case_dim

                    pose(robot, offsetX, offsetY)

                    ######robot manipula su propia pieza torre
                    #pick
                    a = "h1"  # or simply s[:2]

                    l1 = ord(a[0])
                    offsetY = (l1 - 96.6) * case_dim
                    offsetX = (int(a[1]) - 1.2) * case_dim

                    pick(robot, offsetX, offsetY)
                    robot.move_pose(0.13,-0.20,0.22,-3.1416,1.2678,-3.1416)


                    ##drop
                    b = "e1"
                    l2 = ord(b[0])
                    offsetY = (l2 - 96.6) * case_dim
                    offsetX = (int(b[1]) - 1.2) * case_dim

                    pose(robot, offsetX, offsetY)
                    show_board(board, last_move)
                    ref_frame, results_first, centers_first = observation_pose(robot)
            else:
                print("Non-capture move:", mv)
                board.push(mv)
                move_history.append(mv)
                last_move = mv
                print(f"[AI] Computer main: {mv.uci()}")

                salida = invertir_movimiento(mv.uci())
                print(salida)  # Imprime: d2d4
                #pick
                a = salida[0:2]  # or simply s[:2]

                l1 = ord(a[0])
                offsetY = (l1 - 96.6) * case_dim
                offsetX = (int(a[1]) - 1.2) * case_dim

                pick(robot, offsetX, offsetY)


                ##drop
                b = salida[2:4]
                l2 = ord(b[0])
                offsetY = (l2 - 96.6) * case_dim
                offsetX = (int(b[1]) - 1.2) * case_dim

                pose(robot, offsetX, offsetY)
                show_board(board, last_move)
                ref_frame, results_first, centers_first = observation_pose(robot)
            
            #show_board(board, last_move)
            

            # === HIGHLIGHT ===
            try:
                move_str = mv.uci()
                frame_ai = overlay_poly(frame_raw.copy(), center_of_square_removed, (0, 255, 255), 0.45)  # kuning
                frame_ai = overlay_poly(frame_ai, center_of_square_populated, (0, 165, 255), 0.45)  # oranye-ish
                #frame_ai = draw_board_labels(frame_ai)
                cv2.imshow("Chess Tracker", frame_ai)
                cv2.waitKey(900)
            except Exception as e:
                    print(f"[DEBUG]highlight AI: {e}")

            comp_turn = False

        if key == ord('q'):
            print("[INFO] Keluar.")
            break

finally:
    cv2.destroyAllWindows()
    engine.quit()