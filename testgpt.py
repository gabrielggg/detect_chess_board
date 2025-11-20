import cv2
import json

# Chessboard square naming (a8 → h1)
files = "abcdefgh"
ranks = "87654321"   # reversed rank order
square_names = [f + r for r in ranks for f in files]

squares = {}
current_square = []
square_index = 0

def mouse_callback(event, x, y, flags, param):
    global current_square, square_index

    if event == cv2.EVENT_LBUTTONDOWN:
        if square_index >= 64:
            print("All 64 squares already captured.")
            return
        
        current_square.append([int(x), int(y)])
        print(f"Square {square_names[square_index]} corner: {x}, {y}")

        # 4 corners clicked → save square
        if len(current_square) == 4:
            squares[square_names[square_index]] = current_square.copy()
            print(f"Saved {square_names[square_index]}: {current_square}")
            current_square.clear()
            square_index += 1

            if square_index == 64:
                print("All squares captured. Press 'q' to save & quit.")

# Load the board image
img = cv2.imread("empty.jpg")   # change to your image path

cv2.namedWindow("board")
cv2.setMouseCallback("board", mouse_callback)

while True:
    cv2.imshow("board", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

# Save JSON
with open("squares.json", "w") as f:
    json.dump(squares, f, indent=2)

print("Saved squares.json!")
