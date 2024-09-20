**RubiksCubeSolver**
This project is an advanced Rubik's Cube solver that uses computer vision and image processing techniques to detect the state of a Rubik's Cube and provide step-by-step solving instructions.

**Features**
- Real-time cube face detection using OpenCV
- Color recognition and mapping of cube state
- Solving algorithm implementation using the Kociemba two-phase algorithm
- Interactive GUI with visual instructions for cube manipulation
- Video capture and output for documenting the solving process

**How it works**
- Face Detection: The program uses computer vision techniques to detect and isolate the Rubik's Cube faces from video input.
- Color Recognition: Each detected face is analyzed to determine the color of individual squares, creating a digital representation of the cube's state.
- Cube Solving: The Kociemba algorithm is employed to generate a sequence of moves that will solve the cube from its current state.
- User Guidance: The program provides visual cues and instructions to guide the user through the solving process, showing which face to turn and in which direction.
- Video Output: The entire solving process is recorded and saved as a video file for later review or sharing.

**Preview**
--
![frntimg](https://github.com/user-attachments/assets/44173e13-7a8f-4e36-91e9-0d8197823979)

--
![topfce](https://github.com/user-attachments/assets/19261c68-eba8-4765-a6a2-5a29ee400a4c)

--
![move](https://github.com/user-attachments/assets/4c344406-90bb-4b4e-884a-49af45453848)
