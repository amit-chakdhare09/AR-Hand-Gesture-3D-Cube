# AR Hand Gesture Controlled 3D Cube 🖐️🧊

This project demonstrates a real-time **Augmented Reality (AR) 3D Cube Controller** using **hand gestures** captured through a webcam.  
The system uses **MediaPipe Hands** for hand tracking and **OpenGL** for rendering a 3D cube that can be rotated and scaled dynamically.

---

## 🔥 Features

- Real-time hand tracking using MediaPipe
- One-hand and two-hand interaction modes
- Smooth rotation and scaling of a 3D cube
- Palm openness–based scaling
- Position-based rotation control
- OpenGL-rendered transparent 3D cube
- Webcam feed overlay
- Fullscreen AR experience

---

## 🖐️ Gesture Controls

### One-Hand Mode
**Left Hand**
- Open palm → Increase cube size
- Closed fist → Decrease cube size

**Right Hand**
- Move left/right → Rotate cube horizontally
- Move up/down → Rotate cube vertically

### Two-Hand Mode
- Left hand → Size control
- Right hand → Rotation control

---

## 🧠 Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- PyOpenGL
- Pygame

---

## 🏗️ System Architecture

Webcam Input  
↓  
Hand Detection (MediaPipe)  
↓  
Gesture Interpretation  
↓  
Transformation Logic (Rotation & Scale)  
↓  
OpenGL 3D Rendering  

---
