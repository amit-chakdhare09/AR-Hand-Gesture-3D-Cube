import cv2
import mediapipe as mp
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame
from pygame.locals import *
from typing import Optional, Tuple
import time

class ARCubeController:
    def __init__(self):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=2  # Support two hands
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Cube parameters
        self.scale = 2.0
        
        # Smoothing parameters
        self.smooth_factor = 0.15
        self.target_scale = 2.0
        
        # Control hands
        self.control_hand = None  
        self.rotation_hand = None  
        
        # Rotation
        self.rotation_x = 20
        self.rotation_y = 30
        self.target_rotation_x = 20
        self.target_rotation_y = 30
        
        # Rotation sensitivity
        self.rotation_x_sensitivity = 180
        self.rotation_y_sensitivity = 360
        
        # Scale sensitivity (palm openness)
        self.scale_min = 0.5
        self.scale_max = 5.0
        
        # Get screen resolution
        self.screen_info = pygame.display.Info()
        self.screen_width = self.screen_info.current_w
        self.screen_height = self.screen_info.current_h
        
    def calculate_palm_openness(self, hand_landmarks):
        """Calculate how open the palm is (0 = closed fist, 1 = fully open)"""
        # Calculate distances between fingertips and palm center
        palm_center = hand_landmarks[0]  # Wrist as reference
        
        # Get all fingertips
        fingertips = [
            hand_landmarks[4],   # Thumb
            hand_landmarks[8],   # Index
            hand_landmarks[12],  # Middle
            hand_landmarks[16],  # Ring
            hand_landmarks[20]   # Pinky
        ]
        
        # Calculate average distance from palm center to fingertips
        total_distance = 0
        for tip in fingertips:
            dx = tip.x - palm_center.x
            dy = tip.y - palm_center.y
            distance = np.sqrt(dx**2 + dy**2)
            total_distance += distance
        
        avg_distance = total_distance / len(fingertips)
        
        # Normalize to 0-1 range (these values are empirically determined)
        # Typical values: closed fist ~0.15, open palm ~0.35
        min_distance = 0.10
        max_distance = 0.40
        
        openness = (avg_distance - min_distance) / (max_distance - min_distance)
        openness = max(0.0, min(1.0, openness))  # Clamp to 0-1
        
        return openness
    
    def smooth_value(self, current, target, factor):
        """Apply exponential smoothing"""
        return current + (target - current) * factor
    
    def get_hand_position(self, hand_landmarks):
        """Get normalized hand position (x, y)"""
        wrist = hand_landmarks[0]
        return wrist.x, wrist.y
    
    def update_rotation_from_hand(self, hand_landmarks):
        """Update rotation based on hand position"""
        x, y = self.get_hand_position(hand_landmarks)
        
        # X position controls Y rotation (horizontal)
        self.target_rotation_y = (x - 0.5) * self.rotation_y_sensitivity
        
        # Y position controls X rotation (vertical)
        # Inverted: hand up = look down from top, hand down = look up from bottom
        self.target_rotation_x = (0.5 - y) * self.rotation_x_sensitivity
    
    def update_scale_from_palm(self, hand_landmarks):
        """Update scale based on palm openness"""
        openness = self.calculate_palm_openness(hand_landmarks)
        
        # Map openness (0-1) to scale range
        self.target_scale = self.scale_min + (openness * (self.scale_max - self.scale_min))
    
    def process_hands(self, frame):
        """Process hand landmarks and update cube"""
        img_h, img_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        left_hand = None
        right_hand = None
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                                   results.multi_handedness):
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                label = handedness.classification[0].label
                
                if label == "Left":
                    left_hand = hand_landmarks.landmark
                elif label == "Right":
                    right_hand = hand_landmarks.landmark
            
            # Determine control scheme
            if left_hand and right_hand:
                # Both hands: left controls size, right controls rotation
                self.control_hand = "left"
                self.rotation_hand = "right"
                
                self.update_scale_from_palm(left_hand)
                self.update_rotation_from_hand(right_hand)
                
            elif left_hand:
                # Only left hand: controls size
                self.control_hand = "left"
                self.rotation_hand = None
                
                self.update_scale_from_palm(left_hand)
                
            elif right_hand:
                # Only right hand: controls rotation
                self.control_hand = None
                self.rotation_hand = "right"
                
                self.update_rotation_from_hand(right_hand)
        else:
            self.control_hand = None
            self.rotation_hand = None
        
        # Apply smoothing
        self.scale = self.smooth_value(self.scale, self.target_scale, self.smooth_factor)
        self.rotation_x = self.smooth_value(self.rotation_x, self.target_rotation_x, self.smooth_factor)
        self.rotation_y = self.smooth_value(self.rotation_y, self.target_rotation_y, self.smooth_factor)
        
        return frame
    
    def draw_cube(self):
        """Draw a 3D cube with current scale"""
        s = self.scale
        
        # Cube vertices
        vertices = [
            [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
            [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]
        ]
        
        # Cube edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Back face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Front face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
        ]
        
        # Cube faces with colors
        faces = [
            (0, 1, 2, 3, (1, 0, 0)),      # Back - Red
            (4, 5, 6, 7, (0, 1, 0)),      # Front - Green
            (0, 1, 5, 4, (0, 0, 1)),      # Bottom - Blue
            (2, 3, 7, 6, (1, 1, 0)),      # Top - Yellow
            (0, 3, 7, 4, (1, 0, 1)),      # Left - Magenta
            (1, 2, 6, 5, (0, 1, 1))       # Right - Cyan
        ]
        
        # Draw filled faces with transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glBegin(GL_QUADS)
        for face in faces:
            color = face[4]
            glColor4f(color[0], color[1], color[2], 0.7)
            for vertex_index in face[:4]:
                glVertex3fv(vertices[vertex_index])
        glEnd()
        
        glDisable(GL_BLEND)
        
        # Draw edges (wireframe)
        glColor3f(1, 1, 1)
        glLineWidth(3)
        glBegin(GL_LINES)
        for edge in edges:
            for vertex_index in edge:
                glVertex3fv(vertices[vertex_index])
        glEnd()


def init_opengl(width, height):
    """Initialize OpenGL settings"""
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, width / height, 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.1, 0.1, 0.1, 1.0)


def draw_info_text(controller, frame_width, frame_height):
    """Draw information overlay"""
    # Determine status text
    if controller.control_hand and controller.rotation_hand:
        status = "Both Hands Active"
        left_status = "Left: Size Control"
        right_status = "Right: Rotation"
    elif controller.control_hand == "left":
        status = "Left Hand: Size Control"
        left_status = "Open palm = Larger"
        right_status = "Close palm = Smaller"
    elif controller.rotation_hand == "right":
        status = "Right Hand: Rotation"
        left_status = "Move to rotate cube"
        right_status = ""
    else:
        status = "No Hands Detected"
        left_status = "Show your hand(s)"
        right_status = ""
    
    # Calculate palm openness percentage for display
    openness_percent = int(((controller.target_scale - controller.scale_min) / 
                           (controller.scale_max - controller.scale_min)) * 100)
    
    info_lines = [
        "AR 3D Cube Controller",
        f"Status: {status}",
        "",
        f"Cube Size: {controller.scale:.2f}",
        f"Palm Openness: {openness_percent}%",
        f"Rotation: X:{controller.rotation_x:.0f}° Y:{controller.rotation_y:.0f}°",
        "",
        "Controls:",
        left_status,
        right_status,
        "",
        "ONE HAND:",
        "  Left Hand - Size control",
        "    Open palm = Bigger",
        "    Close fist = Smaller",
        "  Right Hand - Rotation",
        "    Move around to rotate",
        "",
        "TWO HANDS:",
        "  Left - Size control",
        "  Right - Rotation control",
        "",
        "ESC - Exit"
    ]
    
    overlay = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    cv2.rectangle(overlay, (10, 10), (400, 520), (50, 50, 50), -1)
    
    y_offset = 30
    for line in info_lines:
        if line:
            font_scale = 0.6 if not line.startswith("  ") else 0.5
            thickness = 2 if line.startswith("AR") or line.startswith("Status") else 1
            color = (100, 255, 100) if line.startswith("Status") else (255, 255, 255)
            
            cv2.putText(overlay, line, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        y_offset += 24
    
    return overlay


def main():
    pygame.init()
    
    controller = ARCubeController()
    
    # Fullscreen mode
    display = pygame.display.set_mode((controller.screen_width, controller.screen_height), 
                                     DOUBLEBUF | OPENGL | FULLSCREEN)
    
    pygame.display.set_caption('AR 3D Cube Controller')
    
    init_opengl(controller.screen_width, controller.screen_height)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    clock = pygame.time.Clock()
    
    print("=== AR 3D Cube Controller ===")
    print("\nSimplified Controls:")
    print("\nONE HAND MODE:")
    print("  Left Hand:")
    print("    • Open palm → Cube grows larger")
    print("    • Close fist → Cube shrinks smaller")
    print("  Right Hand:")
    print("    • Move left/right → Rotate horizontally")
    print("    • Move up → View from top")
    print("    • Move down → View from bottom")
    print("\nTWO HANDS MODE:")
    print("  Left Hand → Control size (open/close)")
    print("  Right Hand → Control rotation (move around)")
    print("\nPress ESC to exit\n")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Process hands
        frame = controller.process_hands(frame)
        
        # Create info overlay
        info_overlay = draw_info_text(controller, frame.shape[1], frame.shape[0])
        frame = cv2.addWeighted(frame, 1, info_overlay, 0.7, 0)
        
        # Convert to OpenGL texture
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Clear and setup 3D scene
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        glTranslatef(0.0, 0.0, -15)
        
        # Apply rotation
        glRotatef(controller.rotation_y, 0, 1, 0)
        glRotatef(controller.rotation_x, 1, 0, 0)
        
        # Draw cube
        controller.draw_cube()
        
        # Display webcam feed in corner
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, controller.screen_width, controller.screen_height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        
        feed_width = 400
        feed_height = 300
        
        glEnable(GL_TEXTURE_2D)
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame_rgb.shape[1], frame_rgb.shape[0],
                    0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        glColor3f(1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(controller.screen_width - feed_width - 20, 20)
        glTexCoord2f(1, 0)
        glVertex2f(controller.screen_width - 20, 20)
        glTexCoord2f(1, 1)
        glVertex2f(controller.screen_width - 20, 20 + feed_height)
        glTexCoord2f(0, 1)
        glVertex2f(controller.screen_width - feed_width - 20, 20 + feed_height)
        glEnd()
        
        glDisable(GL_TEXTURE_2D)
        glDeleteTextures([texture_id])
        
        glEnable(GL_DEPTH_TEST)
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        
        pygame.display.flip()
        clock.tick(60)
    
    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()