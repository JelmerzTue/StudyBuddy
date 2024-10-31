import csv
import threading

import mediapipe as mp

from FaceMeshProcessor import FaceMeshProcessor
from Liveplotter import LivePlotter
from BlinkDetector import BlinkDetector
from helpers import *

# Initialize face mesh and drawing utilities
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


def calibrate_features(calib_frame_count=25):
    """Perform calibration to get the neutral position."""
    ears, mars, pucs, moes = [], [], [], []
    calibration_completed = False

    cap = cv2.VideoCapture(CAPTURE_DEVICE_INDEX)

    while not calibration_completed:
        while cap.isOpened():
            image = capture_frame(cap)
            face_processor.process_image(image)
            image = face_processor.image
            if image is None:
                continue
            if face_processor.face_detected:
                face_processor.draw_head_pose_points(image)
                show_message(image, 'Ready for calibration')
                show_message(image, f'Please look at the center of your screen and press "c"', (20, 100), 0.7)
            else:
                show_message(image, 'Face Not Detected')
                show_message(image, 'Please make sure you are visible for the camera', (20, 100), 0.7)
            key = cv2.waitKey(5) & 0xFF
            if key == 27:  # Esc key
                return None

            elif key == ord('c'):  # 'c' key
                break

            cv2.imshow('Calibration', image)

        # Collect calibration data
        while len(ears) < calib_frame_count:
            image = capture_frame(cap)
            if image is None:
                continue

            ear, mar, puc, moe, _, image = face_processor.process_image(image)

            if face_processor.face_detected:
                ears.append(ear)
                mars.append(mar)
                pucs.append(puc)
                moes.append(moe)
                face_processor.draw_head_pose_points(image)
                show_message(image, 'Calibrating, please stay still', (20, 100))
            else:
                break  # Exit if face not detected

            cv2.imshow('Calibration', face_processor.image)
            key = cv2.waitKey(5) & 0xFF
            if key == 27:  # Esc key
                return None

        if len(ears) == calib_frame_count:
            calibration_completed = True  # Exit the outer loop if successful
        else:
            ears, mars, pucs, moes = [], [], [], []  # Reset data for a new attempt


    cv2.destroyAllWindows()
    ears = np.array(ears)
    mars = np.array(mars)
    pucs = np.array(pucs)
    moes = np.array(moes)

    return {
        "ears": [ears.mean(), ears.std()],
        "mars": [mars.mean(), mars.std()],
        "pucs": [pucs.mean(), pucs.std()],
        "moes": [moes.mean(), moes.std()],
    }


def initialize_workspace(cap, scale=0.10):
    """Function to handle workspace initialization."""
    # List of corner names
    corners = ["Top Left", "Top Right", "Bottom Right", "Bottom Left"]
    work_space_lines = []

    while len(work_space_lines) < 4:
        image = capture_frame(cap)
        face_processor.process_image(image)
        image = face_processor.image
        if image is None:
            continue

        if face_processor.face_detected:
            face_processor.draw_head_pose(image)
            face_processor.draw_head_pose_points(image)

            # Get the current corner name based on the number of points saved
            corner_name = corners[len(work_space_lines)]

            # Display "Look at" in the middle of the screen
            text = "Look at corner of your workspace"
            show_message(image, text, get_center_position(image, text, font_scale=1, thickness=2, y_offset=120),
                         color=(0, 0, 255))
            text = "and press 'i'"
            show_message(image, text, get_center_position(image, text, font_scale=1, thickness=2, y_offset=150),
                         color=(0, 0, 255))

            # Get the position for the current corner and display the corner name
            corner_position = get_corner_position(image, corner_name)
            show_message(image, corner_name, corner_position, color=(255, 0, 0))

        else:
            show_message(image, 'Face Not Detected')
            show_message(image, 'Please make sure you are visible for the camera', (20, 100), 0.7)

        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # Esc key
            return None, None
        elif key == ord('i'):  # 'i' key
            if face_processor.face_detected:
                work_space_lines.append(face_processor.tip)
                print(f"Saved workspace point: {corner_name}")

        # cv2.imshow('StudyBuddy Feed', image)
        #show image in center of screen\
        cv2.imshow('StudyBuddy Feed', image)

    work_space_border = expand_points(work_space_lines, scale=scale)
    print("Workspace Initialized")
    return work_space_lines, work_space_border


def process_workspace(image, tip, work_space_lines, work_space_border):
    """Process and display workspace border and check if the point is inside."""
    draw_lines(image, work_space_lines, color=(0, 255, 0))  # Green color
    draw_lines(image, work_space_border, color=(0, 0, 255))  # Red color
    # show_message(image, 'Workspace Initialized', (20, 150))

    inside_border = is_point_inside(work_space_border, tip)
    if inside_border:
        show_message(image, 'Cursor Inside Workspace', (20, 100), color=(0, 255, 0))  # Green color
    else:
        show_message(image, 'Cursor Outside Workspace', (20, 100), color=(0, 0, 255))  # Red color


class StudyBuddy:
    def __init__(self, data_filename='data.csv', collect_mode=False):
        """Initialize the StudyBuddy class with necessary resources and variables."""
        self.data_filename = data_filename
        self.collect_mode = collect_mode
        self.cap = cv2.VideoCapture(CAPTURE_DEVICE_INDEX)
        self.detector = BlinkDetector(history_length=5, ear_threshold=2)
        # self.ear_change_plotter = LivePlotter(self.detector.ear_changes, data_label='EAR Change')
        # self.puc_change_plotter = LivePlotter(self.detector.puc_changes, data_label='PUC Change')
        # self.ear_plotter = LivePlotter(self.detector.ear_values, data_label='EAR')
        # self.puc_plotter = LivePlotter(self.detector.puc_values, data_label='PUC')
        # self.ear_dchange_plotter = LivePlotter(self.detector.ear_dchanges, data_label='EAR DChange')

        self.imgshow = True
        self.expressions = [('Sleepy', 0), ('Distracted', 1), ('Focused', 2)]
        self.collect_time = 15
        self.decay = 0.7
        self.work_space_initialized = False
        self.initialize_variables()
        self.start_signal = False
        self.collecting = False
        self.current_expression = 0
        self.norm_values = None
        self.last_data = []

        self.last_collection_time = time.time()
        self.initialize_data_collection()

        # Ensure the data file has the header
        self.ensure_data_file_header()

        self.running = True

    def initialize_variables(self):
        """Initialize variables used across methods."""
        self.ear_main = self.mar_main = self.puc_main = self.moe_main = -1000
        self.eyes_start_time = None
        self.eyes_message_active = False

    def initialize_data_collection(self):
        """Initialize variables related to data collection."""
        self.ear_values = []
        self.mar_values = []
        self.puc_values = []
        self.moe_values = []
        self.total_frames = 0
        self.frames_outside_workspace = 0
        self.frames_no_face_detected = 0
        self.frames_yawn_detected = 0
        self.total_eyes_closed_time = 0
        self.start_time = time.time()
        self.detector.total_blinks = 0

    def ensure_data_file_header(self):
        """Ensure the CSV data file has a header row."""
        with open(self.data_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(['EAR_Mean', 'EAR_Std', 'EAR_Median',
                                 'MAR_Mean', 'MAR_Std', 'MAR_Median',
                                 'PUC_Mean', 'PUC_Std', 'PUC_Median',
                                 'MOE_Mean', 'MOE_Std', 'MOE_Median',
                                 '%_Outside_Workspace', 'Total_Eyes_Closed_Time',
                                 '%_No_Face_Detected', '%_Yawn_Detected',
                                 'Total_Blinks',
                                 'Expression_Label'])

    def run(self):
        """Main loop to process video frames and collect data."""
        # self.ear_change_plotter.start_plot()
        # self.puc_change_plotter.start_plot()
        # self.ear_plotter.start_plot()
        # self.puc_plotter.start_plot()
        # self.ear_dchange_plotter.start_plot()

        if self.norm_values is None:  # Perform calibration if not already done
            self.norm_values = calibrate_features(calib_frame_count=25)
            if self.norm_values is None:
                print("Calibration failed. Exiting...")
                self.release_resources()
            else:
                print("Calibration completed successfully")

        while self.cap.isOpened():
            image = capture_frame(self.cap)
            if image is None:
                break
            start = time.time()

            image = self.process_frame(image)

            if self.collect_mode:
                self.handle_collect_mode(image)
            else:
                # Automatic data collection every 15 seconds
                if time.time() - self.last_collection_time >= self.collect_time:
                    self.process_collected_data()
                    self.initialize_data_collection()
                    self.last_collection_time = time.time()
            if self.imgshow:
                cv2.imshow('StudyBuddy Feed', image)
            elif cv2.getWindowProperty('StudyBuddy Feed', cv2.WND_PROP_VISIBLE) >= 1:  # Check if window is open
                self.close_windows()

            key = cv2.waitKey(5) & 0xFF
            if key == 27:  # Quit the program
                break
            elif key == ord('s'):
                self.start_signal = True

        self.release_resources()

    def process_frame(self, image):
        """Process each video frame for feature extraction and workspace handling."""
        start = time.time()
        ear, mar, puc, moe, tip, image = face_processor.process_image(image)

        # Initialize workspace if not already done
        if not self.work_space_initialized:
            self.work_space_lines, self.work_space_border = initialize_workspace(self.cap)
            if self.work_space_lines is None:
                print("Workspace initialization failed. Exiting...")
                self.release_resources()
                exit()
            self.work_space_initialized = True
            if not self.imgshow:
                cv2.destroyAllWindows()
            return image
        else:
            process_workspace(image, tip, self.work_space_lines, self.work_space_border)  # Display workspace border

        if face_processor.face_detected:
            self.process_face_detected(ear, mar, puc, moe, tip, image)
        else:
            self.ear_main = self.mar_main = self.puc_main = self.moe_main = -1000
            self.frames_no_face_detected += 1

        self.total_frames += 1
        self.display_information(image, start)
        return image

    def process_face_detected(self, ear, mar, puc, moe, tip, image):
        """Process features when a face is detected."""
        # Normalize features
        ear = normalize_value(ear, self.norm_values['ears'][0], self.norm_values['ears'][1])
        mar = normalize_value(mar, self.norm_values['mars'][0], self.norm_values['mars'][1])
        puc = normalize_value(puc, self.norm_values['pucs'][0], self.norm_values['pucs'][1])
        moe = normalize_value(moe, self.norm_values['moes'][0], self.norm_values['moes'][1])

        # Apply exponential moving average
        if self.ear_main == -1000:
            self.ear_main, self.mar_main, self.puc_main, self.moe_main = ear, mar, puc, moe
        else:
            self.ear_main = self.ear_main * self.decay + (1 - self.decay) * ear
            self.mar_main = self.mar_main * self.decay + (1 - self.decay) * mar
            self.puc_main = self.puc_main * self.decay + (1 - self.decay) * puc
            self.moe_main = self.moe_main * self.decay + (1 - self.decay) * moe

        # Process data
        if not is_point_inside(self.work_space_border, tip):
            self.frames_outside_workspace += 1

        # Detect blink
        blink_result = self.detector.detect_blink(self.ear_main, self.puc_main)
        if isinstance(blink_result, float):
            self.eyes_start_time = time.time()
            self.eyes_message_active = True
            self.eyes_time = blink_result
            self.total_eyes_closed_time += self.eyes_time

        # Detect yawn
        if detect_yawn(self.moe_main, yawn_threshold=40, min_duration=0.5):
            self.frames_yawn_detected += 1
            show_message(image, 'Yawn Detected', (20, 250), 1.5, (0, 0, 255), 2)

        # Collect data
        if self.collect_mode and self.collecting:
            self.collect_data()
        elif not self.collect_mode:
            self.collect_data()

    def collect_data(self):
        """Collect features data for statistical analysis."""
        self.ear_values.append(self.ear_main)
        self.mar_values.append(self.mar_main)
        self.puc_values.append(self.puc_main)
        self.moe_values.append(self.moe_main)

    def display_information(self, image, start):
        """Display real-time information on the video frame."""
        if self.eyes_message_active:
            self.eyes_message_active = show_message_for_duration(
                image, f'Eyes Closed for {self.eyes_time:.2f} seconds',
                (20, 200), 2, self.eyes_start_time)

        # Display features
        show_message(image, f"EAR: {self.ear_main:.2f}", (int(0.02 * image.shape[1]), int(0.07 * image.shape[0])), 0.8,
                     (255, 0, 0), 2)
        show_message(image, f"PUC: {self.puc_main:.2f}", (int(0.27 * image.shape[1]), int(0.07 * image.shape[0])), 0.8,
                     (255, 0, 0), 2)
        show_message(image, f"MAR: {self.mar_main:.2f}", (int(0.52 * image.shape[1]), int(0.07 * image.shape[0])), 0.8,
                     (255, 0, 0), 2)
        show_message(image, f"MOE: {self.moe_main:.2f}", (int(0.77 * image.shape[1]), int(0.07 * image.shape[0])), 0.8,
                     (255, 0, 0), 2)
        show_message(image, f"Total Blinks: {self.detector.total_blinks}", (30, 150))

        # Draw face mesh
        face_processor.draw_head_pose(image)
        face_processor.draw_head_pose_points(image)

        # Display FPS
        fps = 1 / (time.time() - start)
        show_message(image, f"FPS: {int(fps)}", (20, 450))

    def handle_collect_mode(self, image):
        """Handle data collection when in collect mode."""
        if not self.collecting:
            if self.start_signal:
                self.start_signal = False
                self.collecting = True
                print(f"Collecting data for '{self.expressions[self.current_expression][0]}' expression")
                self.initialize_data_collection()
                self.start_time = time.time()
            else:
                show_message(
                    image,
                    f"Press 's' to start collecting data for '{self.expressions[self.current_expression][0]}' expression",
                    (20, 400), 0.7, (0, 0, 255))
        elif (time.time() - self.start_time) > self.collect_time:
            self.process_collected_data()
            self.collecting = False
        else:
            time_left = self.collect_time - (time.time() - self.start_time)
            show_message(
                image,
                f"Collecting data for '{self.expressions[self.current_expression][0]}': {time_left:.2f} seconds left",
                (20, 400), 0.7, (0, 0, 255))

    def process_collected_data(self):
        """Process and save the collected data at the end of each interval."""
        # Calculate statistics
        ear_mean, ear_std, ear_median = calculate_statistics(self.ear_values)
        mar_mean, mar_std, mar_median = calculate_statistics(self.mar_values)
        puc_mean, puc_std, puc_median = calculate_statistics(self.puc_values)
        moe_mean, moe_std, moe_median = calculate_statistics(self.moe_values)

        percent_outside_workspace = (
                                                self.frames_outside_workspace / self.total_frames) * 100 if self.total_frames > 0 else 0
        percent_no_face_detected = (
                                               self.frames_no_face_detected / self.total_frames) * 100 if self.total_frames > 0 else 0
        percent_yawn_detected = (self.frames_yawn_detected / self.total_frames) * 100 if self.total_frames > 0 else 0

        # Print statistics
        print(f"EAR: Mean={ear_mean:.2f}, Std={ear_std:.2f}, Median={ear_median:.2f}")
        print(f"MAR: Mean={mar_mean:.2f}, Std={mar_std:.2f}, Median={mar_median:.2f}")
        print(f"PUC: Mean={puc_mean:.2f}, Std={puc_std:.2f}, Median={puc_median:.2f}")
        print(f"MOE: Mean={moe_mean:.2f}, Std={moe_std:.2f}, Median={moe_median:.2f}")
        print(f"% Outside Workspace: {percent_outside_workspace:.2f}")
        print(f"% No Face Detected: {percent_no_face_detected:.2f}")
        print(f"% Yawn Detected: {percent_yawn_detected:.2f}")
        print(f"Total Blinks: {self.detector.total_blinks}")

        self.last_data = [ear_mean, ear_std, ear_median,
                          mar_mean, mar_std, mar_median,
                          puc_mean, puc_std, puc_median,
                          moe_mean, moe_std, moe_median,
                          percent_outside_workspace, self.total_eyes_closed_time,
                          percent_no_face_detected, percent_yawn_detected, self.detector.total_blinks,
                          ]

        if self.collect_mode:
            # Write to CSV file
            with open(self.data_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.last_data + [self.expressions[self.current_expression][1]])

            print(f"Data collection for '{self.expressions[self.current_expression][0]}' expression completed")
            self.current_expression = (self.current_expression + 1) % len(self.expressions)

    def release_resources(self):
        """Release all resources used by the class."""
        self.cap.release()
        cv2.destroyAllWindows()
        self.running = False
        # self.ear_plotter.stop_plot()

    def close_windows(self):
        cv2.destroyAllWindows()


# Constants and global instances
CAPTURE_DEVICE_INDEX = 1
face_processor = FaceMeshProcessor(face_mesh)

# Running the StudyBuddy class
if __name__ == '__main__':
    # Pass collect_mode=True to enable collect mode
    study_buddy = StudyBuddy(data_filename='data.csv', collect_mode=True)
    study_buddy.run()
