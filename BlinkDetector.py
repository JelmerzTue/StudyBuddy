from helpers import *
import time


class BlinkDetector:
    def __init__(self, history_length, ear_threshold):
        # History parameters
        self.HISTORY_LENGTH = history_length
        self.EAR_THRESHOLD = ear_threshold

        # Initialize variables
        self.ear_history = []
        self.puc_history = []
        self.ear_changes = []
        self.puc_changes = []
        self.ear_values = []
        self.puc_values = []
        self.ear_dchanges = []

        # Blink detection state
        self.eyes_closed = False
        self.eyes_start_time = None
        self.total_blinks = 0
        self.closed_ear = None
        self.ear_first_change = None
        self.first_ear = None
        self.puc_first_change = None
        self.first_puc = None
        self.closed_puc = None
        self.first_puc = None


    def detect_blink(self, ear_main, puc_main):
        # Append current EAR and PUC values to history
        self.ear_history.append(ear_main)
        self.puc_history.append(puc_main)

        # Keep history length manageable
        if len(self.ear_history) > self.HISTORY_LENGTH:
            self.ear_history.pop(0)
        if len(self.puc_history) > self.HISTORY_LENGTH:
            self.puc_history.pop(0)

        # Check if we have enough history to calculate rate of change
        if len(self.ear_history) >= self.HISTORY_LENGTH and len(self.puc_history) >= self.HISTORY_LENGTH:
            # Calculate rate of change in EAR and PUC
            ear_change = self.ear_history[-1] - self.ear_history[0]  # Difference over the history
            puc_change = self.puc_history[-1] - self.puc_history[0]  # Difference over the history
            # derivate of ear and puc change

            # puc_dchange = puc_change - self.puc_history[-2] + self.puc_history[1]

            # Append the rate of change to the history
            self.ear_changes = append_data(ear_change, self.ear_changes, 250)
            self.puc_changes = append_data(puc_change, self.puc_changes, 250)
            self.ear_values = append_data(ear_main, self.ear_values, 250)
            self.puc_values = append_data(puc_main, self.puc_values, 250)
            if len(self.ear_changes) > 1:
                ear_dchange = ear_change - self.ear_changes[-2]
                self.ear_dchanges = append_data(ear_dchange, self.ear_dchanges, 250)

            # Detect rapid decrease in EAR and PUC for blink detection
            if ear_change < -self.EAR_THRESHOLD and ear_main < -1 and puc_change < 1:
                if not self.eyes_closed:
                    # print(f'Possible blink, Ear Change: {ear_change}, ear : {ear_main}, PUC Change: {puc_change}, PUC: {puc_main}')
                    self.eyes_closed = True
                    self.eyes_start_time = time.time()
                    self.closed_ear = ear_main
                    self.ear_first_change = ear_change
                    self.first_ear = ear_main
                    self.closed_puc = puc_main
                    self.puc_first_change = puc_change
                    self.first_puc = puc_main
                else:
                    if ear_main < self.closed_ear:  # Find the lowest EAR value during eyes closed
                        self.closed_ear = ear_main

            elif self.eyes_closed:
                if puc_main > self.first_puc: # If puc is increasing, reset the eyes closed state
                    self.eyes_closed = False
                    # print('False Blink, reset')
                if puc_main < self.closed_puc: # Find the lowest PUC value during eyes closed
                    self.closed_puc = puc_main

                closed_change = ear_main - self.closed_ear

                if closed_change > self.EAR_THRESHOLD: # if true, eyes are opening
                    self.eyes_closed = False
                    eyes_closed_time = time.time() - self.eyes_start_time
                    # print(f'Ear : {ear_main}, Closed Change: {closed_change}, Closed PUC: {self.closed_puc}')
                    if 0.05 < eyes_closed_time < 0.4:  # Small puc_change when blinking
                        # long enough
                        self.total_blinks += 1
                        # print(f"Blink Detected: {self.total_blinks}")
                    elif eyes_closed_time > 0.4: # Eyes closed for a long time
                        # print(f"Eyes Closed for {eyes_closed_time} seconds")
                        return eyes_closed_time


            # Return the current state (optional)
            return self.total_blinks
