import tkinter as tk
from tkinter import ttk
import threading
import numpy as np
import joblib as jb
import warnings
from treeinterpreter import treeinterpreter as ti

# Import your StudyBuddy class here
from FeatureExtraction import StudyBuddy

# Import helper functions
from helpers import water_fall_chart, ChatGPT_API

warnings.filterwarnings("ignore")


class Application:
    def __init__(self):
        # Create the main application window
        self.window = tk.Tk()
        self.window.title('Study Buddy')
        self.window.geometry('450x300')

        # Load the model
        self.model = jb.load('models/random_forest_model.pkl')
        self.last_data = []
        self.expression = ['Sleepy', 'Distracted', 'Focused']

        # Create the StudyBuddy instance
        self.study_buddy = StudyBuddy(data_filename='data.csv', collect_mode=False)

        self.feature_names = ['EAR_Mean', 'EAR_Std', 'EAR_Median', 'MAR_Mean', 'MAR_Std', 'MAR_Median',
                              'PUC_Mean', 'PUC_Std', 'PUC_Median', 'MOE_Mean', 'MOE_Std', 'MOE_Median',
                              '%_Outside_Workspace', 'Total_Eyes_Closed_Time', '%_No_Face_Detected',
                              '%_Yawn_Detected', 'Total_Blinks']
        self.target_names = ['Sleepy', 'Distracted', 'Focused']

        # Initialize variables
        self.stored_prediction_data = None
        self.current_class = 2
        self.current_expression = "Focused"
        self.important_feature_contributions = {}
        self.pomodoro_minutes = 0
        self.pomodoro_sessions = 0
        self.break_counter = 0
        self.sleepy_counter = 0
        self.distracted_counter = 0
        self.last_expressions = []

        # Start the StudyBuddy run method in a separate thread
        self.study_buddy_thread = threading.Thread(target=self.run_study_buddy)
        self.study_buddy_thread.daemon = True  # Stop the thread when the main program exits

        # Set up frames and widgets
        self.setup_frames()
        self.setup_widgets()

        # Initially hide the main frames
        self.main_frame.grid_forget()
        self.finish_frame.grid_forget()

        # Run the main loop
        self.window.mainloop()

    def run_study_buddy(self):
        print("Starting StudyBuddy...")
        self.study_buddy.run()

    def initialize_study_buddy(self):
        if not self.study_buddy_thread.is_alive():
            if not self.study_buddy.running:  # If StudyBuddy is running, stop it first
                self.window.quit()
                print("Initializing not successful, quitting program...")
            else:
                self.study_buddy_thread.start()
                print("StudyBuddy started!")
                self.initialize_study_buddy()
                self.start_label.config(text='Starting, please wait...', font='Calibri 18')
        else:
            if not self.study_buddy.work_space_initialized:
                self.window.after(2000, self.initialize_study_buddy)
                self.start_label.config(text='Please follow the instructions...', font='Calibri 18')
            else:
                print("StudyBuddy is initialized!")
                self.show_main_page()
                self.study_buddy.imgshow = False

    def show_main_page(self):
        self.start_frame.grid_forget()  # Hide start screen
        self.main_frame.grid()  # Show main screen
        self.count_down()  # Start the countdown after showing the main page
        self.update_gui()  # Start updating the GUI with StudyBuddy stats

    def show_finish_frame(self):
        self.main_frame.grid_forget()
        self.finish_frame.grid()  # Show the finish frame

    def count_down(self, total_in_seconds=None):
        if total_in_seconds is None:
            hour = int(self.hour_var.get())
            minute = int(self.minute_var.get())
            second = int(self.second_var.get())
            total_in_seconds = hour * 3600 + minute * 60 + second

        if total_in_seconds > 0:
            hour, remainder = divmod(total_in_seconds, 3600)
            minute, second = divmod(remainder, 60)
            self.pomodoro_minutes = 25 - minute

            self.hour_var.set(f'{hour:02}')
            self.minute_var.set(f'{minute:02}')
            self.second_var.set(f'{second:02}')

            # Update the GUI every second
            self.window.after(1000, self.count_down, total_in_seconds - 1)
        else:
            # When countdown finished, show finish window
            self.show_finish_frame()

    def update_gui(self):
        # Access the StudyBuddy variables
        if self.study_buddy.running:
            # Retrieve the current state (distracted, focused, etc.)
            data = self.study_buddy.last_data.copy()
            if data:
                data.sort()
                self.last_data.sort()  # Sort to compare data accurately
                if data != self.last_data:
                    if not np.isnan(data).any():
                        sample = np.array(self.study_buddy.last_data.copy()).reshape(1, -1).copy()
                        self.last_data = data
                        self.predict_class(sample)
        else:
            self.feedback_label.config(text="StudyBuddy is not running..., please restart the program")

        # Schedule the next update in 1000 milliseconds (1 second)
        self.window.after(1000, self.update_gui)

    def show_webcam(self):
        if self.study_buddy.imgshow:  # If the webcam is already showing, hide it
            self.study_buddy.imgshow = False
            self.buttonB.config(text='Show Webcam')
        else:
            self.buttonB.config(text='Hide Webcam')
            self.study_buddy.imgshow = True

    def setup_frames(self):
        # Make new frames for the start, main, and finish screens
        self.start_frame = tk.Frame(self.window)
        self.start_frame.grid(row=0, column=0, sticky='nsew')


        self.main_frame = tk.Frame(self.window)

        self.finish_frame = tk.Frame(self.window)

    def setup_widgets(self):
        # Start Frame Widgets
        self.start_label = ttk.Label(self.start_frame, text='Welcome to Study Buddy!', font='Calibri 24 bold')
        self.start_label.grid(row=0, column=0, pady=100, padx=50)

        self.start_button = ttk.Button(self.start_frame, text='Start', command=self.initialize_study_buddy)
        self.start_button.grid(row=1, column=0)

        # Main Frame Widgets
        self.hour_var = tk.StringVar(value='00')
        self.minute_var = tk.StringVar(value='00')
        self.second_var = tk.StringVar(value='400')

        # Load the images
        self.image_robot = tk.PhotoImage(file="robot-neutral.png")
        self.resized_image_robot = self.image_robot.subsample(5, 5)

        # Widgets on the main page
        self.label = ttk.Label(self.main_frame, text='Study buddy', font='Calibri 24 bold')
        self.feedback_label = ttk.Label(
            self.main_frame, text="Initializing...", font='Calibri 12',
            wraplength=200, justify="center", background="white"
        )
        self.buttonA = ttk.Button(self.main_frame, text='Decision insights', command=self.show_waterfall_chart)
        self.buttonB = ttk.Button(self.main_frame, text='Show Webcam', command=self.show_webcam)
        self.buttonC = ttk.Button(self.main_frame, text='Explain decision', command=self.explain_decision)
        self.image_robot_label = ttk.Label(self.main_frame, image=self.resized_image_robot)
        self.hour_lbl = tk.Label(self.main_frame, font=('Arial', 50), textvariable=self.hour_var)
        self.colon1_lbl = tk.Label(self.main_frame, font=('Arial', 50), text=':')
        self.minute_lbl = tk.Label(self.main_frame, font=('Arial', 50), textvariable=self.minute_var)
        self.colon2_lbl = tk.Label(self.main_frame, font=('Arial', 50), text=':')
        self.second_lbl = tk.Label(self.main_frame, font=('Arial', 50), textvariable=self.second_var)

        # Grid configuration for the main frame
        self.main_frame.columnconfigure(tuple(range(8)), weight=1)
        self.main_frame.rowconfigure(tuple(range(8)), weight=1)

        # Widget placement
        self.image_robot_label.grid(row=5, column=1, columnspan=1, rowspan=2)
        self.feedback_label.grid(row=2, column=3, columnspan=4, rowspan=4)
        self.buttonA.grid(row=6, column=4)
        self.buttonB.grid(row=6, column=3)
        self.buttonC.grid(row=6, column=5)
        self.hour_lbl.grid(column=1, row=0)
        self.colon1_lbl.grid(column=2, row=0)
        self.minute_lbl.grid(column=3, row=0)
        self.colon2_lbl.grid(column=4, row=0)
        self.second_lbl.grid(column=5, row=0)

        # Finish Frame Widgets
        self.finish_label = ttk.Label(self.finish_frame, text="Time's Up!", font='Calibri 24 bold')
        self.finish_label.grid(row=0, column=0, pady=30)

        self.close_button = ttk.Button(self.finish_frame, text='Close', command=self.window.quit)
        self.close_button.grid(row=2, column=0, pady=10)

    def predict_class(self, sample):
        # Make prediction using the model
        predicted_class_index, class_contributions, bias = self.make_prediction(sample)

        # Compute confidence
        confidence = self.compute_confidence(sample, predicted_class_index)

        # Store data for the waterfall chart
        self.store_prediction_data(sample, predicted_class_index, class_contributions, bias)

        # Process feature contributions
        self.process_feature_contributions(class_contributions)

        # Update the expression history
        self.last_expressions.append(self.current_expression)

        if len(self.last_expressions) > 8:  # Keep only the last 8 expressions (so last 2 minutes)
            self.last_expressions.pop(0)

        # Check if the user needs feedback
        self.check_feedback()

    def make_prediction(self, sample):
        # Get the prediction, bias, and feature contributions using treeinterpreter
        prediction, bias, contributions = ti.predict(self.model, sample)
        contributions = np.round(contributions, 3)
        predicted_class_index = prediction[0].argmax()

        class_contributions = contributions[0][:, predicted_class_index]
        self.current_expression = self.expression[predicted_class_index]
        self.current_class = predicted_class_index
        print('Predicted class:', self.current_expression)

        return predicted_class_index, class_contributions, bias

    def compute_confidence(self, sample, predicted_class_index):
        # Calculate probabilities from all estimators
        proba_predictions = np.array([tree.predict_proba(sample)[0] for tree in self.model.estimators_])

        # Calculate the mean and standard deviation across all trees
        mean_proba = np.mean(proba_predictions, axis=0)
        std_proba = np.std(proba_predictions, axis=0)
        print('Mean probabilities:', mean_proba)

        # Confidence measure: the inverse of standard deviation for the predicted class
        confidence = 1 - std_proba[predicted_class_index]

        print(f"Standard deviation for predicted class: {std_proba[predicted_class_index]}")
        print(f"Confidence: {confidence}")

        return confidence

    def store_prediction_data(self, sample, predicted_class_index, class_contributions, bias):
        # Store the necessary data for the waterfall chart
        self.stored_prediction_data = {
            'sample': sample,
            'predicted_class_index': predicted_class_index,
            'class_contributions': class_contributions,
            'bias': bias[0][predicted_class_index]
        }

    def process_feature_contributions(self, class_contributions):
        feature_contributions = dict(zip(self.feature_names, class_contributions))
        # Sort the dictionary by absolute value of contributions
        sorted_feature_contributions = dict(
            sorted(feature_contributions.items(), key=lambda item: abs(item[1]), reverse=True)
        )
        # Keep features with contributions greater than 0.05
        self.important_feature_contributions = {
            k: v for k, v in sorted_feature_contributions.items() if abs(v) > 0.05
        }

    def check_feedback(self, min_expressions=4):
        # Check if users had been sleepy or distracted for at least 4 times in the last 8 expressions
        sleepy_count = self.last_expressions.count('Sleepy')
        distracted_count = self.last_expressions.count('Distracted')

        if self.current_class == 0:  # Sleepy
            if sleepy_count >= min_expressions:
                self.sleepy_counter += 1
                self.feedback_user()
            else:
                print(f"Sleepy count: {sleepy_count}")
        elif self.current_class == 1:  # Distracted
            if distracted_count >= min_expressions:
                self.distracted_counter += 1
                self.feedback_user()
            else:
                print(f"Distracted count: {distracted_count}")


    def show_waterfall_chart(self):
        if self.stored_prediction_data:
            # Use the stored data to generate the waterfall chart
            water_fall_chart(
                self.model,
                self.stored_prediction_data['sample'],
                self.stored_prediction_data['predicted_class_index'],
                self.feature_names,
                self.target_names,
                self.stored_prediction_data['class_contributions'],
                self.stored_prediction_data['bias']
            )
        else:
            print("No prediction data available to display.")


    def feedback_user(self):

        PPromptPromodoroTimer = str(self.pomodoro_minutes)
        PPromodoroCycle = str(self.pomodoro_sessions)
        PBreakCounter = str(self.break_counter)
        PSleepyCounter = str(self.sleepy_counter)
        PDistractedcounter = str(self.distracted_counter)

        # opzet identity van coach
        PromptCoach = ("I am going to provide you with a persona, with whom I want to send communications. I want "
                       "you to answer my questions/statements in the  style of that persona. We will use this information to "
                       "create specific content that I request. After the word done I will add my command/question that you "
                       "have to answer.  Demographics: Education: Bachelor’s degree or higher  Occupation: Education mentor, "
                       "(one men’s business) Psychographics:  Values: Values autonomy, personal growth, and work-life "
                       "balance.  Vision: The most successful coaching philosophies empower persons and give them ownership "
                       "of their learning, thus helping them to become independent decision-makers. Beliefs: Believes that "
                       "intrinsic motivation drives better performance and satisfaction in life. Interests: Enjoys reading "
                       "self-improvement books, attending workshops on leadership, and participating in community service.  "
                       "Lifestyle: Strives for a balanced lifestyle that prioritizes hobbies, intellectual development, "
                       "and personal wellness.  Personality Traits: Empathetic, organized, goal-oriented, and reflective; "
                       "tends to overthink decisions but is open to change and learning. Behaviors:  Interactions with "
                       "Products/Services: Prefers products that enhance productivity and personal growth; values "
                       "user-friendly technology and services. Purchasing Habits: Researches extensively before making a "
                       "purchase, often seeking peer reviews and expert opinions. Preferred Communication Channels: Engages "
                       "with brands through email newsletters, social media platforms (especially LinkedIn), and professional "
                       "blogs.  Goals and Aspirations: To cultivate a fulfilling life while helping others with her career. "
                       "Professional Goals: Aims to get 3 people working in her own company  Aspirations: Aspires to become a "
                       "mentor for younger professionals and contribute to meaningful community projects. Pain Points and "
                       "Challenges:  Work-Related Stress: Often feels overwhelmed by balancing work administration.  "
                       "Self-Doubt: Experiences imposter syndrome, questioning her abilities and decisions, which can hinder "
                       "her career progression.   Information Sources: Relies on reputable websites, online courses, "
                       "and self-help books. Engages with thought leaders on LinkedIn and participates in forums related to "
                       "career development and personal growth. This persona is motivated by leadership philosophies that "
                       "integrate emotional intelligence with technical skill, always looking to inspire not just "
                       "performance, but growth and personal fulfillment.  Done")

        if self.current_class == 0:  # Sleepy
            messagePrompt = (
                        PromptCoach + ("You noticed that the person you are coaching has been sleepy for 2 minutes. "
                                       "The person is using the Pomodoro technique. The person has already completed ")
                        + PPromptPromodoroTimer + " min of the 25 min. the person has already done " + PPromodoroCycle +
                        " Pomodoro cycles. You already sent " + PSleepyCounter + " messages during the study session. "
                                                                                 "Decide if they should take the break "
                                                                                 "earlier, stop with studying and start "
                                                                                 "on another time or give a tip to make "
                                                                                 "them less sleepy or send a message "
                                                                                 "that gets them back on track. The "
                                                                                 "possibility is there to use Emoticons. "
                                                                                 "Do NOT use — or - in the text. max. 40 "
                                                                                 "words ")

        if self.current_class == 1:  # Distracted
            messagePrompt = (PromptCoach + (
                "You are coaching someone using the Pomodoro technique. They have been distracted "
                "for 2 minutes during their current Pomodoro session. They have completed ") +
                             PPromptPromodoroTimer + "  min of the 25 min. " + PDistractedcounter + " messages have been sent "
                                                                                                    "yet in this Pomodoro. If "
                                                                                                    "this is their first "
                                                                                                    "distraction and they "
                                                                                                    "have completed less than "
                                                                                                    "15 minutes, "
                                                                                                    "send a motivational "
                                                                                                    "message to help them "
                                                                                                    "refocus. Suggest a break "
                                                                                                    "if they have been "
                                                                                                    "distracted more than 5 "
                                                                                                    "times and if they have "
                                                                                                    "completed 18 minutes or "
                                                                                                    "more of the session. "
                                                                                                    "Keep the message brief "
                                                                                                    "and encouraging. "
                                                                                                    "Strictly follow these "
                                                                                                    "conditions without any "
                                                                                                    "extra explanations. "
                                                                                                    "Avoid using example "
                                                                                                    "phrases directly. The "
                                                                                                    "possibility is there to "
                                                                                                    "use Emoticons. Do NOT "
                                                                                                    "use — or - in the text. "
                                                                                                    "max. 40 words")

        if self.current_class == 2:  # break/focussed
            messagePrompt = (
                        PromptCoach + "The 25 minutes are over of the Pomodoro you are in cycle " + PPromodoroCycle +
                        ". you have now a break. Normally the break is 5 minutes, however when it is cycle 4,8,12,16,20,"
                        "24,28,32,36,40,44,48,52,56 the break will become 30 minutes. read the previous sentence "
                        "carefully and analyze when 5min break and when 30min break " + "Your student has been "
                                                                                        "distracted " +
                        PDistractedcounter + " times and has been sleepy " + PSleepyCounter + " times during this cycle. "
                                                                                              "Calculate the break and "
                                                                                              "send a message that they "
                                                                                              "have a break of .. "
                                                                                              "minutes and give them "
                                                                                              "some feedback about their "
                                                                                              "session, also add some "
                                                                                              "statistics in there of "
                                                                                              "the session. The "
                                                                                              "possibility is there to "
                                                                                              "use Emoticons. Do NOT use "
                                                                                              "— or - in the text.  ")
        message = ChatGPT_API(messagePrompt)
        self.feedback_label.config(text=message)
        self.last_expressions = [] # Reset the last expressions to prevent multiple feedbacks
        print(message)

    def explain_decision(self):

        base = f"""Based on the provided data about the class contribution for the predicted label "{self.current_expression}", explain which features contributed positively (value > 0) or negatively (value < 0). These values do not represent the actual value of the feature, but rather how much it was taking into account for the prediction. {self.important_feature_contributions}"""

        data_description = """
        EAR_Mean: Average eye openness.
        EAR_Std: Variability in eye openness.
        EAR_Median: Median eye openness.
        MAR_Mean: Average mouth openness.
        MAR_Std: Variability in mouth openness.
        MAR_Median: Median mouth openness.
        PUC_Mean: Average pupil roundness.
        PUC_Std: Variability in pupil roundness.
        PUC_Median: Median pupil roundness.
        MOE_Mean: Average combined eye and mouth openness.
        MOE_Std: Variability in combined eye and mouth openness.
        MOE_Median: Median combined eye and mouth openness.
        %_Outside_Workspace: Percentage of time outside workspace boundaries.
        """

        last_data_description = f"""Explain in simple terms what user behavior impacted the decision positively 
        and/ornegatively for the predicted label in less than 50 words.
        Example: The openness of your mouth and fast changing pupil movement contributed the most for prediction 
        "focused", indicating it correspond to foccused behaviour."""

        # Use the ChatGPT API to generate an explanation
        message = base + data_description + last_data_description
        explanation = ChatGPT_API(message)
        self.feedback_label.config(text=explanation)

if __name__ == "__main__":
    app = Application()
