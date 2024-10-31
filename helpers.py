import numpy as np
import cv2
import time
from openai import OpenAI
client = OpenAI(api_key="Insert your API key here")


def append_data(new_data, data, max_length):
    """Append new datapoint to list, make sures it does not go over max_length."""
    data.append(new_data)
    if len(data) > max_length:
        data.pop(0)
    return data


def expand_points(points, scale=0.20):
    """
    Expand the given 4 points with a specified scale percentage

    Points need to be in following order: Top Right, Top Left, Bottom Left, Bottom Right.

    :param points: List of 4 tuples representing the corners (x, y).
    :param scale: The percentage increase to expand the points.
    :return: List of new points.
    """
    # Assuming Bottom Left (0, 0) as origin, adjust accordingly
    top_left, top_right, bottom_right, bottom_left, = points

    # Calculate the differences for expansion
    width = abs(top_right[0] - top_left[0])  # Width between Top Right and Top Left
    height = abs(top_right[1] - bottom_right[1])  # Height between Top Right and Bottom Right

    # Expansion amount
    x_expansion = width * scale
    y_expansion = height * scale

    # Expand each point in the correct direction:
    new_top_right = (int(top_right[0] + x_expansion), int(top_right[1] - y_expansion))
    new_top_left = (int(top_left[0] - x_expansion), int(top_left[1] - y_expansion))
    new_bottom_left = (int(bottom_left[0] - x_expansion), int(bottom_left[1] + y_expansion))
    new_bottom_right = (int(bottom_right[0] + x_expansion), int(bottom_right[1] + y_expansion))

    return [new_top_right, new_top_left, new_bottom_left, new_bottom_right]


def is_point_inside(points, pointer):
    """
    Determine if the pointer is inside the quadrilateral defined by points.
    Points are in the order: ["Top Left", "Top Right", "Bottom Right", "Bottom Left"].

    :param points: List of 4 tuples representing the corners (x, y).
    :param pointer: Tuple (x, y) representing the pointer location.
    :return: True if the pointer is inside the quadrilateral, False otherwise.
    """

    def cross_product(pointA, pointB, pointC):
        """
        Calculate the cross product of vectors AB and AC.
        Cross product tells the relative orientation of the point C with respect to the edge AB.
        """
        vectorAB = (pointB[0] - pointA[0], pointB[1] - pointA[1])
        vectorAC = (pointC[0] - pointA[0], pointC[1] - pointA[1])
        return vectorAB[0] * vectorAC[1] - vectorAB[1] * vectorAC[0]

    # Order of points: Top Left, Top Right, Bottom Right, Bottom Left
    top_left, top_right, bottom_right, bottom_left = points

    # Check cross products for all four edges
    cross1 = cross_product(top_left, top_right, pointer)
    cross2 = cross_product(top_right, bottom_right, pointer)
    cross3 = cross_product(bottom_right, bottom_left, pointer)
    cross4 = cross_product(bottom_left, top_left, pointer)

    # If all cross products have the same sign, the point is inside
    if (cross1 > 0 and cross2 > 0 and cross3 > 0 and cross4 > 0) or \
            (cross1 < 0 and cross2 < 0 and cross3 < 0 and cross4 < 0):
        return True
    else:
        return False


def get_head_direction(angles):
    """Interpret head direction based on angles.
    :param angles:  angles (x, y, z) in radians.
    :return: the direction of the head and the angles in degrees.
    """
    x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360
    if y < -10:
        direction = "Looking Left"
    elif y > 10:
        direction = "Looking Right"
    elif x < -10:
        direction = "Looking Down"
    elif x > 10:
        direction = "Looking Up"
    else:
        direction = "Forward"
    return direction, x, y, z


def distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def detect_yawn(moe, yawn_threshold=40, min_duration=0.5):
    """
    Detects and displays 'Yawn' if the MOE value exceeds the yawn_threshold for at least min_duration seconds.

    :param moe: Current MOE value.
    :param yawn_threshold: The threshold for MOE to detect a yawn (default is 40).
    :param min_duration: Minimum time (in seconds) MOE should stay above the threshold (default is 0.5 seconds).
    :return: True if a yawn is detected, False otherwise.
    """
    # Store start time when MOE exceeds threshold
    if not hasattr(detect_yawn, 'start_time'):
        detect_yawn.start_time = None  # Initialize as None on first call

    if moe > yawn_threshold:
        if detect_yawn.start_time is None:
            # MOE crosses the threshold, start the timer
            detect_yawn.start_time = time.time()
        elif time.time() - detect_yawn.start_time >= min_duration:
            # If MOE has been above the threshold for the required duration
            return True
    else:
        # MOE is below the threshold, reset the timer
        detect_yawn.start_time = None

    return False


def show_message(image, text, position=(20, 50), font_scale=1, color=(0, 255, 0), thickness=2):
    """Helper function to display text on the image."""
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def capture_frame(cap):
    """Capture a single frame from the camera."""
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        return None
    return image


def show_message_for_duration(image, message, position, duration, start_time, font_scale=1, color=(0, 255, 0),
                              thickness=2):
    """Display a message for a certain duration on the image."""
    current_time = time.time()

    # Calculate elapsed time
    elapsed_time = current_time - start_time

    # If the elapsed time is less than the duration, display the message
    if elapsed_time < duration:
        show_message(image, message, position, font_scale, color, thickness)
        return True  # The message is still being shown

    return False  # The message has finished its display duration


def draw_lines(image, lines, color=(0, 255, 0)):
    """Draw all lines from point 1 to point 2, 2 to 3, 3 to 4, 4 to 1."""
    for i in range(4):
        cv2.line(image, lines[i], lines[(i + 1) % 4], color=color, thickness=1)


def process_frame(image, face_mesh):
    """Process the image and extract face landmarks."""
    image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = face_mesh.process(image_rgb)
    image_rgb.flags.writeable = True
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), results


def get_corner_position(image, corner_name, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
    """Calculate the correct position for displaying text in the given corner of the image.

    Args:
        image (numpy array): The image on which to display the text.
        corner_name (str): The name of the corner ("Top Left", "Top Right", "Bottom Right", "Bottom Left").
        font (int): Font type for text.
        font_scale (float): Font scale factor.
        thickness (int): Thickness of the text.

    Returns:
        tuple: The (x, y) coordinates to display the text.
    """
    height, width, _ = image.shape
    padding = 20  # Padding from the edges of the image

    # Get the text size to ensure correct alignment
    text_size = cv2.getTextSize(corner_name, font, font_scale, thickness)[0]
    text_width, text_height = text_size

    if corner_name == "Top Left":
        return padding, padding + text_height
    elif corner_name == "Top Right":
        return width - padding - text_width, padding + text_height
    elif corner_name == "Bottom Right":
        return width - padding - text_width, height - padding
    elif corner_name == "Bottom Left":
        return padding, height - padding
    else:
        raise ValueError(f"Invalid corner name: {corner_name}")


def get_center_position(image, text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2, x_offset=0, y_offset=0):
    """Calculate the correct position for displaying text in the center of the image.

    Args:
        image (numpy array): The image on which to display the text.
        text (str): The text to display.
        font (int): Font type for text.
        font_scale (float): Font scale factor.
        thickness (int): Thickness of the text.
        x_offset (int): Horizontal offset for the text, positive values move the text to the right.
        y_offset (int): Vertical offset for the text, positive values move the text down.

    Returns:
        tuple: The (x, y) coordinates to display the text.
    """
    height, width, _ = image.shape
    center_position = (width // 2, height // 2)  # Center position for "Look at" message
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_width, _ = text_size
    center_position = (center_position[0] - text_width // 2 + x_offset, center_position[1] + y_offset)
    return center_position


def normalize_value(value, mean, std):
    """Normalize the given value using the provided mean and standard deviation."""
    return (value - mean) / std


def calculate_statistics(values):
    """Calculate mean, standard deviation, and median of the given list of values."""
    # Check if there are nan values in the list
    if np.isnan(values).any():
        return np.nan, np.nan, np.nan
    return np.mean(values), np.std(values), np.median(values)


import numpy as np
import matplotlib.pyplot as plt
from treeinterpreter import treeinterpreter as ti
from waterfall_chart import plot as waterfall


def predict_class(model, sample, expression, feature_names, target_names, feedback_label):
    # Get the prediction, bias, and feature contributions using treeinterpreter
    prediction, bias, contributions = ti.predict(model, sample)
    contributions = np.round(contributions, 3)
    predicted_class_index = prediction[0].argmax()

    class_contributions = contributions[0][:, predicted_class_index]
    print('Predicted class:', expression[predicted_class_index])

    feedback_label.config(text=f"Predicted class: {expression[predicted_class_index]}")
    return predicted_class_index


def water_fall_chart(model, instance, predicted_class_index, feature_names, target_names,
                     class_contributions, initial_value):
    """Create a waterfall chart to visualize feature contributions to the prediction."""

    contribution_values = [initial_value] + list(class_contributions)
    contribution_labels = ['Bias'] + list(feature_names)
    print(len(contribution_labels), len(contribution_values))

    # Make dictionary of feature names and contributions
    feature_contributions = dict(zip(feature_names, class_contributions))
    # Sort the dictionary by values based on absolute value
    sorted_feature_contributions = dict(
        sorted(feature_contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    )

    waterfall(contribution_labels, contribution_values, formatting='{:,.3f}', sorted_value=True, threshold=0.05)
    plt.ylim(0, 1)
    plt.show()

def ChatGPT_API(message):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": message}
        ]
    )

    outputAI = completion.choices[0].message.content
    return outputAI