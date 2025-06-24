import os
from PIL import Image

def get_id_to_color() -> dict:
    """
    Returns a dictionary mapping class IDs to their corresponding RGB color representations.

    Returns:
        dict: A dictionary where keys are class IDs (integers) and values are RGB tuples.
    """

    return {
        0: (128, 64, 128),    # road
        1: (244, 35, 232),    # sidewalk
        2: (70, 70, 70),      # building
        3: (102, 102, 156),   # wall
        4: (190, 153, 153),   # fence
        5: (153, 153, 153),   # pole
        6: (250, 170, 30),    # light
        7: (220, 220, 0),     # sign
        8: (107, 142, 35),    # vegetation
        9: (152, 251, 152),   # terrain
        10: (70, 130, 180),   # sky
        11: (220, 20, 60),    # person
        12: (255, 0, 0),      # rider
        13: (0, 0, 142),      # car
        14: (0, 0, 70),       # truck
        15: (0, 60, 100),     # bus
        16: (0, 80, 100),     # train
        17: (0, 0, 230),      # motorcycle
        18: (119, 11, 32),    # bicycle
    }



def get_color_to_id() -> dict:
    """
    Creates a dictionary mapping color representations to their corresponding IDs.

    Returns:
        dict: A dictionary where keys are color representations (RGB tuples) and values are IDs.
    """
    
    id_to_color = get_id_to_color()
    color_to_id = {color: id for id, color in id_to_color.items()}
    return color_to_id


def convert_image_to_grayscale(image_path: str, color_to_id: dict, output_path: str):
    """
    Convert a color segmented image to a grayscale segmented image.

    Args:
        image_path (str): Path to the input color segmented image.
        color_to_id (dict): Dictionary mapping RGB color tuples to class IDs.
        output_path (str): Path to save the output grayscale segmented image.
    """
    color_to_id = get_color_to_id()
    image = Image.open(image_path).convert('RGB')
    gray_image = Image.new('L', image.size)
    rgb_pixels = image.load()
    gray_pixels = gray_image.load()

    for i in range(image.width):
        for j in range(image.height):
            rgb = rgb_pixels[i, j]
            gray_pixels[i, j] = color_to_id.get(rgb, 255)  # 255 is the default value for unknown colors

    gray_image.save(output_path)

def process_images(input_folder: str, output_folder: str):
    """
    Process all images in the input folder, converting them to grayscale and saving them in the output folder.

    Args:
        input_folder (str): Path to the folder containing input color segmented images.
        output_folder (str): Path to the folder to save output grayscale segmented images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    color_to_id = get_color_to_id()
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            convert_image_to_grayscale(input_path, color_to_id, output_path)
            print(f"Converted {filename} and saved to {output_path}")

# Directory paths
input_folder = 'labels'
output_folder = 'masks'

# Process images
process_images(input_folder, output_folder)
