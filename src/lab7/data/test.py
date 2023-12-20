import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_cdt


def render_image(pixels, width, height, max_gray_value):
    # 将像素列表转换为NumPy数组
    pixels_array = np.array(pixels).reshape(height, width)

    # 显示图像
    plt.imshow(pixels_array, cmap='gray', vmin=0, vmax=max_gray_value)
    plt.show()


def load_map(map_path):
    with open(map_path, 'r') as file:
        data = yaml.safe_load(file)
        image_path = data['image']
        resolution = data['resolution']
        origin = data['origin']
        occupied_thresh = data['occupied_thresh']
        
        # Load PGM file
        with open(image_path, 'rb') as pgm_file:
            pgm_header = pgm_file.readline().decode('utf-8').strip()
            if pgm_header != 'P5':
                raise ValueError('Invalid PGM file format')
            
            # Comment line
            pgm_file.readline()
            
            # Read dimensions from the PGM header
            width, height = map(int, pgm_file.readline().decode('utf-8').strip().split())
            
            # Read the max value (usually 255)
            max_val = int(pgm_file.readline().decode('utf-8').strip())
            
            # Read image data
            image = np.fromfile(pgm_file, dtype=np.uint8)
        
        # Reshape the image based on the dimensions
        image = image.reshape((height, width))
        
        return image, resolution, origin, occupied_thresh, width, height


def get_map_coordinates(x_rel, y_rel, resolution, origin, width, height):
    x_map = int((x_rel - origin[0]) / resolution)
    y_map = height - int((y_rel - origin[1]) / resolution)
    return x_map, y_map


def find_nearest_occupied_distance(map_image, x_map, y_map, resolution):
    # Assuming 255 represents occupied cells in the PGM file
    occupied_cells = np.argwhere(map_image == 255)
    
    if len(occupied_cells) == 0:
        return None  # No occupied cells found
    
    distances = distance_transform_cdt(map_image != 0) * resolution
    return distances[y_map, x_map]


def main():
    map_path = './map.yaml'
    x_rel, y_rel = -1.1849005393302707, -1.7624668049043297
    
    map_image, resolution, origin, _, width, height = load_map(map_path)
    
    x_map, y_map = get_map_coordinates(x_rel, y_rel, resolution, origin, width, height)
    
    map_image[y_map, x_map] = 127
    
    render_image(map_image, width, height, 255)
    
    distance = find_nearest_occupied_distance(map_image, x_map, y_map, resolution)
    
    if distance is not None:
        print(f"The nearest occupied cell is {distance} meters away.")
    else:
        print("No occupied cells found in the map.")


if __name__ == "__main__":
    main()

