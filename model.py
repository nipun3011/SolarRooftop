import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Minimum building area threshold
    min_building_area = 1000
    building_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_building_area]

    # Draw contours on the image
    result_image = image.copy()
    cv2.drawContours(result_image, building_contours, -1, (0, 255, 0), 2)

    # Calculate total rooftop area
    total_rooftop_area = sum([cv2.contourArea(cnt) for cnt in building_contours])

    # Solar panel efficiency and solar radiation
    solar_panel_efficiency = 0.18
    solar_radiation = 5.0
    
    # Calculate estimated energy production
    energy_production = total_rooftop_area * solar_panel_efficiency * solar_radiation

    # Prepare response data
    response_data = {
        "number_of_buildings": len(building_contours),
        "total_rooftop_area": total_rooftop_area,
        "estimated_energy_production": energy_production
        
    }

    return response_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_buildings', methods=['POST'])
def detect_buildings():
    try:
        file = request.files['image']
        print(f"Received file: {file.filename}")

        image_path = "input_image.jpg"
        file.save(image_path)

        result_data = process_image(image_path)
        print("Image processed successfully")

        return jsonify(result_data)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
