# Augmented Reality-based Instructor for Origami

This AR Instructor teaches you origami step-by-step.

### Installation

1. Download this project and `cd` to it

2. Refer to [Python's guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment) to create a virtual environment

3. Install Python packages

   ```
   pip install -r requirements.txt
   ```

4. Run main.py \*note: You might want to calibrate the HSV values to improve detection accuracy
   ```
   python main.py
   ```

#### HSV Calibration

1. Run hsvCalibration.py

   ```
   python hsvCalibration.py
   ```

2. To toggle HSV trackbars:
   'x': Paper silhouette
   'y': Skin colour
   'z': Coloured side of paper

3. To quit and save a particular HSV range:
   's': Paper silhouette
   't': Skin colour
   'u': Coloured side of paper

   To quit without saving: 'q'
