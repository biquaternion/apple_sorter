## Apple Detection and Sorting App
Developed using Python 3.12

### Requirements
- **ultralytics yolov8** or
- **grounding-dino**
- **depth-anything-v2** and
- **hydra** for configuration management
- **opencv** for visualization and some image processing routines

see requirements.txt

Requires Git-LFS for classifier's weights storage
```
git lfs install
```
if classification's not needed, this step can be skipped

Supports command-line input and interactive mode

### Usage:
- via make:
  ```
  make install-dependencies
  make run-interactive
  ```
- via python (set PYTHONPATH to src)
  ```
  python main.py interactive=true detector=grounding_dino | python src/visualization/viewer.py
  ```
  to switch off classification stage
  ```
  python main.py interactive=true detector=grounding_dino pipeline.classifier=null | python src/visualization/viewer.py
  ```

### Basic pipeline structure
1. run object detection on the input image (grounding-dino-base recommended). 
The detector has a low threshold to maximize recall. 
2. Simple postprocessing of detections:
- detections not labeled as **apple** are removed
- detections whose boxes fully cover other boxes are removed

The center of each bounding box is used as the estimated apple center
3. false positives are filtered using a binary classifier trained on MinneApple dataset
(acc 0.95-0.98 depending on training/validation subsets).
4. apply depth estimation to the input image using depth-anything-v2 (currently the only supported depth model)
5. postprocessing:

for each remaining detection (representing a single apple), the median depth value within the box is used as apple's relative distance
6. sort apples by their relative distance

The classification step is set on by default and can be switched off manually using `pipeline.classifier=null` key added to `main.py`

### Output

The `main.py` writes csv with image name, center and depth to stdout. It also writes images with apple centers drawn to outputs folder.

The `viewer.py` reads csv from stdin and visualizes the results.