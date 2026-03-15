## Apple Detection and Sorting App
Developed using Python 3.12

Requires
- **ultralytics yolov8** or
- **grounding-dino**
- **depth-anything-v2** (see requirements.txt) and
- **hydra** for configuration management
- **opencv** for visualization and some image processing routines

Supports command-line input and interactive mode

Usage:
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