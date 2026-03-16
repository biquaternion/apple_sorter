.PHONY: help run train download-datasets install-dependencies clean

help:
	@echo 'Apple Sorter App Makefile'
	@echo ''
	@echo 'Usage:'
	@echo '  make inference           	Run inference on images'
	@echo '  make download-datasets 	Download datasets'
	@echo '  make install             	Install dependencies'
	@echo '  make help                	Show this help message'

install:
	pip install -r requirements.txt

run:
	PYTHONPATH=./src python main.py input_path='$(INPUT_PATH)' output_path='$(OUTPUT_PATH)' detector=grounding_dino | PYTHONPATH=./src python src/visualization/viewer.py

run-interactive:
	PYTHONPATH=./src python main.py interactive=true detector=grounding_dino pipeline.classifier=null detector.model_name=IDEA-Research/grounding-dino-base | PYTHONPATH=./src python src/visualization/viewer.py

download-dataset:
	@echo 'Downloading dataset'
	# Add commands to download datasets here
	# Example commands (adjust according to actual dataset sources):
	mkdir -p 'datasets/minneapple'
	wget -O 'datasets/minneapple/counting.tar.gz' 'https://conservancy.umn.edu/server/api/core/bitstreams/9a8d9477-4549-4896-974b-e824c5ab2d19/content'
	tar -xf 'datasets/minneapple/counting.tar.gz' -C 'datasets/minneapple'
#    @echo 'Dataset was downloaded'

train:
	python train/train_classifier.py --config configs/train/config.yaml

clean:
	rm -rf outputs/
	rm -f *.csv
	rm -f *.log
