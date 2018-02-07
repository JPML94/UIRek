# Steps to create your custom object detector with TensorFlow's Object Detection API


###	1. Collect a few hundred images that contain the objects you want to detect. Make sure you get some edge cases where it's difficult for the object to be recognized so your model is does not over fit.
###	2. Annotate your classes (Button, form, navigation, etc) with labelImg. To install labelImg on your machine follow these steps:

		a. $ sudo apt-get install pyqt5-dev-tools
		b. $ sudo pip3 install lxml
		c. $ make qt5py3
		d. $ python3 labelImg.py
		e. When the program is open click 'Change default saved annotation folder' (This is a folder that is going to contain an XML file for every image you annotate with the same filename as your images.
		f. Click 'Open Dir' and select the folder that contains your images.
###	3. Once labelImg is loaded with all your images you can start creating RectBoxes for every class you want your model to be trained on. After you finish with an image click 'Save' and then 'Next' until you go through all the images in your folder. 
###	4. You'll have to split your data into train & test samples (90/10 split for now) for both the images and xml's.
###	5. Once you have all your images and xml files ready we can start coding. You'll have to convert your data into TF Records from each split:

		a. For this first we have to run xml_to_csv.py
		b. Then run generate_tfrecord.py (assuming you already have everything installed to run this tutorial, that means protobuf 2.6, pillow 1.0, lxml, tf Slim, Jupyter Notebook, Matplotlib, Tensorflow) I will include more in depth installation instructions with a requirements.txt file to install everything with one step.
		c. $ python3 generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record
		d. $ python3 generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record
###	6. Next, you can either train a new model from scratch or do Transfer Learning with a pre-trained model to achieve better accuracy faster. (probably best to do tests with Transfer Learning to achieve results faster but then train a model specifically for our use case). To select the model you'll have to use a .config file provided in this repository
###	7. Train!

		a. $ python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config
		b. While training you can load TensorBoard via this commmand $ tensorboard --logdir='training'
		c. Go to 127.0.0.1:6006
###	8. Export graph from new trained model to be able to use the model via the following command:

		a. $ python3 export_inference_graph.py \
          --input_type image_tensor \
          --pipeline_config_path training/ssd_mobilenet_v1_pets.config \
          --trained_checkpoint_prefix training/model.ckpt-10856 \
          --output_directory mac_n_cheese_inference_graph

###	9. Test your model by detecting in real time with the Jupyter Notebook with the Run All command.

