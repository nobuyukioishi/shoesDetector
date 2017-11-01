import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import re

def get_data(input_path):
	all_imgs = []

	classes_count = {}

	class_mapping = {}

	visualise = False

	data_paths = [os.path.join(input_path,s) for s in ['shoesSamples']]
	

	print('Parsing annotation files')

	for data_path in data_paths:

		annot_path = os.path.join(data_path, 'Annotations')
		imgs_path = os.path.join(data_path, 'PNGImages')
		imgsets_path_trainval = os.path.join(data_path, 'ImageSets','Main','trainval.txt')
		imgsets_path_test = os.path.join(data_path, 'ImageSets','Main','test.txt')

		trainval_files = []
		test_files = []
		try:
			with open(imgsets_path_trainval) as f:
				for line in f:
					trainval_files.append(line.strip() + '.png')
			with open(imgsets_path_test) as f:
				for line in f:
					test_files.append(line.strip() + '.png')
		except Exception as e:
			print(e)
		
		# print "trainval_files:", trainval_files
		# print "test_files:", test_files

		annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
		idx = 0
		for annot in annots:
			try:
				idx += 1

				et = ET.parse(annot)
				element = et.getroot()

				element_objs = element.findall('object')
				element_filename = element.find('filename').text
				element_filename = re.sub('\t', "", element_filename)
				element_filename = re.sub('\n', "", element_filename)
				element_filename = re.sub(' ', "",  element_filename)
				element_filename = element_filename+".png"
				element_width = int(element.find('size').find('width').text)
				element_height = int(element.find('size').find('height').text)
				# print element_filename
				# print element_width
				# print element_height
				if len(element_objs) > 0:
					annotation_data = {'filepath': os.path.join(imgs_path, element_filename), 'width': element_width,
									   'height': element_height, 'bboxes': []}

					if element_filename in trainval_files:
						annotation_data['imageset'] = 'trainval'
					elif element_filename in test_files:
						annotation_data['imageset'] = 'test'
					else:
						annotation_data['imageset'] = 'trainval'

				for element_obj in element_objs:
					class_name = element_obj.find('name').text
					class_name = re.sub('shoes', 'shoe', class_name)
					class_name = re.sub('Copy of ', "", class_name)			
					class_name = re.sub('\t', "", class_name)
					class_name = re.sub('\n', "", class_name)
					class_name = re.sub(' ', "",  class_name)
					
					# first attempt: only detect slipper
					# for avoiding shoe slipper
					if class_name == "shoe" or class_name == "slipper":
						print("pass done: ", class_name)
						continue
					

					if class_name not in classes_count:
						classes_count[class_name] = 1
					else:
						classes_count[class_name] += 1

					if class_name not in class_mapping:
						class_mapping[class_name] = len(class_mapping)

					obj_bbox = element_obj.find('bndbox')
					x1 = int(round(float(obj_bbox.find('xmin').text)))
					y1 = int(round(float(obj_bbox.find('ymin').text)))
					x2 = int(round(float(obj_bbox.find('xmax').text)))
					y2 = int(round(float(obj_bbox.find('ymax').text)))
					difficulty = int(element_obj.find('difficult').text) == 1
					annotation_data['bboxes'].append(
						{'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})
				all_imgs.append(annotation_data)

				if visualise:
					img = cv2.imread(annotation_data['filepath'])
					for bbox in annotation_data['bboxes']:
						cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox[
									  'x2'], bbox['y2']), (0, 0, 255))
					cv2.imshow('img', img)
					cv2.waitKey(0)

			except Exception as e:
				print(e)
				continue
	return all_imgs, classes_count, class_mapping
