import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import cv2
import numpy as np
import matplotlib.pyplot as plt

import pickle
import pandas as pd
from keras.models import model_from_json
from PIL import Image
from tqdm import tqdm

# Helper functions

def show_image(img, title):
	plt.plot()
	plt.imshow(img, cmap="gray") 
	plt.title(title), plt.xticks([]), plt.yticks([])
	plt.show()

def save_image(img, filename):
	plt.plot()
	plt.imshow(img, cmap="gray") 
	plt.title("Edges"), plt.xticks([]), plt.yticks([])
	plt.axes("off")
	plt.savefig(filename)


def detect_edges(img, threshold):
    neiborhood24 = np.array ([[ 1 , 1 , 1 , 1 , 1 , 1 ],
                              [ 1 , 1 , 1 , 1 , 1 , 1 ],
                              [ 1 , 1 , 1 , 1 , 1 , 1 ],
                              [ 1 , 1 , 1 , 1 , 1 , 1 ],
                              [ 1 , 1 , 1 , 1 , 1 , 1 ]],
                              np.uint8)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dilated = cv2.dilate(gray, neiborhood24, iterations = 1 )
    diff = cv2.absdiff(dilated, gray)
    (T, edges) = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY) 
    edges = edges // 255

    return edges

# Guo-Hall thinning
def thin_edges(src):
	def iteration(src, iter):
		marker = np.ones(src.shape, np.uint8)
		h,w = src.shape
		changed = 0
		for j,i in np.transpose(np.nonzero(src)):
			if i==0 or i==w-1: continue
			if j==0 or j==h-1: continue
			assert src.item(j,i)!=0
			p2 = src.item((j,   i-1))
			p3 = src.item((j+1, i-1))
			p4 = src.item((j+1, i))
			p5 = src.item((j+1, i+1))
			p6 = src.item((j,   i+1))
			p7 = src.item((j-1, i+1))
			p8 = src.item((j-1, i))
			p9 = src.item((j-1, i-1))
			C = ((~p2 & (p3 | p4)) + (~p4 & (p5 | p6)) + (~p6 & (p7 | p8)) + (~p8 & (p9 | p2)))
			N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8)
			N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9)
			N = min(N1, N2)
			if iter==0:
				m = (p8 & (p6 | p7 | ~p9))
			else:
				m = (p4 & (p2 | p3 | ~p5))
			if C==1 and 2<= N <=3 and m==0:
				marker.itemset((j,i),0)
				changed += 1
		return src & marker, changed

	dst = src.copy()
	i=0;
	while True:
		i+=1
		dst, changed  = iteration(dst, 0)
		dst, changed2 = iteration(dst, 1)

		d = changed + changed2
		if d == 0:
			break
			
	return dst

# ASCII art synthesis using DeepAA
# Modified from source: https://github.com/OsciiArt/DeepAA

def load_model(model_path = "DeepAA/model_light.json", weights_path = "DeepAA/weight_light.hdf5"):
  json_string = open(model_path).read()
  model = model_from_json(json_string)
  model.load_weights(weights_path)
  model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

  return model

model = load_model() # Preload the model

# Resize the image maintaining the aspect ratio
def resize_image(img, new_width=0):
  if new_width==0:
    new_width = img.size[0]

  new_height = int(img.size[1] * new_width / img.size[0])
  img = img.resize((new_width, new_height), Image.LANCZOS)
  img = np.array(img)

  if img.shape == 3:
    img = img[:, :, 0]

  return img

# Add a calculated padding to the image
def add_margin(img):
  input_shape = [64, 64, 1]
  margin = (input_shape[0] - 18) // 2

  height = img.shape[0] + 2 * margin + 18
  width  = img.shape[1] + 2 * margin + 18
  img_new = np.ones((height, width), dtype=np.uint8) * 255
  img_new[margin : margin + img.shape[0], margin : margin + img.shape[1]] = img

  return img_new

def get_char_dict(path="DeepAA/char_dict.pkl"):
  with open(path, mode='rb') as infile:
      char_dict = pickle.load(infile)

  return char_dict

def get_char_list(path="DeepAA/char_list.csv"):
  char_list = pd.read_csv(path, encoding="cp932")
  char_list = char_list[char_list['frequency']>=10]
  char_list = char_list['char'].to_numpy()

  return char_list

def get_predictions(img, model, char_dict, char_list):
  input_shape = [64, 64, 1]
  num_line = (img.shape[0] - input_shape[0]) // 18
  predicts = []

  for h in range(num_line):
      w = 0
      penalty = 1
      predict_line = []
      while w <= img.shape[1] - input_shape[1]:
          input_img = img[h*18 : h*18+input_shape[0], w : w+input_shape[1]]
          input_img = input_img.reshape([1, input_shape[0], input_shape[1], 1])
          predict = model.predict(input_img)

          if penalty: 
            predict[0, 1] = 0

          predict = np.argmax(predict[0])
          penalty = (predict == 1)
          char = char_list[predict]

          predict_line.append(char)

          char_width = char_dict[char].shape[1]
          w += char_width
          
      predicts.append(predict_line)

  return predicts

def generate_ascii_art(img, new_width=0, save_img=False, save_txt=False):
  img = Image.fromarray(img)
  
  img = resize_image(img, new_width=new_width)
  img = add_margin(img)
  img = (img.astype(np.float32)) / 255
  
  char_dict = get_char_dict()
  char_list = get_char_list()

  predicts = get_predictions(img, model, char_dict, char_list)

  input_shape = [64, 64, 1]
  num_line = (img.shape[0] - input_shape[0]) // 18
  img_aa = np.ones_like(img, dtype=np.uint8) * 255
  
  widths = []
  for h in range(num_line):
      w = 0
      for char in predicts[h]:
          char_width = char_dict[char].shape[1]
          char_img = 255 - char_dict[char].astype(np.uint8) * 255
          img_aa[h*18:h*18+16, w:w+char_width] = char_img
          w += char_width
      widths.append(w)
  
  img_aa = img_aa[0:(num_line-1)*18+16, 0:max(widths)]

  if save_img:
    img_aa = Image.fromarray(img_aa)
    img_aa.save('ascii-art.png')

  if save_txt:
    with open('ascii-art.txt', 'w', encoding="utf-8") as f:
      for each in predicts:
        f.write("".join(each) + '\r\n')

  return img_aa


if __name__ == "__main__":
  img_path = "sample-images/sample-image (24).jpg"

  img = cv2.imread(img_path)
  # img = add_margin(img)

  edges = detect_edges(img, threshold=100)
  thinned_edges = 255 - 255 * thin_edges(edges)
 
  artwork = generate_ascii_art(thinned_edges, new_width=550)
  
  show_image(img, "original image")
  show_image(edges, "edges")
  show_image(thinned_edges, "thinned edges")
  show_image(artwork, "ascii art")
