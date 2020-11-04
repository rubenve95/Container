import cv2
import numpy as np

def get_zone_labels():

    return np.array([[238 ,201, 0],#yellow
                    [205, 92, 92],  #red 
                    [132 ,112 ,255],  #blue
                    [34 ,139 ,34], #green
                    [0,0,0]]) # black

def convert_img():
	img_name = 'IMG_0545.png'
	img = cv2.imread(img_name)
	polygons = []

	for color_index in range(4):
		polygon = get_polygon_contour(img.copy(), color_index)
		if polygon is not None:
			polygons.append(polygon)

	if len(polygons) == 0:	
		print('No polygons found')
	elif len(polygons) == 1:
		print('One polygon')
	elif len(polygons) == 2:
		matched_corners = nearest_neighbour(polygons[0], polygons[1])
		new_p0, new_p1 = merge_polygons(polygons[0], polygons[1], matched_corners)

		p = {}
		p['c0'] = {0: matched_corners[0][0], 1: matched_corners[1][0]}
		p['c1'] = {0: matched_corners[0][1], 1: matched_corners[1][1]}

		polygons = [new_p0, new_p1]
	elif len(polygons) == 3:
		return 

def get_polygon_contour(img, color_index, show=False):

	color = get_zone_labels()[color_index]
	#print(color)
	for i in range(3):
		img[:,:,i][img[:,:,i] != color[i]] = 0

	if not img.any():
		return None

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img, (600,400))

	if show:
		cv2.imshow('yo', img)

	#ret,thresh = cv2.threshold(img,127,255,0)
	thresh = img
	if show:
		cv2.imshow('threshold ',thresh)

	# dilate thresholded image - merges top/bottom 
	kernel = np.ones((3,3), np.uint8)
	dilated = cv2.dilate(thresh, kernel, iterations=3)
	if show:
		cv2.imshow('threshold dilated',dilated)

	# find contours
	contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	# simplify contours
	try:
		epsilon = 0.05*cv2.arcLength(contours[0],True)
	except:
		return None
	iters = 0
	while True:
		if iters == 100:
			return None
		iters += 1
		approx = cv2.approxPolyDP(contours[0],epsilon,True)
		if len(approx) > 4:
			epsilon *= (2**(1/iters))
		elif len(approx) < 4:
			epsilon /= (2**(1/iters))
		else:
			break
	if show:
		cv2.drawContours(img, [approx], 0, (255,255,255), 3)
		cv2.imshow('image',img)
		key = cv2.waitKey(0)
		if key == ord('q') & 0xFF:
			cv2.destroyAllWindows()
			exit()
	approx = approx.reshape(4,2)
	return approx


if __name__=="__main__":
	convert_img()