from styx_msgs.msg import TrafficLight
from keras.models import load_model
import numpy as np
import tensorflow as tf
import cv2
import rospy


class TLClassifier(object):
    def __init__(self):
        self.testing_location = rospy.get_param('/testing_location')       #TODO load classifier
        if self.testing_location == 'simulator':
            self.get_classification = self.get_classification_simulator
        elif self.testing_location == 'site':
	    self.model = load_model('light_classification/_tf_lights_classifier.h5') 
            self.model._make_predict_function()
            self.graph = tf.get_default_graph()
            self.get_classification = self.get_classification_site

    def get_classification_simulator(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """

        r_channel = image[:,:,2]
        g_channel = image[:,:,1]



        # Threshold color channel
        s_rgy_min = 50
        s_thresh_min = 245
        s_thresh_max = 255
   
        #s_binary = np.zeros_like(r_channel)
        r_binary = np.zeros_like(r_channel)
        g_binary = np.zeros_like(r_channel)
        y_binary = np.zeros_like(r_channel)
   
        #s_binary[((r_channel >= s_thresh_min) & (r_channel <= s_thresh_max)) | ((g_channel >= s_thresh_min) & (g_channel <= s_thresh_max))] = 1
   
   
        r_binary[((r_channel >= s_thresh_min) & (r_channel <= s_thresh_max)) & (g_channel <= s_rgy_min)] = 1
        g_binary[((g_channel >= s_thresh_min) & (g_channel <= s_thresh_max)) & (r_channel <= s_rgy_min)] = 1
        y_binary[((r_channel >= s_thresh_min) & (r_channel <= s_thresh_max)) & ((g_channel >= s_thresh_min) & (g_channel <= s_thresh_max))] = 1
   

        #res = cv2.bitwise_and(img,img,mask = s_binary)
   
        #maxx=image.shape[1]
        maxy=image.shape[0]
   
        y_top=0
        window_size_y=50
        y_bottom=y_top+window_size_y
   
        max_color=0
        tf_color=TrafficLight.UNKNOWN
   
        while (y_bottom< maxy):
            #print(img[y_top:y_bottom,:,:])
            rs= r_binary[y_top:y_bottom,:].sum()
            gs= g_binary[y_top:y_bottom,:].sum()
            ys= y_binary[y_top:y_bottom,:].sum()
            if (rs>max_color):
                max_color=rs
                tf_color=TrafficLight.RED
            if (gs>max_color):
                max_color=gs
                tf_color=TrafficLight.GREEN
            if (ys>max_color):
                max_color=ys
                tf_color=TrafficLight.YELLOW
            y_top+=window_size_y
            y_bottom+=window_size_y
   
        if (max_color<100):
            tf_color=TrafficLight.UNKNOWN
       


        return tf_color

    def get_classification_site(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
   
	img=cv2.resize(image,(224,224))
	img=img/255.0
	img = np.expand_dims(img, axis=0)
        with self.graph.as_default():
	    pred=self.model.predict(img)
	pclass=np.argmax(pred)

   	tf_color=TrafficLight.UNKNOWN
        if (pclass==1):
	    tf_color=TrafficLight.RED
        elif (pclass==2):
	    tf_color=TrafficLight.GREEN

        return tf_color

