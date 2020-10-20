

"""
Created on Wed Mar 18 11:05:46 2020
q
@author: gang3
"""

import numpy as np
import os
#Ignore all warnings on Ipython console
import warnings
warnings.filterwarnings('ignore')
import sys
import tensorflow as tf
import cv2


cap = cv2.VideoCapture("1-2.mp4")
soccerfield=cv2.imread('soccerfield.png')
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:

from utils import label_map_util

from utils import visualization_utils as vis_util

# In[4]:

# What model to download.
#MODEL_NAME = 'new_soccer9'
MODEL_NAME = 'soccer_graph16'#Select which model will you use
 
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
 
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')
 
NUM_CLASSES = 1#Detect only soccerplayers
 


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:

label_map = label_map_util.load_labelmap(os.path.join('C:/Users/gang3/Desktop/models/research/object_detection/data/','object-detection.pbtxt'))
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)



global home_flag
global away_flag


#Detect which side is on Offense or Defense
def display_flow(img, flow, stride=40):    
    
    
    home_flag=0
    away_flag=0
    for index in np.ndindex(flow[::stride, ::stride].shape[:2]):
        pt1 = tuple(i*stride for i in index)
        delta = flow[pt1].astype(np.int32)[::-1]
        pt2 = tuple(pt1 + 10*delta)
        if 2 <= cv2.norm(delta) <= 10:
            #Arrow is Opposite with attacking way
            if pt2>pt1:
                #right arrow ==> away attack
                away_flag+=1
                
            else:
                #left arrow == home attack
                home_flag+=1
                
       
        
        if home_flag>away_flag:
            cv2.putText(img,"Away Team Attack", (320,260), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,220),2)
            
        elif home_flag<away_flag:
            cv2.putText(img,"Home Team Attack",(6,260), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,100,255),2)
            
        elif home_flag==away_flag or (abs(home_flag-away_flag)<=3):
            cv2.putText(img,"Build Up Process", (175,260), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),2)
        else:
            cv2.putText(img,"Calculating.....", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),2)

        
    
    
    norm_opt_flow = np.linalg.norm(flow, axis=2)
    norm_opt_flow = cv2.normalize(norm_opt_flow, None, 0, 1, cv2.NORM_MINMAX)
    
    

    return img
    





# In[10]:
global x
x=list()
home_count=0
away_count=0


_, prev_frame = cap.read()
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_frame = cv2.resize(prev_frame, (0,0), None, 0.5, 0.5)
init_flow = True
formations=["4-5-1","4-4-2","4-3-3","4-2-3-1","3-6-1","3-5-2","3-4-3","5-4-1","5-3-2"]

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
     
     while True:
     
      rett,original=cap.read()
      
      ret, image_np = cap.read()
  
      
      re,draw_roi=cap.read()
      
      
      #cv2.putText(draw_roi,"Original Screen", (0,35), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255),3)

      
      #Set Frame Speed as 60 per second
      cap.set(cv2.CAP_PROP_FPS, int(60))
      
      font = cv2.FONT_HERSHEY_SIMPLEX
      
      status_cap, frame = cap.read()
      frame = cv2.resize(frame, (0,0), None, 0.5, 0.5)
      if not status_cap:
        break
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
      if init_flow:
        opt_flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 
                                                0.5, 5, 13, 10, 5, 1.1, 
                                                cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        init_flow = False
      else:
        opt_flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, opt_flow, 
                                                0.5, 5, 13, 10, 5, 1.1, 
                                                cv2.OPTFLOW_USE_INITIAL_FLOW)
    
      prev_frame = np.copy(gray)
      img=display_flow(frame, opt_flow)
           
      if cv2.waitKey(1) & 0xFF == ord('q'):
               cv2.destroyAllWindows()
               break
    
      
      
             
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      
     
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      
      
      final_score = np.squeeze(scores)    
      count = 0
      for i in range(100):
          if scores is None or final_score[i] > 0.5:
                    count = count + 1
                    
      #print('player numbers = ',count)
      total_count=vis_util.home_count+vis_util.away_count
      
      #print('len of x = ',len(vis_util.x))
      #Show Player number informations on Screen(Original,Image_np(analyzed screen also))
      cv2.putText(image_np,"Total Players = "+str(total_count), (0,65), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255),2)
      cv2.putText(draw_roi,"Detected Players = "+str(count), (0,65), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255),2)

      cv2.putText(image_np,"HOME = "+str(vis_util.home_count), (0,95), cv2.FONT_HERSHEY_PLAIN, 2,(250,255,255),2)
      cv2.putText(image_np,"AWAY = "+str(vis_util.away_count), (0,125), cv2.FONT_HERSHEY_PLAIN, 2,(250,255,255),2)
      
      #Initializing variables of num players
      vis_util.home_count=0
      vis_util.away_count=0
      
      
      
     
     
      
      #Analysis Process starts!
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=1)
      
        
      #Player recognition on  images
      vis_util.visualize_boxes_and_labels_on_image_array2(
          original,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=1)
      
       #Player recognition on original images
      vis_util.visualize_boxes_and_labels_on_image_array3(
          draw_roi,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=1)
      
      xx=vis_util.x
      
      sum_1_2=0
      sum_5_6=0
      
      #Formation info list
      formation_H=list()
      H_result=""
      formation_A=list()
      A_result=""
      RH=""
      RA=""
      orange=vis_util.orange
      blue=vis_util.blue
      #NUM of players in each ROI areas
      if len(xx)!=0:
          #Home Team Number status

          #Whether it is 4,5,3,2 back of strategy
          formation_H.append(vis_util.area_11)
          formation_H.append(vis_util.area_22)
          formation_H.append(vis_util.area_33)
          formation_H.append(vis_util.area_44)
         
         
          
          
          formation_A.append(vis_util.area_4)
          formation_A.append(vis_util.area_3)
          formation_A.append(vis_util.area_2)
          formation_A.append(vis_util.area_1)
          
       

          #Gather Formation Information
          for f in range(4):
              if formation_H[f]!=0:
#                  
                      H_result+=str(formation_H[f])
                      
              if formation_A[f]!=0:
#                  
                      A_result+=str(formation_A[f])
          
            
          #HOME TEAM FORMATION
          if H_result[0]==str(4):#4 back formation
              if H_result[1]==str(5):
                  RH=formations[0]
                 
                  
              elif H_result[1]==str(4):
                  RH=formations[1]
             
                  
              elif H_result[1]==str(3):
                  RH=formations[2]
               
              elif H_result[1]==str(2):
                  RH=formations[3]
             
         
          elif H_result[0]==str(3):
             if H_result[1]==str(6):
                  RH=formations[4]
               
                  
             elif H_result[1]==str(5):
                  RH=formations[5]
                  
                  
             else:
                  RH=formations[6]
                 
                  
          elif H_result[0]==str(5):
              
               if H_result[1]==str(4):
                  RH=formations[7]
                  
                  
               else:# H_result[1]==3 or A_result[1]==3:
                  RH=formations[8]
                  
          else:
              RH="Calculating...."
             
              
              
              
              
              
       #AWAY TEAM FORMATION
          if A_result[0]==str(4):#4 back formation
              if  A_result[1]==str(5):
                 
                  RA=formations[0]
                  
              elif A_result[1]==str(4):
                  
                  RA=formations[1]
                  
              elif A_result[1]==str(3):
                  
                  RA=formations[2]
                  
              elif A_result[1]==str(5):
                  RA=formations[5]
         
          elif A_result[0]==str(3):
              
             if A_result[1]==str(6):
                  
                  RA=formations[4]
                  
             elif A_result[1]==str(5):
                  
                  RA=formations[5]
                  
             else:
                  RA=formations[6]
                  
          elif A_result[0]==str(5):
              
               if A_result[1]==str(4):
             
                  RA=formations[7]
                  
               else:
                 
                  RA=formations[8]
          else:
           
              RA="Calculating...."       
      
      
      if RA == "" or RH=="":
          RA="Calculating"
          RH=RA
          cv2.putText(draw_roi,"H_Formation = "+RA, (200,500), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255),2)
          cv2.putText(draw_roi,"A_Formation = "+RH, (200,550), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255),2)
          cv2.putText(original,"H_Formation = "+RA, (200,500), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255),2)
          cv2.putText(original,"A_Formation = "+RH, (200,550), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255),2)
          cv2.putText(orange,"H_Formation = "+RA, (20,280), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255),2)
          cv2.putText(orange,"A_Formation = "+RH, (280,280), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255),2)
          cv2.putText(blue,"H_Formation = "+RA, (20,280), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255),2)
          cv2.putText(blue,"A_Formation = "+RH, (280,280), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255),2)
      else:
          cv2.putText(draw_roi,"H_Formation = "+RA, (200,500), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255),2)
          cv2.putText(draw_roi,"A_Formation = "+RH, (200,550), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255),2)
          cv2.putText(original,"H_Formation = "+RA, (200,500), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255),2)
          cv2.putText(original,"A_Formation = "+RH, (200,550), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255),2)
          cv2.putText(orange,"H_Formation = "+RA, (20,280), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255),2)
          cv2.putText(orange,"A_Formation = "+RH, (280,280), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255),2)
          cv2.putText(blue,"H_Formation = "+RA, (20,280), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255),2)
          cv2.putText(blue,"A_Formation = "+RH, (280,280), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255),2)
          
      
    
      #Initializing variables of num players
      formation_H.clear()
      formation_A.clear()
      H_result=None
      A_result=None
      vis_util.area_1=0
      vis_util.area_2=0
      vis_util.area_3=0
      vis_util.area_4=0
      vis_util.area_11=0
      vis_util.area_22=0
      vis_util.area_33=0
      vis_util.area_44=0
      xx.clear()#xx 초기화
      vis_util.positions_x.clear()
      vis_util.positions_y.clear()
      vis_util.positions_x2.clear()
      vis_util.positions_y2.clear()
    
     
      
     
           

      
      
      #Final Process
      #Resize results and visualize on screen
      result=image_np
   
      #resize display
      result=cv2.resize(result,(750,500))
      original=cv2.resize(original,(750,500))
      draw_roi=cv2.resize(draw_roi,(750,500))
      img=cv2.resize(img,(750,500))
      orange=cv2.resize(orange,(450,500))
      blue=cv2.resize(blue,(450,500))
      soccerfield=cv2.resize(soccerfield,(450,500))
#      orange=cv2.bitwise_or(orange,soccerfield)
#      blue=cv2.bitwise_or(blue,soccerfield)
      
      
      #concatenate original image & analyzed display
      temp=cv2.hconcat([draw_roi,original])
      temp=cv2.hconcat([temp,orange])
      temp2=cv2.hconcat([result,img])
      temp2=cv2.hconcat([temp2,blue])
      screen_result=cv2.vconcat([temp,temp2])
      cv2.imshow('final result',screen_result)
             
      #Set video restriction as quit button 'q'
      if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        
        break