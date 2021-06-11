import os
from os.path import join, basename, exists
import numpy as np
import logging
import argparse
import random
from random import randrange,choice,randint
import pickle
from glob import glob
from shutil import copyfile
import cv2

_LOGGER = logging.getLogger(__name__)

def main(base_dir):
"""
   A basic method for occluding silhouettes with a random rectangle.
   param: base_dir - directory containing silhouettes.

"""
    lowest_dirs = []
    for root,dirs,files in os.walk(base_dir):
        if files and not dirs:
            lowest_dirs.append(root)

    for low_dir in lowest_dirs:

        synthetic_occlusion(low_dir)
        	 	

		
def synthetic_occlusion(frame_dir):
    """
    Add a random rectangule to a silhouette to occlude the gait.
    param: frame_dir - directory of frames to occlude.
    """
    f_split = frame_dir.split("/")

    var_num = f_split[-2].split("-")[1]

    path = join(f_split[-3],"oc-" + var_num ,f_split[-1])
    out_dir = join("raw_data/occluded_sils", path)
    print (out_dir)
    if not exists(out_dir):
        os.makedirs(out_dir)


    frames = sorted(glob(join(frame_dir,"*.png")))

    img = cv2.imread(frames[0])
    height, width = img.shape[:2]
    
    #initialise random rect parameters
    rand_start_x = randint(0, int(width/3))
    rand_start_y = randint(height/2, height -int(height/3))

    rand_end_x = randint(rand_start_x, width)
    rand_end_y = randint(rand_start_y, height)

    #iterate through frames and occlude each frame
    for x, fr in enumerate(frames):

        frame_name = basename(fr)
        img = cv2.imread(fr)    
    
     #   height, width = img.shape[:2]
        #start_x = randint(0, int(width - width/3))
        #start_y = randint(0, int(height - height/3))
            
        #Modify the start point for each frame to simulate a moving
        #object (or moving person). 
        start_x = rand_start_x + (x * 3)
        start_y = rand_start_y 

        
        end_x = rand_end_x + (x * 3)
        end_y = rand_end_y
         
        
        if end_x > width-1:
            end_x = width-1

        if start_x >= width-1:
            start_x, start_y = None, None
            end_x, end_y = None, None


        start_coord = (start_x, start_y)

        end_coord = (end_x,end_y)

        print ("Start: ", start_coord)

        print ("End: ", end_coord)



        occ_frame = occlude_frame(img,start_coord, end_coord)
   
        out_file = join(out_dir,frame_name)
        cv2.imwrite(out_file, occ_frame)
        
        #_LOGGER.debug("Image written to " +  out_dir)



def occlude_frame(frame,start_coord,end_coord):
"""
    Draw a black rectangle on the frame.
    param: frame - sil frame.
    param: start_coord - start coordinate for rect
    param: end_coord - end coordinate for rect
    return: occluded frame.

"""
    if start_coord[0] == None:
        print ("No coords")

        occ_frame = frame
    
    else:

        occ_frame = cv2.rectangle(frame, start_coord, end_coord,(0,0,0),cv2.FILLED)

    #cv2.imshow("occ",occ_frame)
    
    #cv2.waitKey(0)

    return occ_frame

if __name__=='__main__':

	
    logging.basicConfig(level=logging.DEBUG)	
	
    parser = argparse.ArgumentParser(description='Method for generating basic occlusion on a silhouette.')
	 
    parser.add_argument(
    'base_dir',
    help="Directory that contains the raw silhouettes.")   
   	      
    args = parser.parse_args()
	
    main(args.base_dir)
