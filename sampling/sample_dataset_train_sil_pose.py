import os
from os.path import join, basename, exists
import numpy as np
import logging
import argparse
import random
from random import randrange,choice
import pickle
from glob import glob
from shutil import copyfile


_LOGGER = logging.getLogger(__name__)

def main(base_dir):
    
    number_of_samples = 128000
    number_of_subjects = 94

    sil_out_dir = "dataset/train/sil" 
    pose_out_dir = "dataset/train/pose"
    label_dir = "dataset/train/"

    if not exists(label_dir):
            os.makedirs(label_dir)

    if not exists(sil_out_dir):
           os.makedirs(sil_out_dir)

    if not exists(pose_out_dir):
           os.makedirs(pose_out_dir)
	
	
    with open(join(label_dir,"labels.txt"), 'a') as f:

        for x in range(0, int(number_of_samples)):
		
            #set seed to x to get different random choice for each sample
            #also makes the sampling repeatable with the same data.
            random.seed(x)
                    
            _LOGGER.debug("Sample number: %d" % (x+1))
			
	    #get samples and label
            sample, label = get_sample(base_dir,number_of_subjects)
	    #copy image to dataset
            copyfile(sample[0], join(sil_out_dir,"%03d.png" % (x + 1) ))
	    #save pose data to pkl output	
            output = open(join(pose_out_dir,"%03d.pkl" % (x + 1)), 'wb')
            pickle.dump(sample[1], output)
            output.close()			

            f.write("%03d.png " % (x + 1) + "%03d.pkl " % (x + 1) + label + "\n")
		
def get_sample(base_dir,number_of_subjects):
	
	
    variations = ['nm-01', 'nm-02','nm-03','nm-04','nm-05','nm-06']

    angles = ['000','018','036','054','072','090','108','126','144','162','180']


    empty = True

    while empty:

        empty = False
		
        subject = '%03d' % randrange(1,number_of_subjects+1)
        var = choice(variations)
        ang = choice(angles)

        path = join(subject, var, ang)
	
        #load pkl file for pose
        pose_dir = join(base_dir,"pose/train", path + ".pkl")
        if not exists(pose_dir):
            empty = True
            _LOGGER.debug("The pose dir is empty.")
            continue

        all_poses = load_pose(pose_dir)
        #print (len(all_poses))
        #randomly select a frame
        #pose = choice(all_poses)
        pose_idx = randrange(0,len(all_poses))
        pose = all_poses[pose_idx]
        #select the related sil frame
        img_dir = join(base_dir,"sils/train", path) 
        
        if not exists(img_dir):
            empty = True
            _LOGGER.debug("The sil is empty.")
            continue		
 
        all_imgs = sorted(glob(join(img_dir,"*.png")))
       
        print ("pose idx",pose_idx)
        print ("pose len",len(all_poses))
        print ("imgs len",len(all_imgs))

        #Sometimes the sils are a different length than the poses due to interp.
        #to ensure an error doesnt occur, if the pose list is longer the last
        #element in the imgs is used.
        if pose_idx < len(all_imgs):
        
            img = all_imgs[pose_idx]
   
        else:
                        
            img = all_imgs[-1]

       				
	#print ("pose 1 :" ,pose_1)
	
        
    sample = [img,pose]
    label = str(angles.index(ang))
    return sample, label





def load_pose(pose_dir):
	
    with open(pose_dir, 'rb') as f:

        pose_data = pickle.load(f, encoding="latin1")
    
    return pose_data 

if __name__=='__main__':

	
    logging.basicConfig(level=logging.DEBUG)	
	
    parser = argparse.ArgumentParser(description='Sampling of silhouettes and poses for training angle detector.')
	 
    parser.add_argument(
    'base_dir',
    help="Directory that contains the raw training data.")   
   	      
    args = parser.parse_args()
	
    main(args.base_dir)
