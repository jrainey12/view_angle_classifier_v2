import torch
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from os.path import join, isfile
import numpy as np
from PIL import Image
import pickle

class Dataset:
	
	
	
    def __init__(self, 
            base_dir,
            transform
            ):
					
					
        self._base_dir = base_dir
        self._sil_dir = join(self._base_dir, 'sil')
        self._lab_file = join(self._base_dir, 'labels.txt')
        self._pose_dir = join(self._base_dir, 'pose')
        self._transform = transform

        self.imgs = []
        self.poses = []
        self.labels = []
		
		
        with open(self._lab_file, "r") as f:
            lines = f.read().splitlines()
				
        #_LOGGER.info("Lines: %s", lines)
        for i, line in enumerate(lines):
            line_split = line.split(' ')
            #print (line_split)
            _sil = join(self._base_dir, "sil", line_split[0])
	
            _pose = join(self._base_dir, "pose",line_split[1])
			
            _label = int(line_split[2])
			

			
            #_LOGGER.info("Pose: %s", _pose)
            #print ("Pose", _pose)
            assert isfile(_sil)
            assert isfile(_pose)
			
            self.imgs.append(_sil)
            self.poses.append(_pose)
            self.labels.append(_label)
                


        assert len(self.imgs) == len(self.labels)
        assert len(self.poses) == len(self.imgs)

        print('Number of images in dataset %s.' % len(self.imgs))
			

    def __len__(self):
        
        return len(self.imgs)	
				


    def __getitem__(self, index):

        _img = Image.open(self.imgs[index])
        #print ("TEST",self.probe_imgs[index])
        _target = self.labels[index]
        with open(self.poses[index], 'rb') as f:
            #print ("F:",f)
            _pose =  pickle.load(f)
        
        img = self._transform(_img)
		
        #print(type(_pose_pro))
        pose_ten = torch.Tensor(_pose).unsqueeze(0).unsqueeze(0)
#        print ("img shape", img.shape)
#        print ("pose ten shape",pose_ten.shape)
        pad_1 = img.shape[1]-pose_ten.shape[1]
        pad_2 = img.shape[2]-pose_ten.shape[2]
        pad = (int(pad_2/2),int(pad_2/2),int(pad_1/2),int(pad_1/2)+1)
        #print(pad)
        pose = F.pad(input=pose_ten, pad=pad,mode='constant',value=0) 
        #print ("Pose", pose)
        #print ("img shape",pro.shape)
        #print ("pose shape",pro_pose.shape)		
        concat = torch.cat((img,pose),0)
        #concat_gal = torch.cat((gal,gal_pose),0)
        return (concat,_target)#(concat_pro,concat_gal,_target)
		
