#coding=utf-8

import os
import numpy as np
import sys
caffe_root = '/home/xqianxin/workspace/caffe-LOCAL/'
sys.path.insert(0,caffe_root + 'python')
import caffe

class AgeAndGenderClassifier:
    def __init__(self):
        self.age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
        self.gender_list=['Male','Female']
        print 'Initing'
    
    def setModel(self):
        mean_filename = './mean.binaryproto'
        proto_data = open(mean_filename, 'rb').read()
        a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
        mean = caffe.io.blobproto_to_array(a)[0]
        
        #loading Age network
        age_net_pretrained = './age_net.caffemodel'
        age_net_model_file = './deploy_age.prototxt'
        self.age_net = caffe.Classifier(age_net_model_file, age_net_pretrained,
                               mean=mean,
                               channel_swap=(2,1,0),
                               raw_scale=255,
                               image_dims=(256, 256))
        self.age_net.set_mode_gpu()

        #loading Gender network
        gender_net_pretrained = './gender_net.caffemodel'
        gender_net_model_file = './deploy_gender.prototxt'
        self.gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,
                               mean=mean,
                               channel_swap=(2,1,0),
                               raw_scale=255,
                               image_dims=(256, 256))
        #self.gender_net.set_mode_gpu()                       
    def getAgeAndGender(self, imgfile):
        
        
        #if is image file, like *.jpg
        #'''
        if str(type(imgfile)) == "<type 'str'>":
            input_image = caffe.io.load_image(imgfile)
            #print input_image
        #'''
        else:
            #img = imgfile.astype(np.float32)
            input_image = imgfile / 255.0
            input_image = input_image[:,:,(2,1,0)]
            #print input_image
        #predicting age
        age_prediction = self.age_net.predict([input_image])
        age = self.age_list[age_prediction[0].argmax()]
        
        #predicting gender
        gender_prediction = self.gender_net.predict([input_image])
        gender = self.gender_list[gender_prediction[0].argmax()]
        
        return (age,gender)
