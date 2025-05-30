from django.shortcuts import render,redirect
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from django.http import HttpResponse
import os
from .models import Image

images=['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt']

def index(request):
    return render(request,'index.html')


def result(request):
    if request.method=='POST':
        
        file=request.FILES['file']
        fn=Image(image=file)
        fn.save()
        path=os.path.join('webapp/static/images/',file.name)
        acc=pd.read_csv("webapp/Accuracy.csv")
        
        new_model=load_model("webapp/CNN.h5")
        test_image=image.load_img(path,target_size=(256,256))
        test_image=image.img_to_array(test_image)
        test_image/=255
        a=acc.iloc[0 -1,1]

        test_image=np.expand_dims(test_image,axis=0)
        result=new_model.predict(test_image)
        pred=images[np.argmax(result)]
        print(pred)
        
        return render(request,'result.html',{'text':pred,'path':'static/images/'+file.name,'a':round(a*100,3)})
    return render(request,'result.html')

