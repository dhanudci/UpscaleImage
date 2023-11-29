from django.shortcuts import render
from django.http import HttpResponse  
# from upscaleapp.functions import handle_uploaded_file  
from upscaleapp.forms import UpscaleForm 
import cv2
from cv2 import dnn_superres
# from django.core.files.storage import FileSystemStorage 
def upscale(request):  
    if request.method == 'POST':  
        upscale = UpscaleForm(request.POST, request.FILES)  
        if upscale.is_valid():
            if 'btn1' in request.POST:
                handle_uploaded_file(request.FILES['file'])
            elif 'btn2' in request.POST:
                handle_uploaded_file1(request.FILES['file'])
                
            elif 'btn3' in request.POST: 
                handle_uploaded_file2(request.FILES['file'])   
            else:
                handle_uploaded_file3(request.FILES['file'])          
    else:  
        upscale = UpscaleForm()  
        return render(request,"index.html",{'form':upscale})  

def handle_uploaded_file(f):  
    with open('input/'+f.name, 'wb+') as destination:  
        for chunk in f.chunks():  
            destination.write(chunk)          
    sr = dnn_superres.DnnSuperResImpl_create()
    path = 'EDSR_x2.pb'
    sr.readModel(path)
    sr.setModel('edsr', 2)
    sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    # path1='upscaleproject/input/'+f.name
    path1='input/'+f.name
    print(path1)
    image = cv2.imread(path1)
    upscaled = sr.upsample(image) 
    with cv2.imwrite('output/'+ 'upscale2_'+f.name,upscaled) as destination:  
        for chunk in f.chunks():  
            destination.write(chunk) 

    # with open('E:\JAYA'+'upscale_'+f.name, 'wb+') as destination:  
    #     for chunk in f.chunks():  
    #         destination.write(chunk) 

def handle_uploaded_file1(f):  
    with open('input/'+f.name, 'wb+') as destination:  
        for chunk in f.chunks():  
            destination.write(chunk)          
    sr = dnn_superres.DnnSuperResImpl_create()
    path = 'EDSR_x3.pb'
    sr.readModel(path)
    sr.setModel('edsr', 3)
    sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    # path1='upscaleproject/input/'+f.name
    path1='input/'+f.name
    print(path1)
    image = cv2.imread(path1)
    upscaled = sr.upsample(image) 
    with cv2.imwrite('output/'+ 'upscale3_'+f.name,upscaled) as destination:  
        for chunk in f.chunks(): 
            destination.write(chunk)  

def handle_uploaded_file2(f):  
    with open('input/'+f.name, 'wb+') as destination:  
        for chunk in f.chunks():  
            destination.write(chunk)          
    sr = dnn_superres.DnnSuperResImpl_create()
    path = 'EDSR_x4.pb'
    sr.readModel(path)
    sr.setModel('edsr', 4)
    sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    # path1='upscaleproject/input/'+f.name
    path1='input/'+f.name
    print(path1)
    image = cv2.imread(path1)
    upscaled = sr.upsample(image) 
    with cv2.imwrite('output/'+ 'upscale4_'+f.name,upscaled) as destination:  
        for chunk in f.chunks():  
            destination.write(chunk) 

def handle_uploaded_file3(f):  
    with open('input/'+f.name, 'wb+') as destination:  
        for chunk in f.chunks():  
            destination.write(chunk)          
    sr = dnn_superres.DnnSuperResImpl_create()
    path = 'LapSRN_x8.pb'
    sr.readModel(path)
    sr.setModel('lapsrn', 8)
    sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    # path1='upscaleproject/input/'+f.name
    path1='input/'+f.name
    print(path1)
    image = cv2.imread(path1)
    upscaled = sr.upsample(image) 
    with cv2.imwrite('output/'+ 'upscale8_'+f.name,upscaled) as destination:  
        for chunk in f.chunks():  
            destination.write(chunk) 






              
                                 
