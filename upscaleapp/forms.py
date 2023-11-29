from django import forms  
class UpscaleForm(forms.Form):  
    file= forms.FileField() 
    # upscale = forms.CharField()
    # upscaled= [
    # ('x2', 'x2'),
    # ('x3', 'x3'),
    # ('x4', 'x4'),
    # ]
    # upscale= forms.CharField()