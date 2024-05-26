from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import onnxruntime
import numpy as np
from PIL import Image

imageClassList = {'0': 'шашлык', '1': 'цыпленок', '2': 'стейк'}  #Сюда указать классы
imageDescriptionList = {'шашлык': 'Шашлык - это традиционное блюдо кавказской кухни, состоящее из кусков мяса, обычно баранины или говядины, пропитанных специями и жареных на мангале. Это блюдо обладает богатым ароматом и сочным вкусом, благодаря особой технике приготовления и использованию пряностей, таких как перец, кумин и лук. Шашлык - это популярный выбор для праздничных ужинов и семейных посиделок на свежем воздухе.'
                        , 'стейк': 'Стейк - это изысканное блюдо, состоящее из отборных кусков мяса, обычно говядины или свинины, приготовленных на гриле или сковороде. Особенность стейка заключается в его неповторимом вкусе и нежности, достигаемой благодаря правильному выбору и подготовке мяса. Подается стейк часто с соусами на выбор и гарниром из овощей или картофеля. Это блюдо является символом роскоши и изысканности в кулинарии.'
                        , 'цыпленок': 'Цыпленок - это популярное блюдо, приготовленное из молодой курицы, нежного и сочного мяса. Он может быть приготовлен различными способами, включая запекание, жарку или варку. Цыпленок отличается своим мягким вкусом и сочной текстурой, а также способностью принимать на себя разнообразные пряности и приправы. Это универсальное блюдо подходит как для семейного обеда, так и для праздничного застолья.'
                        }
imageColorList = {'шашлык': 'Коричневый', 'стейк': 'Розовый', 'цыпленок': 'Белый'}  #Сюда указать цвета
imageTimeList = {'шашлык': '1-2 часа', 'стейк': '10-15 минут', 'цыпленок': '45-60 минут'}  #Сюда указать время суток

def scoreImagePage(request):
    return render(request, 'scorepage.html')

def predictImage(request):
    if 'filePath' not in request.FILES:
        error_message = "Пожалуйста, выберите файл для загрузки первым"
        return render(request, 'scorepage.html', {'error_message': error_message})
    
    fileObj = request.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save('images/'+fileObj.name,fileObj)
    filePathName = fs.url(filePathName)
    modelName = request.POST.get('modelName')
    scorePrediction = predictImageData(modelName, '.'+filePathName)
    context = {'scorePrediction': scorePrediction, 
               'uploaded_image': filePathName, 
               'DescriptionPrediction': imageDescriptionList[scorePrediction], 
               'color' : imageColorList[scorePrediction] , 
               'time' : imageTimeList[scorePrediction]}  # 将上传的图片路径传递给模板
    return render(request, 'scorepage.html', context)

def predictImageData(modelName, filePath):
    img = Image.open(filePath).convert("RGB")
    img = np.asarray(img.resize((32, 32), Image.LANCZOS))
    sess = onnxruntime.InferenceSession(r'media/models/cifar100.onnx') #<-Здесь требуется указать свой путь к модели
    outputOFModel = np.argmax(sess.run(None, {'input': np.asarray([img]).astype(np.float32)}))
    score = imageClassList[str(outputOFModel)]
    return score