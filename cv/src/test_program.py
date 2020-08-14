from evaluating.model_evaluation import *
# from processing.data_processing import *
from evaluating.azure_model_evaluation import *

# evaluate_sample_model("https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTj2axolJYlhVluFbluLQ39fTDwTXHlp7botA&usqp=CAU")
# evaluate_model_smile("https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTj2axolJYlhVluFbluLQ39fTDwTXHlp7botA&usqp=CAU")
# evaluate_fastai_model("https://image.shutterstock.com/image-photo/portrait-sad-man-260nw-126009806.jpg")
# evaluate_fastai_model1("/Users/haiho/Internship/computervision/data/original/ravdess/Actor_02/01-01-03-02-01-02-02.mp4")
image_path = 'https://raw.githubusercontent.com/Microsoft/Cognitive-Face-Windows/master/Data/detection1.jpg'
evaluate_azure_model(image_path)
# scrap_LinkedIn_profile('http://www.linkedin.com/in/haiho91/')