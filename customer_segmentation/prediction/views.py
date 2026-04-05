from django.shortcuts import render
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
from customer_segmentation import settings
import os
import json

predict_pipeline = PredictPipeline()

def index(request):
    return render(request, 'index.html')

def result(request):
    if request.method == "POST":
        # Get all input values from POST
        spending_score = request.POST.get('spending_score')
        annual_income = request.POST.get('annual_income')

        data = CustomData(
            annual_income=float(annual_income),
            spending_score=float(spending_score)
        )
        pred_df = data.get_data_as_data_frame()
        # predict_pipeline = PredictPipeline()
        predicted_cluster = int(predict_pipeline.predict(pred_df)[0])
        cluster_name = cluster_label(predicted_cluster)

    return render(request, 'result.html', {
        "predicted_cluster": int(predicted_cluster),
        "cluster_name": cluster_name
    })

def cluster_label(predicted_cluster):
    cluster_labels_path = os.path.join(settings.ML_ARTIFACTS_DIR, 'cluster_labels.json')
    with open(cluster_labels_path, 'r', encoding='utf-8') as labels_file:
        cluster_name_map = json.load(labels_file)
    return cluster_name_map.get(str(predicted_cluster), "Unknown Cluster")