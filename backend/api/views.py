from django.shortcuts import render
from django.contrib import messages
from django.contrib.auth.models import User
from django.conf import settings
from rest_framework import generics, response, status
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.decorators import api_view
import pandas as pd
import os
from django.views.decorators.csrf import csrf_exempt
import pickle
from .serializers import UserSerializer
from .utility import EmergencyClassi
from django.http import JsonResponse
import json
import logging

logger = logging.getLogger(__name__)

class CreateUserView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            self.perform_create(serializer)
            headers = self.get_success_headers(serializer.data)
            return response.Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)
        logger.error(f"User creation failed: {serializer.errors}")
        return response.Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request, *args, **kwargs):
        return response.Response({"message": "Use POST method to create a new user"}, status=status.HTTP_200_OK)

@api_view(['GET', 'POST'])
def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.data.get('loginname')
        pswd = request.data.get('pswd')
        logger.info(f"Login attempt with ID: {loginid}")
        try:
            check = User.objects.get(username=loginid)
            if check.check_password(pswd):
                if check.is_active:
                    request.session['id'] = check.id
                    request.session['loggeduser'] = check.username
                    request.session['loginid'] = loginid
                    request.session['email'] = check.email
                    logger.info(f"User {loginid} logged in successfully.")
                    return JsonResponse({"message": "Login successful", "user": check.username})
                else:
                    logger.warning(f"Login attempt for inactive account: {loginid}")
                    return JsonResponse({"error": "Your account is not activated"}, status=403)
            else:
                logger.warning(f"Invalid password for user: {loginid}")
                return JsonResponse({"error": "Invalid login id or password"}, status=401)
        except User.DoesNotExist:
            logger.warning(f"Login attempt failed for non-existent user: {loginid}")
            return JsonResponse({"error": "Invalid login id or password"}, status=401)
    elif request.method == "GET":
        return JsonResponse({"message": "Please use POST method for login"})

@api_view(['GET'])
def usersViewDataset(request):
    dataset = os.path.join(settings.MEDIA_ROOT, 'EmergencyDataset.csv')
    df = pd.read_csv(dataset)
    data = df.to_dict(orient='records')
    return JsonResponse(data, safe=False)

@api_view(['GET'])
def userClassificationResults(request):
    try:
        rf_report = EmergencyClassi.process_randomForest()
        dt_report = EmergencyClassi.process_decesionTree()
        nb_report = EmergencyClassi.process_naiveBayes()
        gb_report = EmergencyClassi.process_knn()
        lg_report = EmergencyClassi.process_LogisticRegression()
        svm_report = EmergencyClassi.process_SVM()

        reports = {
            'lg': pd.DataFrame(lg_report).transpose().to_dict(),
            'svm': pd.DataFrame(svm_report).transpose().to_dict(),
            'rf': pd.DataFrame(rf_report).transpose().to_dict(),
            'dt': pd.DataFrame(dt_report).transpose().to_dict(),
            'nb': pd.DataFrame(nb_report).transpose().to_dict(),
            'gb': pd.DataFrame(gb_report).transpose().to_dict(),
        }

        return JsonResponse(reports)
    except Exception as e:
        logger.error(f"Error generating classification results: {str(e)}")
        return JsonResponse({'error': 'Error generating classification results'}, status=500)

@api_view(['GET', 'POST'])
@csrf_exempt
def UserPredictions(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            logger.info(f"Received prediction request data: {data}")

            required_params = [
                'age', 'gender', 'pulse', 'systolicBloodPressure', 'diastolicBloodPressure',
                'respiratoryRate', 'spo2', 'randomBloodSugar', 'temperature'
            ]
            for param in required_params:
                if param not in data:
                    return JsonResponse({'error': f"Missing parameter: {param}"}, status=400)

            test_data = [
                int(data['age']), int(data['gender']), int(data['pulse']),
                int(data['systolicBloodPressure']), int(data['diastolicBloodPressure']),
                int(data['respiratoryRate']), float(data['spo2']),
                int(data['randomBloodSugar']), float(data['temperature'])
            ]

            logger.info(f"Test data: {test_data}")

            model_path = os.path.join(settings.MEDIA_ROOT, 'alexmodel.pkl')
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)

            logger.info("Model loaded successfully")

            result = model.predict([test_data])
            logger.info(f"Prediction result: {result}")

            msg = 'Level 2' if result[0] == 0 else 'Level 1'

            response_data = {'prediction': msg}
            return JsonResponse(response_data)

        except ValueError as ve:
            logger.error(f"ValueError: {str(ve)}")
            return JsonResponse({'error': f"Invalid input: {str(ve)}"}, status=400)
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    
    elif request.method == 'GET':
        return JsonResponse({'message': 'Use POST method to make predictions'}, status=200)

@api_view(['GET'])
def api_root(request):
    return JsonResponse({
        'user_register': request.build_absolute_uri('/api/user/register/'),
        'user_login': request.build_absolute_uri('/api/user/login/'),
        'view_dataset': request.build_absolute_uri('/api/dataset/'),
        'classification_results': request.build_absolute_uri('/api/classification/'),
        'predictions': request.build_absolute_uri('/api/predictions/'),
    })
