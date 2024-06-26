from django.shortcuts import render
from django.contrib import messages
from django.contrib.auth.models import User
from django.conf import settings
from rest_framework import generics, response
from rest_framework.permissions import IsAuthenticated, AllowAny
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

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
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
                    return render(request, 'users/UserHome.html', {})
                else:
                    messages.success(request, 'Your account is not activated')
                    return render(request, 'UserLogin.html')
            else:
                messages.success(request, 'Invalid login id and password')
        except User.DoesNotExist:
            messages.success(request, 'Invalid login id and password')
            logger.warning(f"Login attempt failed for non-existent user: {loginid}")
    return render(request, 'UserLogin.html', {})

def usersViewDataset(request):
    dataset = os.path.join(settings.MEDIA_ROOT, 'EmergencyDataset.csv')
    df = pd.read_csv(dataset)
    data = df.to_dict(orient='records')
    return JsonResponse(data, safe=False)

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

@csrf_exempt
def UserPredictions(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            logger.info(f"Received prediction request data: {data}")

            # Validate and convert parameters
            required_params = [
                'age', 'gender', 'pulse', 'systolicBloodPressure', 'diastolicBloodPressure',
                'respiratoryRate', 'spo2', 'randomBloodSugar', 'temperature'
            ]
            for param in required_params:
                if param not in data:
                    return JsonResponse({'error': f"Missing parameter: {param}"}, status=400)

            # Convert to appropriate types
            age = int(data['age'])
            gender = int(data['gender'])
            pulse = int(data['pulse'])
            systolicBloodPressure = int(data['systolicBloodPressure'])
            diastolicBloodPressure = int(data['diastolicBloodPressure'])
            respiratoryRate = int(data['respiratoryRate'])
            spo2 = float(data['spo2'])
            randomBloodSugar = int(data['randomBloodSugar'])
            temperature = float(data['temperature'])

            test_data = [
                age, gender, pulse, systolicBloodPressure, diastolicBloodPressure,
                respiratoryRate, spo2, randomBloodSugar, temperature
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
    
    else:
        return JsonResponse({'error': 'GET method not supported'}, status=405)
