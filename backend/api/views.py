from django.shortcuts import render
from django.contrib import messages
from django.contrib.auth.models import User
from django.conf import settings
from rest_framework import generics
from rest_framework import response
from rest_framework.permissions import IsAuthenticated, AllowAny
import pandas as pd
import os
from django.views.decorators.csrf import csrf_exempt
import pickle
from .serializers import UserSerializer
from .utility import EmergencyClassi
from django.http import JsonResponse
import json
class CreateUserView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [AllowAny]

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID =", loginid, 'Password =', pswd)
        try:
            check = User.objects.get(username=loginid)
            if check.check_password(pswd):
                if check.is_active:
                    request.session['id'] = check.id
                    request.session['loggeduser'] = check.username
                    request.session['loginid'] = loginid
                    request.session['email'] = check.email
                    print("User id At", check.id)
                    return render(request, 'users/UserHome.html', {})
                else:
                    messages.success(request, 'Your account is not activated')
                    return render(request, 'UserLogin.html')
            else:
                messages.success(request, 'Invalid login id and password')
        except User.DoesNotExist:
            messages.success(request, 'Invalid login id and password')
    return render(request, 'UserLogin.html', {})


def usersViewDataset(request):
    dataset = os.path.join(settings.MEDIA_ROOT, 'EmergencyDataset.csv')
    df = pd.read_csv(dataset)
    data = df.to_dict(orient='records')  # Correct argument to convert DataFrame to list of dictionaries
    return JsonResponse(data, safe=False)

 # Adjust the import as needed

def userClassificationResults(request):
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


@csrf_exempt
def UserPredictions(request):
    if request.method == 'POST':
        try:
            # Extract and log received POST data
            received_data = json.loads(request.body.decode('utf-8'))
            print('Received POST data:', received_data)
            # Extract and validate each parameter
            data = json.loads(request.body)
            age = data.get("age")
            gender = data.get('gender')
            pulse = data.get('pulse')
            systolicBloodPressure = data.get('systolicBloodPressure')
            diastolicBloodPressure = data.get('diastolicBloodPressure')
            respiratoryRate = data.get('respiratoryRate')
            spo2 = data.get('spo2')
            randomBloodSugar = data.get('randomBloodSugar')
            temperature = data.get('temperature')
            # Check for missing parameters
            missing_params = [param for param in [
                ('age', age), ('gender', gender), ('pulse', pulse), 
                ('systolicBloodPressure', systolicBloodPressure), 
                ('diastolicBloodPressure', diastolicBloodPressure), 
                ('respiratoryRate', respiratoryRate), ('spo2', spo2), 
                ('randomBloodSugar', randomBloodSugar), ('temperature', temperature)
            ] if param[1] is None]

            if missing_params:
                return JsonResponse({'error': f"Missing parameters: {[param[0] for param in missing_params]}"}, status=400)

            # Convert parameters to appropriate types
            age = int(age)
            gender = int(gender)
            pulse = int(pulse)
            systolicBloodPressure = int(systolicBloodPressure)
            diastolicBloodPressure = int(diastolicBloodPressure)
            respiratoryRate = int(respiratoryRate)
            spo2 = float(spo2)
            randomBloodSugar = int(randomBloodSugar)
            temperature = float(temperature)

            test_data = [
                age, gender, pulse, systolicBloodPressure, diastolicBloodPressure,
                respiratoryRate, spo2, randomBloodSugar, temperature
            ]

            # Log the test data
            print("Test data:", test_data)

            model_file = 'alexmodel.pkl'
            model_path = os.path.join(settings.BASE_DIR, 'media', model_file)

            # Log the model path
            print("Model path:", model_path)

            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)

            # Log model loading success
            print("Model loaded successfully")

            result = model.predict([test_data])

            # Log the prediction result
            print("Prediction result:", result)

            msg = 'Level 2' if result[0] == 0 else 'Level 1'

            response_data = {
                'prediction': msg,
            }

            return JsonResponse(response_data)

        except ValueError as ve:
            # Log the error message
            print("ValueError:", str(ve))
            return JsonResponse({'error': f"Invalid input: {str(ve)}"}, status=400)
        except Exception as e:
            # Log the error message
            print("Error:", str(e))
            return JsonResponse({'error': str(e)}, status=500)
    
    else:
        return JsonResponse({'error': 'GET method not supported'}, status=405)