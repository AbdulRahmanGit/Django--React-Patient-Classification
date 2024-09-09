from django.urls import path
from . import views

urlpatterns = [
    path('', views.api_root, name='api-root'),
    path('user/register/', views.CreateUserView.as_view(), name='user-register'),
    path('user/login/', views.UserLoginCheck, name='user-login'),
    path('dataset/', views.usersViewDataset, name='view-dataset'),
    path('classification/', views.userClassificationResults, name='classification-results'),
    path('predictions/', views.UserPredictions, name='predictions'),
]
