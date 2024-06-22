from django.urls import path
from . import views

urlpatterns = [
    path("user/register/", views.CreateUserView.as_view(), name="register"),
    path("user/login/", views.UserLoginCheck, name="login"),
    path("view-dataset/", views.usersViewDataset, name="view-dataset"),
    path("userClassificationResults/", views.userClassificationResults, name="user-classification-results"),
    path("predictions/", views.UserPredictions, name="predictions"),
    path("error/", views.UserPredictions, name="error")
]
