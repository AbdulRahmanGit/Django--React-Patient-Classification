from django.db import models
from django.contrib.auth.models import User


class UserRegistrationModel(models.Model):
    name = models.CharField(max_length=100, default='Default Name')
    loginid = models.CharField(unique=True, max_length=100,  default='default_loginid')
    password = models.CharField(max_length=100, default='default_password')
    email = models.CharField(unique=True, max_length=100,  default='default@example.com')
    def __str__(self):
        return self.name
