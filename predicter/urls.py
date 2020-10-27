from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('test/', views.test, name='train'),
    path('training/', views.train, name='train'),
    path('prediction/', views.predict, name='predict'),
]
