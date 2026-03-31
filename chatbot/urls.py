from django.urls import path
from . import views



app_name = 'chatbot'

urlpatterns = [
    path('', views.home, name='home'),
    path('get_bot_response/', views.get_bot_response, name='get_bot_response'),
    
    
    

]
