from django.urls import path
from . import views

app_name = "ml_tools"
urlpatterns = [
	path("", views.list_all, name="list_all"),
	path("covid19/", views.covid19, name="covid19"),
	path("heartdisease/", views.heartdisease, name="heartdisease"),
	path("breastcancer/", views.breastcancer, name="breastcancer"),
	path("malaria/", views.malariadetect, name="malaria"),

]
