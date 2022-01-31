from django.db import models
import uuid


# Create your models here.
class MalariaImage(models.Model):
	id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
	image = models.ImageField(verbose_name="Malaria Image",
	                          upload_to="static/ml_tools/images/malaria_uploads/", default=None, null=True)

class Covid19Image(models.Model):
	id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
	image = models.ImageField(verbose_name="Covid19 Image",
	                          upload_to="static/ml_tools/images/covid_uploads/", default=None, null=True)
