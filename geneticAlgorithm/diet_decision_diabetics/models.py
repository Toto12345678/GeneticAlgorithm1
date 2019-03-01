from django.db import models

# Create your models here.
class Food(models.Model):
    food_group = models.CharField(max_length=50)
    food_name = models.CharField(max_length=100)
    kcal = models.FloatField()
    protein = models.FloatField()
    fat = models.FloatField()
    carbs = models.FloatField()
    size = models.FloatField()

    class Meta:
        db_table = 'foods'