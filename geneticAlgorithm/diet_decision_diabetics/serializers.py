from rest_framework import serializers
from diet_decision_diabetics.models import Food

class FoodsSerializers(serializers.ModelSerializer):
  class Meta:
    model = Food
    fields = ('__all__')
    #fields = ('id', 'user', 'address', 'phone', 'email', 'edad')