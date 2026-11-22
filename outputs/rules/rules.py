def findDecision(obj): #obj[0]: sepal-length, obj[1]: sepal-width, obj[2]: petal-length, obj[3]: petal-width
   # {"feature": "petal-length", "instances": 120, "metric_value": 1.585, "depth": 1}
   if obj[2]>1.9927933150920052:
      # {"feature": "petal-width", "instances": 80, "metric_value": 1.0, "depth": 2}
      if obj[3]<=1.7:
         # {"feature": "sepal-width", "instances": 42, "metric_value": 0.3712, "depth": 3}
         if obj[1]>2.2:
            # {"feature": "sepal-length", "instances": 39, "metric_value": 0.2918, "depth": 4}
            if obj[0]<=6.0:
               return 'Iris-versicolor'
            elif obj[0]>6.0:
               return 'Iris-versicolor'
            else:
               return 'Iris-versicolor'
         elif obj[1]<=2.2:
            # {"feature": "sepal-length", "instances": 3, "metric_value": 0.9183, "depth": 4}
            if obj[0]>5.0:
               return 'Iris-versicolor'
            elif obj[0]<=5.0:
               return 'Iris-versicolor'
            else:
               return 'Iris-versicolor'
         else:
            return 'Iris-versicolor'
      elif obj[3]>1.7:
         # {"feature": "sepal-length", "instances": 38, "metric_value": 0.1756, "depth": 3}
         if obj[0]>5.9:
            return 'Iris-virginica'
         elif obj[0]<=5.9:
            # {"feature": "sepal-width", "instances": 6, "metric_value": 0.65, "depth": 4}
            if obj[1]<=3.0:
               return 'Iris-virginica'
            elif obj[1]>3.0:
               return 'Iris-versicolor'
            else:
               return 'Iris-virginica'
         else:
            return 'Iris-virginica'
      else:
         return 'Iris-versicolor'
   elif obj[2]<=1.9927933150920052:
      return 'Iris-setosa'
   else:
      return 'Iris-setosa'
