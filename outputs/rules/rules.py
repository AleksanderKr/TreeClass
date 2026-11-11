def findDecision(obj): #obj[0]: Outlook, obj[1]: Temperature, obj[2]: Humidity, obj[3]: Wind
   # {"feature": "Temperature", "instances": 14, "metric_value": 0.9403, "depth": 1}
   if obj[1]<=83:
      # {"feature": "Outlook", "instances": 13, "metric_value": 0.8905, "depth": 2}
      if obj[0] == 'Rain':
         # {"feature": "Wind", "instances": 5, "metric_value": 0.971, "depth": 3}
         if obj[3] == 'Weak':
            return 'Yes'
         elif obj[3] == 'Strong':
            return 'No'
         else:
            return 'Yes'
      elif obj[0] == 'Sunny':
         # {"feature": "Humidity", "instances": 4, "metric_value": 1.0, "depth": 3}
         if obj[2]>70:
            return 'No'
         elif obj[2]<=70:
            return 'Yes'
         else:
            return 'No'
      elif obj[0] == 'Overcast':
         return 'Yes'
      else:
         return 'Yes'
   elif obj[1]>83:
      return 'No'
   else:
      return 'Yes'
