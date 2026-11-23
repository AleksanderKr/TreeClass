def findDecision(obj): #obj[0]: handicapped-infants, obj[1]: water-project-cost-sharing, obj[2]: adoption-of-the-budget-resolution, obj[3]: physician-fee-freeze, obj[4]: el-salvador-aid, obj[5]: religious-groups-in-schools, obj[6]: anti-satellite-test-ban, obj[7]: aid-to-nicaraguan-contras, obj[8]: mx-missile, obj[9]: immigration, obj[10]: synfuels-corporation-cutback, obj[11]: education-spending, obj[12]: superfund-right-to-sue, obj[13]: crime, obj[14]: duty-free-exports, obj[15]: export-administration-act-south-africa
   # {"feature": "physician-fee-freeze", "instances": 393, "metric_value": 0.9627, "depth": 1}
   if obj[3] == 'n':
      # {"feature": "adoption-of-the-budget-resolution", "instances": 222, "metric_value": 0.0741, "depth": 2}
      if obj[2] == 'y':
         return 'democrat'
      elif obj[2] == 'n':
         # {"feature": "education-spending", "instances": 22, "metric_value": 0.4395, "depth": 3}
         if obj[11] == 'n':
            # {"feature": "synfuels-corporation-cutback", "instances": 14, "metric_value": 0.3712, "depth": 4}
            if obj[10] == 'y':
               return 'democrat'
            elif obj[10] == 'n':
               # {"feature": "religious-groups-in-schools", "instances": 5, "metric_value": 0.7219, "depth": 5}
               if obj[5] == 'y':
                  return 'democrat'
               elif obj[5] == 'n':
                  # {"feature": "crime", "instances": 2, "metric_value": 1.0, "depth": 6}
                  if obj[13] == 'y':
                     return 'republican'
                  elif obj[13] == 'n':
                     return 'democrat'
                  else:
                     return 'republican'
               else:
                  return 'democrat'
            else:
               return 'democrat'
         elif obj[11] == 'y':
            return 'democrat'
         elif obj[11] == '?':
            return 'republican'
         else:
            return 'democrat'
      elif obj[2] == '?':
         return 'democrat'
      else:
         return 'democrat'
   elif obj[3] == 'y':
      # {"feature": "synfuels-corporation-cutback", "instances": 161, "metric_value": 0.4048, "depth": 2}
      if obj[10] == 'n':
         # {"feature": "duty-free-exports", "instances": 125, "metric_value": 0.1633, "depth": 3}
         if obj[14] == 'n':
            # {"feature": "adoption-of-the-budget-resolution", "instances": 105, "metric_value": 0.0776, "depth": 4}
            if obj[2] == 'n':
               return 'republican'
            elif obj[2] == 'y':
               # {"feature": "export-administration-act-south-africa", "instances": 12, "metric_value": 0.4138, "depth": 5}
               if obj[15] == 'y':
                  return 'republican'
               elif obj[15] == '?':
                  # {"feature": "handicapped-infants", "instances": 2, "metric_value": 1.0, "depth": 6}
                  if obj[0] == 'y':
                     return 'republican'
                  elif obj[0] == 'n':
                     return 'democrat'
                  else:
                     return 'republican'
               else:
                  return 'republican'
            else:
               return 'republican'
         elif obj[14] == 'y':
            # {"feature": "immigration", "instances": 13, "metric_value": 0.6194, "depth": 4}
            if obj[9] == 'y':
               return 'republican'
            elif obj[9] == 'n':
               # {"feature": "export-administration-act-south-africa", "instances": 4, "metric_value": 1.0, "depth": 5}
               if obj[15] == '?':
                  # {"feature": "water-project-cost-sharing", "instances": 2, "metric_value": 1.0, "depth": 6}
                  if obj[1] == 'n':
                     return 'democrat'
                  elif obj[1] == 'y':
                     return 'republican'
                  else:
                     return 'democrat'
               elif obj[15] == 'y':
                  return 'democrat'
               elif obj[15] == 'n':
                  return 'republican'
               else:
                  return 'democrat'
            else:
               return 'republican'
         elif obj[14] == '?':
            return 'republican'
         else:
            return 'republican'
      elif obj[10] == 'y':
         # {"feature": "export-administration-act-south-africa", "instances": 30, "metric_value": 0.9183, "depth": 3}
         if obj[15] == 'y':
            # {"feature": "mx-missile", "instances": 15, "metric_value": 0.5665, "depth": 4}
            if obj[8] == 'n':
               return 'republican'
            elif obj[8] == 'y':
               # {"feature": "handicapped-infants", "instances": 3, "metric_value": 0.9183, "depth": 5}
               if obj[0] == 'y':
                  # {"feature": "adoption-of-the-budget-resolution", "instances": 2, "metric_value": 1.0, "depth": 6}
                  if obj[2] == 'y':
                     return 'republican'
                  elif obj[2] == 'n':
                     return 'democrat'
                  else:
                     return 'republican'
               elif obj[0] == 'n':
                  return 'democrat'
               else:
                  return 'democrat'
            else:
               return 'republican'
         elif obj[15] == 'n':
            # {"feature": "adoption-of-the-budget-resolution", "instances": 10, "metric_value": 1.0, "depth": 4}
            if obj[2] == 'n':
               # {"feature": "superfund-right-to-sue", "instances": 7, "metric_value": 0.8631, "depth": 5}
               if obj[12] == 'y':
                  # {"feature": "water-project-cost-sharing", "instances": 6, "metric_value": 0.65, "depth": 6}
                  if obj[1] == 'y':
                     return 'republican'
                  elif obj[1] == 'n':
                     # {"feature": "handicapped-infants", "instances": 2, "metric_value": 1.0, "depth": 7}
                     if obj[0] == 'y':
                        return 'republican'
                     elif obj[0] == 'n':
                        return 'democrat'
                     else:
                        return 'republican'
                  else:
                     return 'republican'
               elif obj[12] == 'n':
                  return 'democrat'
               else:
                  return 'republican'
            elif obj[2] == 'y':
               return 'democrat'
            elif obj[2] == '?':
               return 'democrat'
            else:
               return 'republican'
         elif obj[15] == '?':
            # {"feature": "education-spending", "instances": 5, "metric_value": 0.971, "depth": 4}
            if obj[11] == 'y':
               # {"feature": "water-project-cost-sharing", "instances": 2, "metric_value": 1.0, "depth": 5}
               if obj[1] == 'y':
                  return 'democrat'
               elif obj[1] == 'n':
                  return 'republican'
               else:
                  return 'democrat'
            elif obj[11] == 'n':
               return 'democrat'
            elif obj[11] == '?':
               return 'republican'
            else:
               return 'democrat'
         else:
            return 'republican'
      elif obj[10] == '?':
         return 'republican'
      else:
         return 'republican'
   elif obj[3] == '?':
      # {"feature": "education-spending", "instances": 10, "metric_value": 0.7219, "depth": 2}
      if obj[11] == 'n':
         return 'democrat'
      elif obj[11] == '?':
         # {"feature": "handicapped-infants", "instances": 3, "metric_value": 0.9183, "depth": 3}
         if obj[0] == 'n':
            return 'republican'
         elif obj[0] == '?':
            return 'democrat'
         elif obj[0] == 'y':
            return 'democrat'
         else:
            return 'democrat'
      elif obj[11] == 'y':
         return 'republican'
      else:
         return 'democrat'
   else:
      return 'democrat'
