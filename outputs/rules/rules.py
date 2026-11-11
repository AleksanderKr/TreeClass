def findDecision(obj): #obj[0]: cap-shape, obj[1]: cap-surface, obj[2]: cap-color, obj[3]: bruises, obj[4]: odor, obj[5]: gill-attachment, obj[6]: gill-spacing, obj[7]: gill-size, obj[8]: gill-color, obj[9]: stalk-shape, obj[10]: stalk-root, obj[11]: stalk-surface-above-ring, obj[12]: stalk-surface-below-ring, obj[13]: stalk-color-above-ring, obj[14]: stalk-color-below-ring, obj[15]: veil-type, obj[16]: veil-color, obj[17]: ring-number, obj[18]: ring-type, obj[19]: spore-print-color, obj[20]: population, obj[21]: habitat
   # {"feature": "odor", "instances": 7311, "metric_value": 0.9991, "depth": 1}
   if obj[4] == 'n':
      # {"feature": "spore-print-color", "instances": 3173, "metric_value": 0.2112, "depth": 2}
      if obj[19] == 'n':
         return 'e'
      elif obj[19] == 'k':
         return 'e'
      elif obj[19] == 'w':
         # {"feature": "veil-color", "instances": 560, "metric_value": 0.3971, "depth": 3}
         if obj[16] == 'w':
            # {"feature": "gill-size", "instances": 553, "metric_value": 0.3543, "depth": 4}
            if obj[7] == 'b':
               return 'e'
            elif obj[7] == 'n':
               # {"feature": "gill-spacing", "instances": 80, "metric_value": 0.9959, "depth": 5}
               if obj[6] == 'w':
                  return 'e'
               elif obj[6] == 'c':
                  return 'p'
               else:
                  return 'e'
            else:
               return 'e'
         elif obj[16] == 'y':
            return 'p'
         else:
            return 'e'
      elif obj[19] == 'r':
         return 'p'
      elif obj[19] == 'o':
         return 'e'
      elif obj[19] == 'y':
         return 'e'
      elif obj[19] == 'h':
         return 'e'
      elif obj[19] == 'b':
         return 'e'
      else:
         return 'e'
   elif obj[4] == 'f':
      return 'p'
   elif obj[4] == 'y':
      return 'p'
   elif obj[4] == 's':
      return 'p'
   elif obj[4] == 'a':
      return 'e'
   elif obj[4] == 'l':
      return 'e'
   elif obj[4] == 'p':
      return 'p'
   elif obj[4] == 'c':
      return 'p'
   elif obj[4] == 'm':
      return 'p'
   else:
      return 'e'
