import numpy as np
import skfuzzy as fuzz
import os
from skfuzzy import control as ctrl
price = ctrl.Antecedent(np.arange(0, 11, 1), 'price')
performance = ctrl.Antecedent(np.arange(0,101,1),'performance')
decision = ctrl.Consequent(np.arange(0, 11, 1), 'decision')
# Fuzzy
price['high'] = fuzz.trimf(price.universe, [7, 10, 10])
performance['bad'] = fuzz.trimf(performance.universe, [0, 0, 40])
decision['refuse'] = fuzz.trimf(decision.universe, [0,0,10])
decision['accept'] = fuzz.trimf(decision.universe, [0,10,10])
price.view()
performance.view()
decision.view()
# Define a rule
rule = ctrl.Rule(price['high']|performance['bad'], decision['refuse'])
control = ctrl.ControlSystem([rule])
control_simulation = ctrl.ControlSystemSimulation(control)
control_simulation.input['price'] = 8
control_simulation.input['performance'] = 20
control_simulation.compute()
decision.view(sim=control_simulation)
