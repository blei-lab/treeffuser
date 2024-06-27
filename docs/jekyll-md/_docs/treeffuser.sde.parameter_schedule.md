# treeffuser.sde.parameter_schedule module

### *class* treeffuser.sde.parameter_schedule.ExponentialSchedule(min_value: float, max_value: float)

Bases: [`ParameterSchedule`](#treeffuser.sde.parameter_schedule.ParameterSchedule)

Exponential schedule for a parameter, between min_value and max_value.
The value of the parameter at time t is given by:
min_value \* (max_value / min_value) \*\* t

* **Parameters:**
  * **min_value** (*float*) – Minimum value of the parameter (at time 0).
  * **max_value** (*float*) – Maximum value of the parameter (at time 1).

#### get_derivative(t: ndarray | float)

Get the derivative of the parameter at time t.

#### get_integral(t: ndarray | float)

Get the integral of the parameter at time t from time 0.

#### get_value(t: ndarray | float)

Get the value of the parameter at time t.

### *class* treeffuser.sde.parameter_schedule.LinearSchedule(min_value: float, max_value: float)

Bases: [`ParameterSchedule`](#treeffuser.sde.parameter_schedule.ParameterSchedule)

Linear schedule for a parameter, between min_value and max_value.
The value of the parameter at time t is given by:
min_value + (max_value - min_value) \* t

* **Parameters:**
  * **min_value** (*float*) – Minimum value of the parameter (at time 0).
  * **max_value** (*float*) – Maximum value of the parameter (at time 1).

#### get_derivative(t: ndarray | float)

Get the derivative of the parameter at time t.

#### get_integral(t: ndarray | float)

Get the integral of the parameter at time t from time 0.

#### get_value(t: ndarray | float)

Get the value of the parameter at time t.

### *class* treeffuser.sde.parameter_schedule.ParameterSchedule

Bases: `ABC`

Base class representing a parameter as a function of time.

#### get_derivative(t: ndarray | float)

Get the derivative of the parameter at time t.

#### get_integral(t: ndarray | float)

Get the integral of the parameter at time t from time 0.

#### *abstract* get_value(t: ndarray | float)

Get the value of the parameter at time t.
