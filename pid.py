
class PID:
    def __init__(self, kp, ki, kd, set_point):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.set_point = set_point  # Desired set point
        self.integral = 0
        self.last_error = 0

    def compute(self, current_value):
        error = self.set_point - current_value
        self.integral += error
        derivative = error - self.last_error

        # PID formula
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        # Update last error
        self.last_error = error

        # Return valve setting value as a percentage of valve opening
        return output