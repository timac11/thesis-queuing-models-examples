class AvgMetric:
    def __init__(self, name):
        self.count = 0
        self.value = 0
        self.name = name

    def collect(self, value):
        self.value = float((self.value * self.count) + value) / (self.count + 1)
        self.count = self.count + 1

    def get_value(self):
        return [self.value, self.count]

    def __repr__(self):
        return f"""
            Metric {self.name} 
            Count: {self.count}, Value: {self.value}
        """


class CountMetric:
    def __init__(self, name):
        self.value = 0
        self.name = name

    def collect(self):
        self.value += 1

    def get_value(self):
        return self.value

    def __repr__(self):
        return f"""
            Metric {self.name} 
            Value: {self.value}
        """
