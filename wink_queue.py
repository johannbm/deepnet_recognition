
class WinkQueue:

	def __init__(self):
		self.size = 5
		self.queue = [0] * 5
		self.current_index = 0
		self.trigger_limit = 3
		self.current_state = -1

	def add(self, classification):
		self.queue[self.current_index] = classification
		self.current_index = (self.current_index + 1) % self.size

		self.check_trigger()

	def check_trigger(self):
		count_winks = sum(1 for x in self.queue if x==1)
		if count_winks >= self.trigger_limit and self.current_state != 1:
			self.current_state = 1
			print "Wink detected"

		count_not_winks = self.size - count_winks
		if count_not_winks >= self.trigger_limit and self.current_state != -1:
			self.current_state = -1

	def reset(self):
		self.queue = [0] * self.size
		self.current_index = 0
