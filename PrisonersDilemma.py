import numpy as np
import math
import cmath
import matplotlib.pyplot as plt

# Basis states
C = np.array([1,0])
D = np.array([0,1])

def Hermitian(matrix):
	"""
	Returns the Hermitian conjugate of a matrix.
	"""
	return matrix.conjugate().transpose()

class PrisonersDilemma():

	def __init__(self):
		self.J = None
		self.U_A = None
		self.U_B = None
		self.output = None
		# Possible inputs and outputs (as combinations of the basis states)
		self.gameStates = {
			'CC': np.kron(C, C),
			'CD': np.kron(C, D),
			'DC': np.kron(D, C),
			'DD': np.kron(D, D),
		}
		self.AlicePayoff = {
			'CC': 3,
			'CD': 0,
			'DC': 5,
			'DD': 1,
		}
		self.BobPayoff = {
			'CC': 3,
			'CD': 5,
			'DC': 0,
			'DD': 1,
		}

	# J matrix (creates entanglement)
	def setJ(self, gamma):
		"""
		Gamma is the entanglement parameter.
		0 for no entanglement (classical case), pi/2 for maximal entanglement.
		"""
		cosine = math.cos(gamma/2)
		sine = 1j*math.sin(gamma/2)
		matrix = np.zeros((4,4), dtype=complex)
		for i in range(4):
			matrix[i][i] = cosine
		matrix[0][3] = sine
		matrix[1][2] = -sine
		matrix[2][1] = -sine
		matrix[3][0] = sine

		self.J = matrix

	def playerMove(self, theta, phi):
		"""
		Possible moves that can be taken by the players.
		Returns unitary matrices that are parameterized by two variables.
		"""
		cosine = math.cos(theta/2)
		sine = math.sin(theta/2)
		matrix = np.array([[cmath.exp(1j*phi)*cosine, sine],[-sine, cmath.exp(-1j*phi)*cosine]])
		return matrix
	
	def setAliceMove(self, theta, phi):
		self.U_A = self.playerMove(theta, phi)
	
	def setBobMove(self, theta, phi):
		self.U_B = self.playerMove(theta, phi)

	def setOutput(self):
		initialState = np.matmul(self.J, self.gameStates['CC'])
		gameMoves = np.kron(self.U_A, self.U_B)
		lastStep = np.matmul(Hermitian(self.J), gameMoves)
		output = np.matmul(lastStep, initialState)
		self.output = output

	def expectedPayoff(self):
		self.setOutput()
		alice = 0
		bob = 0
		for key in self.gameStates:
			amplitude = np.dot(self.gameStates[key], self.output)
			alice += self.AlicePayoff[key] * abs(amplitude)**2
			bob += self.BobPayoff[key] * abs(amplitude)**2
		return (alice, bob)

# To cooperate - (0,0)
# To defect - (pi, 0)
pi = math.pi

game = PrisonersDilemma()

game.setJ(pi/2)
game.setAliceMove(pi/2,pi/2)

# game.setBobMove(pi,0)
# (alice, bob) = game.expectedPayoff()

# print("\nExpected payoff:")
# print("    For Alice - ", alice)
# print("    For Bob - ", bob, "\n")

X = np.linspace(0, pi, 50)
Alice = np.empty(len(X))
Bob = np.empty(len(X))

for i in range(len(X)):
	theta = X[i]
	game.setBobMove(theta, 0)
	(alice, bob) = game.expectedPayoff()
	Alice[i] = alice
	Bob[i] = bob

plt.figure()
plt.plot(X, Alice, '-', label="Alice's payoff")
plt.plot(X, Bob, '-.', label="Bob's payoff")
plt.legend()
plt.xlabel("Theta parameter for Bob")
plt.ylabel("Payoff value")
plt.title("Payoffs when Alice is playing the 'Miracle Move'")
plt.show()