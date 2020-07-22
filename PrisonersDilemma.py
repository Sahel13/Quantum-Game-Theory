import numpy as np
import math
import cmath

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
		self.Output = None

		# Basis states
		self.C = np.array([1,0])
		self.D = np.array([0,1])

		# Possible inputs and outputs (as combinations of the basis states)
		self.GameStates = {
			'CC': np.kron(self.C, self.C),
			'CD': np.kron(self.C, self.D),
			'DC': np.kron(self.D, self.C),
			'DD': np.kron(self.D, self.D),
		}

		# The payoff matrix
		self.PayoffMatrix = {
			'CC': [3,3],
			'CD': [0,5],
			'DC': [5,0],
			'DD': [1,1],
		}

	def playerMove(self, theta, phi):
		"""
		Possible moves that can be taken by the players.
		Returns unitary matrices that are parameterized by two variables.
		"""
		cosine = math.cos(theta/2)
		sine = math.sin(theta/2)
		matrix = np.array([[cmath.exp(1j*phi)*cosine, sine],[-sine, cmath.exp(-1j*phi)*cosine]])
		return matrix

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
		matrix[0][3] = -sine
		matrix[1][2] = sine
		matrix[2][1] = sine
		matrix[3][0] = -sine

		self.J = matrix
	
	def setAliceMove(self, theta, phi):
		self.U_A = self.playerMove(theta, phi)
	
	def setBobMove(self, theta, phi):
		self.U_B = self.playerMove(theta, phi)

	def setOutput(self):
		initialState = np.matmul(self.J, self.GameStates['CC'])
		gameMoves = np.kron(self.U_A, self.U_B)
		lastStep = np.matmul(Hermitian(self.J), gameMoves)
		output = np.matmul(lastStep, initialState)
		self.Output = output

	def expectedPayoff(self):
		self.setOutput()
		alice = 0
		bob = 0
		for key in self.GameStates:
			amplitude = np.dot(self.GameStates[key], self.Output)
			alice += self.PayoffMatrix[key][0] * abs(amplitude)**2
			bob += self.PayoffMatrix[key][1] * abs(amplitude)**2
		return (alice, bob)
