import matplotlib.pyplot as plt 
from truncnormkde import *
import jax.numpy as jnp

def test_plot_works():
	# Generate 8000 from truncated normal
	N = 8000
	mu = 0.2; sigma=0.2
	Xb = np.random.randn(N*10,2)*sigma + mu # N(0.2, 0.5)
	#Xb = np.random.rand(N,2)
	Xb = Xb[ np.all( (Xb > 0) & (Xb < 1), axis=1) , :]
	X = Xb[0:N,:]

	# Bounds
	a = jnp.array([0,0])
	b = jnp.array([1,1])
	bandwidth = compute_bandwidth(X) # Uses the scotts rule for computation

	# Generate the evaluate grid
	x,y = jnp.linspace(0,1,100), jnp.linspace(0,1,100)
	x_2d, y_2d = jnp.meshgrid(x,y)
	X_grid = jnp.stack([x_2d, y_2d],axis=-1)

	# Define the object
	KDE = BoundedKDE(a=a, b=b, bandwidth=bandwidth)

	# Evaluate the KDE on the grid
	computed_values = KDE(X_grid, X)

	fig, ax = plt.subplots(1)
	ax.contourf(x_2d, y_2d, computed_values, vmin=0.0)

	ax.scatter([mu], [mu], c='r', marker='x')
	ax.set_xlim(a[0],b[0])
	ax.set_ylim(a[1],b[1])

	ax.set_aspect('equal')
	#plt.show()
	plt.savefig("./tests/mu0p2-sig0p2-test.png")


def test_kernel_gram_works():
	# Generate 8000 from truncated normal
	N = 8000
	mu = 0.2; sigma=0.2
	Xb = np.random.randn(N*10,2)*sigma + mu # N(0.2, 0.5)
	#Xb = np.random.rand(N,2)
	Xb = Xb[ np.all( (Xb > 0) & (Xb < 1), axis=1) , :]
	X = Xb[0:N,:]

	# Bounds
	a = jnp.array([0,0])
	b = jnp.array([1,1])
	bandwidth = compute_bandwidth(X) # Uses the scotts rule for computation

	# Define the object
	KDE = BoundedKDE(a=a, b=b, bandwidth=bandwidth)

	# Evaluate the Kernel gram
	computed_values = KDE.gram_matrix(X)

	print(computed_values.shape)
	print(jnp.any(jnp.isnan(computed_values)))
