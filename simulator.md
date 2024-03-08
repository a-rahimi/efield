# Static Electric Field as a Variational Problem

We would like to recover the potential function $V:\R^3 \to \R$ in free space
subject to the constraint that $V$ has a specified value on a given set of
conductors. Typically, we would find $V$ by solving the Poisson equation $\Delta
V=0$ subject to the constraint $V(x)=V_0(x)$ whenever $x$ lies on a conductor,
and $V_0(x)$ is the specified potential of conductor. Here $\Delta$ is the
Laplace operator, which sums the second derivative of $V$ across each dimension.
This is a differential equation with boundary constraints, and the are many ways
to solve this equation. The simulator converts this to a variational problem,
which it solves using a least squares solver.

## Variational Formulation

The simulator solves the problem

$$
    \min_V \int \|\nabla V(x)\|^2 \; dx \quad \text{s.t.}\quad\forall_{x \in C}\; V(x) = V_0(x)
$$

The optimizer of the above happens to satisfy the original Lapalace equation we
seek to solve. This is known as the [Dirichlet's
principle](https://en.wikipedia.org/wiki/Dirichlet%27s_principle). The proof of
this claim is in the appendix.

This variational representation, coupled with the representation of $V$
described below, lets us avoid checking for $\Delta V=0$ throughout space.
Instead, we can enforce this condition everywhere without checking for it on a
dense grid.

## Representing the field

We'll consider fields that take the Radial Basis Function (RBF) form $$V(x) =
\sum_{i=1}^N \alpha_i \exp\left(-\frac{\|x-x_i\|^2}{2\sigma^2}\right).$$ Here,
$x_1,\ldots x_N\in \R^3$ are a set of pre-determined _anchor points_. The
parameters of the potential function are the weights $\alpha_1\ldots\alpha_N$.  We'll be
searching for values of these that solve the Dirichlet formulation above.

## Representing the objective function

The objective function above relies on the gradient of the objective function
(which is the corresponding electric field). It is
$$\begin{align}
\nabla V(x) &= \sum_i \alpha_i \nabla \exp\left(-\frac{\|x-x_i\|^2}{2\sigma^2}\right) \\
&= -\sum_i \alpha_i \frac{x-x_i}{\sigma^2}\exp\left(-\frac{\|x-x_i\|^2}{2\sigma^2}\right) 
\end{align}$$

Its squared norm is therefore

$$
\|\nabla V(x)\|^2 = \frac{1}{\sigma^4} \sum_{ij} \alpha_i \alpha_j (x-x_i)^\top(x-x_i)\exp\left(-\frac{\|x-x_i\|^2+\|x-x_j\|^2}{2\sigma^2}\right) 
$$

Completing the square for the quantity in the exponent yields

$$\begin{align}
\|x-x_i\|^2+\|x-x_j\|^2 &= 2 \|x\|^2-2(x_i+x_j)^\top x+\|x_i\|^2 + \|x_j\|^2 \\
&= 2 \left\|x-\frac{x_i+x_j}{2}\right\|^2-2\left\|\frac{x_i+x_j}{2}\right\|^2+\|x_i\|^2 + \|x_j\|^2 \\
&= 2 \left\|x-\frac{x_i+x_j}{2}\right\|^2 + \frac{1}{2}\|x_i-x_j\|^2.
\end{align}$$

Plugging back gives

$$
\|\nabla V(x)\|^2 = \frac{1}{\sigma^4} \sum_{ij} \alpha_i \alpha_j (x-x_i)^\top (x-x_i)\exp\left(-\frac{ \left\|x-\frac{x_i+x_j}{2}\right\|^2} {\sigma^2}\right) \exp\left(-\frac{\|x_i-x_j\|^2}{4 \sigma^2}\right)
$$

The objective is the indefinite integral of this quantity. This is a Gaussian integral and it can be computed in closed form:

$$\begin{align}
\int \|\nabla V(x)\|^2 \; dx &= \frac{1}{\sigma^4} \sum_{ij} \alpha_i \alpha_j\exp\left(-\frac{\left\|x_i-x_j\right\|^2}{4\sigma^2}\right) \int (x-x_i)^\top(x-x_i)\exp\left(-\frac{ \left\|x-\frac{x_i+x_j}{2}\right\|^2} {\sigma^2}\right) \; dx \\
&=
 \frac{\pi^{3/2}}{2\sigma} \sum_{ij} \alpha_i \alpha_j\exp\left(-\frac{\left\|x_i-x_j\right\|^2}{4\sigma^2}\right) \left(\sigma^2 - \|x_1\|^2 - \|x_2\|^2\right).
\end{align}$$

The second equality follows because
the integral is the expected value of $(x-x_i)^\top(x-x_j)$ when $x$ is drawn from a Gaussian distribution with mean $(x_i+x_j)/2$ and variance $\sigma^2/2$. This is derived step-by-step in the appendix. 

# Solving for the parameters of the field

We can restate the variational search over the potential function as an
optimization problem over the coefficients $a_1,\ldots,a_N$. Define the $N\times N$ matrix M whose $ij$th entry is
$$
M_{ij} \equiv \frac{\pi^{3/2}}{2\sigma} \exp\left(-\frac{\left\|x_i-x_j\right\|^2}{4\sigma^2}\right) \left(\sigma^2 - \|x_1\|^2 - \|x_2\|^2\right),
$$
and a matrix $K$ whose $ui$th entry is

$$K_{ui} = \exp\left(-\frac{\|x_u-x_i\|^2}{2\sigma^2}\right),$$

where each $x_u$ is a point on one of the conductors. Then 
search over the parameters of the potential function becomes
$$\begin{align}
\min_{\alpha \R^n} \; \alpha^\top M \alpha \quad \text{s.t.}\quad K \alpha = v_0,
\end{align}$$

where entries of the vector $v_0$ correspond to the given potentials on each of
the conductors. This problem admits a closed form solution, which is derived in
the appendix:

$$\begin{align}
\alpha^* &= V(KV)^{-1} v_0 \\
v_0 &= M^{-1} K^\top
\end{align}$$

# Appendix: Proof of the Dirichlet Principle

# Appendix: The expected value of a Quadratic form under a Gaussian

Let $\mu=(x_1+x_2)/2$. Then

$$\begin{align}
\E_{x \sim \mathcal{N}(\mu, \sigma^2)} (x-x_1)^\top (x-x_2) &= \E x^2 - (x_1+x_2)^\top\E x + x_1^\top x_2 \\
&= \sigma^2 + \mu^2 - (x_1+x_2)^\top\mu + x_1^\top x_2 \\
&= \sigma^2  - \|x_1+x_2\|^2/2 + x_1^\top x_2 \\
&= \sigma^2 - \|x_1\|^2/2 - \|x_2\|^2/2.
\end{align}$$

# Appendix: Closed form solution to linearly constrained quadratic minimization

We want to minimize $\tfrac{1}{2}x^\top Mx$ subject to $Ax = b$. The Lagrangian for this problem is $\tfrac{1}{2}x^\top Mx + \lambda^\top (Ax-b)$. At a saddle point of the Lagrangian, we have $0=Mx + A^\top\lambda$, which means $x = M^{-1} A^\top \lambda$. Plugging this back into the constraint gives $A M^{-1}A^\top \lambda = b$, so that $\lambda = \left[A M^{-1}A^\top\right]^{-1} b$, and therefore $x=M^{-1}A^\top \left[A M^{-1}A^\top\right]^{-1} b$.