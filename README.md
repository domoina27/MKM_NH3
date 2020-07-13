### MICROKINETIC MODEL OF NH<sub>3</sub> DISSOCIATION AND N<sub>2</sub> AND H<sub>2</sub> FROMATION
---
### Reactions:
#### Ammonia adsorption and desorption step:
* NH<sub>3</sub> (gas) + * = NH<sub>3</sub>*

#### NH<sub>3</sub> dissociation is done in 3 elementarty steps
* NH<sub>3</sub>* + * = NH<sub>2</sub>* + H*
* NH<sub>2</sub>* + * = NH* + H*
* NH* + * = N* + H*

#### N<sub>2</sub> formation is done in 3 elementarty steps
* 2 N* = NN**
* NN*** = N<sub>2</sub>* + *
* N2* = N<sub>2</sub> (gas) + *

#### H<sub>2</sub> formation is done in one elementary step
* 2 H* = H<sub>2</sub> (gas) + 2*

### Initial conditions
In this model, the initial temperature T<sub>0</sub> start at 50K, and a heating rate Beta	 = 3 K/s. 

To simplify our current model, steady-state approximation of the following adsorbed species was used: NH<sub>2</sub><sub>ads</sub>, N-N<sub>ads</sub>, N<sub>2</sub><sub>ads</sub>, and H<sub>ads</sub>

### Solve the ODE
Because our model is stiff, the ODEs cannot be solved using an explicit method. Consequently, **BDF method** (Backward-Differentiation Formulas) was used instead of Runge Kutta. The former however requires a **Jacobian matrix** of the right-hand side of the system with respect to y. 
Thus, a 12 x 12 Jacobian matrix has to be define prior to solving the ODEs.
