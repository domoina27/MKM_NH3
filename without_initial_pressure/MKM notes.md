### Elementary steps of Ammonia dissociation

---

__Ammonia adsorption and desorption step__

* NH<sub>3</sub> (gas) + * &rarr; NH<sub>3</sub>*			(R1), k<sub>1</sub> 				
  $$
  r_1=k_1 P_{NH3}\theta
  $$

__NH<sub>3</sub> dissociation is done in 3 elementary steps__
* NH<sub>3</sub>* + * &rlarr; NH<sub>2</sub>* + H* 		  (R2),  k<sub>f2</sub>,  k<sub>b2</sub>		 
  $$
  r_2 = k_{f2}\theta_{NH3}\theta-k_{b2}\theta_{NH2}\theta_{H}
  $$

* NH<sub>2</sub>* + * &rlarr; NH* + H*             (R3), k<sub>f3</sub>,  k<sub>b3</sub>
  $$
  r_3 = k_{f3}\theta_{NH2}\theta-k_{b3}\theta_{NH}\theta_{H}
  $$

* NH* + * &rlarr; N* + H*                 (R4),  k<sub>f4</sub>,  k<sub>b4</sub>
  $$
  r_4 = k_{f4}\theta_{NH}\theta-k_{b4}\theta_{N}\theta_{H}
  $$

__N<sub>2</sub> formation is done in 3 elementary steps__
* 2 N* &rlarr; N-N******                          (R5),  k<sub>f5</sub>,  k<sub>b5</sub>
  $$
  r_5=k_{f5}\theta_N^2 - k_{b5}\theta_{N-N}
  $$

* N-N****** &rarr; **N<sub>2</sub>* + *                     (R6), k<sub>6</sub>
  $$
  r_6= k_6\theta_{N-N}
  $$

* N<sub>2</sub>* &rarr; N<sub>2</sub> (gas) + *                  (R7), k<sub>7</sub>
  $$
  r_7= k_7 \theta_{N2}
  $$

__H<sub>2</sub> formation is done in one elementary step__
* 2 H* &rarr; H<sub>2</sub> (gas) + 2*              (R8), k<sub>8</sub>
  $$
  r_8= k_8 \theta_H^2
  $$

### Rate constant equations

---

##### Surface reactions

$$
k_{sur} =\frac{k_BT}{h}\ exp(\frac{-E_a}{k_BT})
$$
##### Desorption 

$$
k_{des} = \frac{k_BT^3}{h^3} \frac{A(2\pi m k_B)}{\sigma \theta _{rot}} exp(\frac{-E_{des}}{k_B T})
$$

### Rate equations of the different species

---

$$
\frac{dP_{NH3}}{dt} = -r_1
$$

$$
\frac{d\theta_{NH3}}{dt} = r_1-r_2
$$

$$
\frac{d\theta_{NH2}}{dt} = r_2-r_3
$$

$$
\frac{d\theta_{NH}}{dt} = r_3-r_4
$$

$$
\frac{d\theta_N}{dt}= r_4 - 2r_5
$$

$$
\frac{d\theta_{N-N}}{dt}= r_5-r_6
$$

$$
\frac{d\theta_{N2}}{dt}= r_6-r_7
$$

$$
\frac{d\theta_{H}}{dt}= r_2+r_3+r_4 - 2r_8
$$

$$
\frac{d\theta}{dt}= r_6+r_7+2r_8-r_1-r_2-r_3-r_4
$$

$$
\frac{dP_{N2}}{dt}= r_5
$$

$$
\frac{dP_{H2}}{dt}= r_8
$$

$$
\frac{dT}{dt}= \beta
$$

### Steady State Approximation

---

$$
\frac{d\theta_{NH2}}{dt} =0
$$

$$
\frac{d\theta_{N-N}}{dt} =0
$$

$$
\frac{d\theta_{N2}}{dt} = 0
$$

$$
\frac{d\theta_{H}}{dt} =0
$$

Thus:
$$
r_3 = r_2
$$

$$
r_6 = r_5
$$

$$
r_7 = r_5
$$

$$
r_8 = r_2 + 1/2 r_4
$$

which gives:
$$
\frac{dP_{NH3}}{dt} = -r_1
$$

$$
\frac{d\theta_{NH3}}{dt} = r_1-r_2
$$

$$
\frac{d\theta_{NH2}}{dt} = 0
$$

$$
\frac{d\theta_{NH}}{dt} = r_3-r_4
$$

$$
\frac{d\theta_N}{dt}= r_4 - 2r_5
$$

$$
\frac{d\theta_{N-N}}{dt}= 0
$$

$$
\frac{d\theta_{N2}}{dt}= r_6-r_7
$$

$$
\frac{d\theta_{H}}{dt}= 0
$$

$$
\frac{d\theta}{dt}= 2r_5-r_1
$$

$$
\frac{dP_{N2}}{dt}= r_5
$$

$$
\frac{dP_{H2}}{dt}= r_2-1/2r_4
$$

$$
\frac{dT}{dt}= \beta
$$



