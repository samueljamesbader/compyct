# Trivial Xtor Example

## Intrinsic equations:

Charge control:

$$qn_s= n V_{th} C \log\left(1+\exp\left[\frac{V-V_T}{nV_{th}}\right]\right)$$

Charge division for capacitances:

##$$Q_S=Q_D=Q_G/2 = qn_s/2$$
$$Q_S = Q_G = qn_s/2$$
$$Q_D = 0$$

Velocity saturation:

$$v_\mu= \frac{ \mu_0 V_{DS} }{ L }$$

$$v= \frac{v_\mu}{\left(1+\left(\frac{v_\mu}{v_s}\right)^\beta\right)^{\frac{1}{\beta}}}$$

Current:

$$I_D=w \times qn_s \times v$$

Trapping:

Modelled by an R-C network
```
           rtrap
V(d,g) - /\/\/\/\ -  Vtrap
                   |
                _______
                _______  ctrap
                   |
                  ---
                   -
```
where $V_{trap}$ shifts the $V_T$ like so:
$$V_T=V_{T0}+gtrap*V_{trap}$$