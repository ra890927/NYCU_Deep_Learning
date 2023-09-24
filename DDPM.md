$$
\begin{aligned}
L =& \mathbb{E}_q[- \log \frac{p_{\theta}(x_{0\ :\ T})}{q(x_{1\ :\ T}\ |\ x_0)}] \\
=& \mathbb{E}_q[- \log \frac{p(x_T) \cdot \prod_{t=1}^{T} p_{\theta}(x_{t-1}\ |\ x_t)}{\prod_{t=1}^{T} q(x_t\ |\ x_{t - 1},\ x_0)}] \\
=& \mathbb{E}_q[ - \log \frac{p(x_1) \cdot p_{\theta}(x_0\ |\ x_1) \cdot \prod_{t=2}^T p_{\theta}(x_{t-1}\ |\ x_t)}{q(x_1\ |\ x_0) \cdot \prod_{t=2}^T q(x_t\ |\ x_{t-1},\ x_0)} ] \\
=& \mathbb{E}_q[- \log \frac{p(x_T) \cdot p_{\theta}(x_0\ |\ x_1)}{q(x_1\ |\ x_0)} - \log \prod_{t=2}^T \frac{p_{\theta}(x_{t - 1}\ |\ x_t)}{q(x_t\ |\ x_{t-1},\ x_0)}] \\
=& \mathbb{E}_q[- \log \frac{p(x_T) \cdot p_{\theta}(x_0\ |\ x_1)}{q(x_1\ |\ x_0)} - \log \prod_{t=2}^T \frac{p_{\theta}(x_{t - 1}\ |\ x_t)}{\frac{q(x_{t-1}\ |\ x_{t},\ x_0)\ q(x_t\ |\ x_0)}{q(x_{t-1}\ |\ x_0)}} \\
=& \mathbb{E}_q[- \log \frac{p(x_T) \cdot p_{\theta}(x_0\ |\ x_1)}{q(x_1\ |\ x_0)} - \log \prod_{t=2}^T \frac{q(x_{t-1}\ |\ x_0)}{q(x_t\ |\ x_0)} - \log \prod_{t=2}^T \frac{p_{\theta}(x_{t - 1}\ |\ x_t)}{q(x_{t-1}\ |\ x_t,\ x_0)} ] \\
=& \mathbb{E}_q[- \log \frac{p(x_T) \cdot p_{\theta}(x_0\ |\ x_1)}{q(x_1\ |\ x_0)} - \log \frac{q(x_1\ |\ x_0) q(x_2\ |\ x_0) \cdots q(x_{T-1}\ |\ x_0)}{q(x_2\ |\ x_0)q(x_3\ |\ x_0) \cdots q(x_T\ |\ x_0)} - \log \prod_{t=2}^T \frac{p_{\theta}(x_{t - 1}\ |\ x_t)}{q(x_{t-1}\ |\ x_t,\ x_0)} ] \\
=& \mathbb{E}_q[- \log \frac{p(x_T) \cdot p_{\theta}(x_0\ |\ x_1)}{q(x_1\ |\ x_0)} - \log \frac{q(x_1\ |\ x_0)}{q(x_T\ |\ x_0)} - \log \prod_{t=2}^T \frac{p_{\theta}(x_{t - 1}\ |\ x_t)}{q(x_{t-1}\ |\ x_t,\ x_0)} ] \\
=& \mathbb{E}_q[- \log \frac{p(x_T) \cdot p_{\theta}(x_0\ |\ x_1)}{q(x_T\ |\ x_0)} - \log \prod_{t=2}^T \frac{p_{\theta}(x_{t - 1}\ |\ x_t)}{q(x_{t-1}\ |\ x_t,\ x_0)} ] \\
=& \mathbb{E}_q[ - \log p_{\theta}(x_0\ |\ x_1) - \log \frac{p(x_T)}{q(x_T\ |\ x_0)} - \log \prod_{t=2}^T \frac{p_{\theta}(x_{t - 1}\ |\ x_t)}{q(x_{t-1}\ |\ x_t,\ x_0)} ] \\
=& \mathbb{E}_q[ - \log p_{\theta}(x_0\ |\ x_1) + D_{KL}(q(x_T\ |\ x_0)\ ||\ p(x_T)) + \sum_{t=2}^T D_{KL}(q(x_{t-1}\ |\ x_t,\ x_0)\ ||\ p_{\theta}(x_{t-1}\ |\ x_t))] \\
=& \mathbb{E}_q[ - \log p_{\theta}(x_0\ |\ x_1)  + \sum_{t=2}^T D_{KL}(q(x_{t-1}\ |\ x_t,\ x_0)\ ||\ p_{\theta}(x_{t-1}\ |\ x_t))] + D_{KL}(q(x_T\ |\ x_0)\ ||\ p(x_T)) \\
\end{aligned}
$$

