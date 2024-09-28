import torch
from utilities.interplation import cubicBezier, cubicBezierFirstDerivative, cubicBezierSecondDerivative

class TrajectoryPlanner:
    def __init__(self, num_envs, action_dim, device):
        self._num_envs = num_envs
        self._action_dim = action_dim
        self._device = device
        self._p0 = torch.zeros((self._num_envs, self._action_dim), device=self._device)
        self._pf = torch.zeros((self._num_envs, self._action_dim), device=self._device)
        self._duration = torch.zeros(self._num_envs, device=self._device)
        self._p = torch.zeros((self._num_envs, self._action_dim), device=self._device)
        self._v = torch.zeros((self._num_envs, self._action_dim), device=self._device)
        self._a = torch.zeros((self._num_envs, self._action_dim), device=self._device)
        
    def setInitialPosition(self, idx:torch.Tensor, p0_idx:torch.Tensor):
        self._p0[idx] = p0_idx.clone()
        self._p[idx] = p0_idx.clone()
        self._v[idx] = torch.zeros_like(p0_idx)
        self._a[idx] = torch.zeros_like(p0_idx)
        self._pf[idx] = p0_idx.clone()
        
    def setFinalPosition(self, idx:torch.Tensor, pf_idx:torch.Tensor):
        self._pf[idx] = pf_idx.clone()

    def setDuration(self, idx:torch.Tensor, duration_idx:torch.Tensor):
        self._duration[idx] = duration_idx.clone()

    def getPosition(self):
        return self._p
    
    def getVelocity(self):
        return self._v
    
    def getAcceleration(self):
        return self._a
    
    def update(self, phase:torch.Tensor):
        self.computeTrajectoryBezier(phase)

    def computeTrajectoryBezier(self, phase:torch.Tensor):
        self._p = cubicBezier(self._p0, self._pf, phase)
        self._v = cubicBezierFirstDerivative(self._p0, self._pf, phase) / self._duration[:, None]
        self._a = cubicBezierSecondDerivative(self._p0, self._pf, phase) / (self._duration[:, None] * self._duration[:, None])

    def computeTrajectoryLinear(self, phase:torch.Tensor):
        self._p = self._p0 + (self._pf - self._p0) * phase
        self._v = (self._pf - self._p0) / self._duration
        self._a = torch.zeros((self._num_envs, self._action_dim), device=self._device)
