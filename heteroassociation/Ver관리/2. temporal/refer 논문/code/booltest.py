import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaSynapseSNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(AlphaSynapseSNN, self).__init__()
        # 시냅스 가중치 정의
        self.w1 = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.5)
        self.w2 = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.5)
        
        # 뉴런 및 시냅스 파라미터
        self.v_threshold = 1.0  # 발화 임계값
        self.tau = 1.0          # 시간 상수

    def alpha_kernel(self, t):
        """ 알파 시냅스 함수 형태 구현: h(t) = t * exp(-t / tau) """
        # t는 입력 스파이크 시간과 현재 시간의 차이 (>= 0)
        return t * torch.exp(-t / self.tau)

    def forward(self, spike_times, time_windows):
        """
        spike_times: [batch_size, input_dim] - 각 입력 뉴런의 스파이크 발생 시간
        time_windows: 평가할 시간 축 (예: 0부터 T까지의 타임스텝)
        """
        batch_size = spike_times.size(0)
        num_steps = time_windows.size(0)
        
        # 1. 입력 스파이크를 알파 시냅스 전류로 변환
        # [batch_size, num_steps, input_dim] 형태로 시간 흐름에 따른 포텐셜 계산
        # t - t_spike 형태의 시차 계산
        t_expanded = time_windows.view(1, num_steps, 1) # [1, T, 1]
        s_expanded = spike_times.view(batch_size, 1, -1) # [B, 1, D_in]
        
        delta_t = t_expanded - s_expanded
        delta_t = torch.clamp(delta_t, min=0.0) # 과거 시점의 스파이크만 영향
        
        # 알파 시냅스 포텐셜 적용
        psp = self.alpha_kernel(delta_t) # [B, T, D_in]
        
        # 2. 은닉층 가중치 적용 및 막 전위 계산
        # hidden_pot = sum(psp * w1)
        hidden_current = torch.matmul(psp, self.w1) # [B, T, hidden_dim]
        
        # 간단한 적분 형태의 막 전위 누적 (Leaky Integrate)
        hidden_potential = torch.cumsum(hidden_current, dim=1)
        
        # 3. 최초 스파이크 시간(TTFS) 또는 출력 발화 여부 결정
        # 임계값을 가장 먼저 넘는 시점을 찾는 연산 (Surrogate Gradient나 근사 미분 적용 구간)
        output_current = torch.matmul(hidden_potential, self.w2) # [B, T, output_dim]
        output_potential = torch.cumsum(output_current, dim=1)
        
        return output_potential