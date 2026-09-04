# Lab02 - Markov Decision Process và Dynamic Programming

## 1. Thông tin sinh viên

- Họ tên: Tran The Anh
- MSSV: 24100707
- Lớp: EEE.AI-24106.1
- Học phần: Reinforcement Learning
- GitHub username: [Điền GitHub username]

## 2. Mục tiêu

Lab02 tập trung vào Markov Decision Process (MDP) và Dynamic Programming trong Reinforcement Learning.

Các mục tiêu chính:

- Tìm hiểu Markov Property và Markov Chain.
- Biểu diễn transition probability bằng ma trận.
- Mô phỏng Markov Chain bằng NumPy.
- Tìm hiểu Reward, Return và Discount Factor.
- Phân biệt deterministic policy và stochastic policy.
- Tính State-Value Function V(s).
- Tính State-Action Value Function Q(s,a).
- Thực hiện Bellman Backup.
- Đọc transition model của FrozenLake-v1.
- Cài đặt Iterative Policy Evaluation.
- Cài đặt Policy Improvement.
- Cài đặt Policy Iteration.
- Cài đặt Value Iteration.
- Trích xuất optimal policy.
- Đánh giá policy bằng simulation.
- So sánh Value Iteration và Policy Iteration.
- Xây dựng chương trình Dynamic Programming hoàn chỉnh.

## 3. Cấu trúc thư mục

```text
Lab02/
├── README.md
├── requirements.txt
├── src/
│   ├── bai01.py
│   ├── bai02.py
│   ├── bai03.py
│   ├── bai04.py
│   ├── bai05.py
│   ├── bai06.py
│   ├── bai07.py
│   ├── bai08.py
│   ├── bai09.py
│   ├── bai10.py
│   ├── bai11.py
│   ├── bai12.py
│   ├── bai13.py
│   ├── bai14.py
│   ├── bai15.py
│   ├── bai16.py
│   ├── bai17.py
│   ├── bai18.py
│   ├── bai19.py
│   ├── bai20.py
│   ├── bai21.py
│   ├── bai22.py
│   ├── bai23.py
│   ├── bai24.py
│   ├── bai25.py
│   ├── bai26.py
│   ├── bai27.py
│   ├── bai28.py
│   ├── bai29.py
│   ├── bai30.py
│   ├── bai31.py
│   ├── bai32.py
│   ├── bai33.py
│   ├── bai34.py
│   ├── bai35.py
│   ├── bai36.py
│   ├── mdp_utils.py
│   └── main.py
├── notebooks/
│   └── Lab02_24100707_TranTheAnh.ipynb
├── figures/
│   ├── markov_distribution.png
│   ├── gamma_comparison.png
│   ├── value_iteration_convergence.png
│   ├── policy_iteration_convergence.png
│   └── algorithm_comparison.png
└── data/
    ├── README.md
    └── value_iteration_convergence.csv