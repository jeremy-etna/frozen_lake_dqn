    Input Layer       Hidden Layer      Output Layer
    (State)           (ReLU)            (Action Q-values)
                        
    [0] --------\     [0]-----\         [0]
                 \              \       [1]
    [1] -------- [ ] - ReLU - [ ] ---- [2]
                 /              /       [3]
    [2] --------/     [1]-----/
    .                   .
    .                   .
    .                   .
    [15]               [49]



Bellman equation:
Value(state) = max(Q(state, action) + gamma * Value(next_state))

deterministe versus stocastique

Procesus de Markov

Value(state) = max(Q(state, action) + gamma * sum(Probability(state, next_state, action) * Value(next_state)))

Q learning:
Q(state, action) = sum(Probability(state, next_state, action) * (Reward(state, next_state, action) + gamma * Value(next_state)))