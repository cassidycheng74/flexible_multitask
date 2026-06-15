# Existing Coverage
| Computation          | Covered? | Tasks                            |
| -------------------- | -------- | -------------------------------- |
| Pro/anti mapping     | ✓        | fdgo, fdanti, delaygo, delayanti |
| Working memory       | ✓        | delaygo, delayanti               |
| Evidence integration | ✓        | delaydm1, delaydm2               |
| Context gating       | ✓        | contextdelaydm1, contextdelaydm2 |
| Match detection      | ✓        | dmsgo, dmsnogo                   |
| Category abstraction | ✓        | dmcgo, dmcnogo                   |
| Online learning      | ✗        | —                                |
| Reversal learning    | ✗        | —                                |
| Few-shot learning    | ✗        | —                                |
| Associative memory   | ✗        | —                                |


# New Task Priority
| Task                   | New computation? | Relevant to plasticity? | Priority |
| ---------------------- | ---------------- | ----------------------- | -------- |
| T20 CueResponseAssoc   | Very high        | Very high               | 5        |
| T22 ReversalLearning   | Very high        | Very high               | 5        |
| T23 OnlineLinearReg    | Very high        | Very high               | 5        |
| T24 OnlineNonlinearReg | Very high        | Very high               | 5        |
| T25 FewShotClassif     | Very high        | Very high               | 4        |
| T12 MultiItemRecall    | Moderate         | Moderate                | 4        |
| T11 ExtendedMemory     | Low              | Moderate                | 3        |
