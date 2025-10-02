cutoff -> accuracy
| cutoff | perceptron_acc | gelm_acc | grvfl_acc |
|:---|:---|:---|:---|
| 2 | 0.298958 | 0.591146 | 0.574479 |
| 3 | 0.380729 | 0.593490 | 0.582812 |
| 4 | 0.448698 | 0.602865 | 0.580469 |

squeezing -> accuracy
| max_squeezing | perceptron_acc | gelm_acc | grvfl_acc |
|:---|:---|:---|:---|
| 0.20 | 0.370833 | 0.598698 | 0.584115 |
| 0.25 | 0.385417 | 0.597396 | 0.579427 |
| 0.30 | 0.375521 | 0.596615 | 0.578385 |
| 0.35 | 0.366406 | 0.594792 | 0.575781 |

modes -> accuracy
| modes | perceptron_acc | gelm_acc | grvfl_acc |
|:---|:---|:---|:---|
| 2 | 0.191667 | 0.457812 | 0.435417 |
| 3 | 0.363802 | 0.551823 | 0.539844 |
| 4 | 0.437760 | 0.638542 | 0.648698 |
| 5 | 0.504167 | 0.641927 | 0.647656 |

modes: 값이 커질수록 정확도가 꾸준히 상승, 4~5에서 증가폭이 둔화되긴 하지만 5에서 최고 성능
squeezing: 0.2일때 가장 안정, 0.2~0.25에서 최고치 형성 (cutoff, 잡음)
cutoff: 2~3에서 가장 높음.. 이유 먼지 모르겠음
other parameters: 이것들은 그냥 hyperparameter로 생각.

입력 상태를 coherent, thermal, displaced-squeezed 등 list of operation에 있는 state preparation 넣어보면서 CV system의 state들에 대한 이해도 높이기

Displaced-squeezed!



