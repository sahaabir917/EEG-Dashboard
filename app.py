import streamlit as st
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



# =========================================================
# 1) PASTE YOUR LOGS HERE
# =========================================================
log_exp1 = r"""
Epoch 1/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 72s 158ms/step - accuracy: 0.6738 - auc: 0.7460 - loss: 0.7592 - val_accuracy: 0.7396 - val_auc: 0.8348 - val_loss: 0.6707 - learning_rate: 3.0000e-04
Epoch 2/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 56s 133ms/step - accuracy: 0.7718 - auc: 0.8595 - loss: 0.6136 - val_accuracy: 0.7829 - val_auc: 0.8725 - val_loss: 0.6155 - learning_rate: 3.0000e-04
Epoch 3/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 59s 143ms/step - accuracy: 0.8239 - auc: 0.9057 - loss: 0.5274 - val_accuracy: 0.8362 - val_auc: 0.9247 - val_loss: 0.4989 - learning_rate: 3.0000e-04
Epoch 4/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 59s 141ms/step - accuracy: 0.8503 - auc: 0.9291 - loss: 0.4705 - val_accuracy: 0.8597 - val_auc: 0.9402 - val_loss: 0.4553 - learning_rate: 3.0000e-04
Epoch 5/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 61s 146ms/step - accuracy: 0.8723 - auc: 0.9463 - loss: 0.4221 - val_accuracy: 0.7926 - val_auc: 0.9156 - val_loss: 0.6039 - learning_rate: 3.0000e-04
Epoch 6/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 63s 151ms/step - accuracy: 0.8855 - auc: 0.9572 - loss: 0.3859 - val_accuracy: 0.8532 - val_auc: 0.9441 - val_loss: 0.4848 - learning_rate: 3.0000e-04
Epoch 7/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 59s 141ms/step - accuracy: 0.9034 - auc: 0.9662 - loss: 0.3522 - val_accuracy: 0.8538 - val_auc: 0.9528 - val_loss: 0.4657 - learning_rate: 3.0000e-04
Epoch 8/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 60s 144ms/step - accuracy: 0.9126 - auc: 0.9735 - loss: 0.3215 - val_accuracy: 0.8535 - val_auc: 0.9577 - val_loss: 0.4616 - learning_rate: 3.0000e-04
Epoch 9/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 76s 182ms/step - accuracy: 0.9354 - auc: 0.9848 - loss: 0.2687 - val_accuracy: 0.8903 - val_auc: 0.9708 - val_loss: 0.3931 - learning_rate: 1.5000e-04
Epoch 10/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 80s 193ms/step - accuracy: 0.9461 - auc: 0.9884 - loss: 0.2456 - val_accuracy: 0.9182 - val_auc: 0.9746 - val_loss: 0.3291 - learning_rate: 1.5000e-04
Epoch 11/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 77s 184ms/step - accuracy: 0.9520 - auc: 0.9903 - loss: 0.2308 - val_accuracy: 0.9257 - val_auc: 0.9781 - val_loss: 0.3069 - learning_rate: 1.5000e-04
Epoch 12/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 77s 185ms/step - accuracy: 0.9557 - auc: 0.9920 - loss: 0.2167 - val_accuracy: 0.9267 - val_auc: 0.9772 - val_loss: 0.3053 - learning_rate: 1.5000e-04
Epoch 13/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 195ms/step - accuracy: 0.9575 - auc: 0.9928 - loss: 0.2094 - val_accuracy: 0.9344 - val_auc: 0.9775 - val_loss: 0.3036 - learning_rate: 1.5000e-04
Epoch 14/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 79s 190ms/step - accuracy: 0.9630 - auc: 0.9940 - loss: 0.1980 - val_accuracy: 0.9252 - val_auc: 0.9757 - val_loss: 0.3204 - learning_rate: 1.5000e-04
Epoch 15/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 77s 186ms/step - accuracy: 0.9650 - auc: 0.9943 - loss: 0.1928 - val_accuracy: 0.9391 - val_auc: 0.9820 - val_loss: 0.2830 - learning_rate: 1.5000e-04
Epoch 16/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 77s 186ms/step - accuracy: 0.9669 - auc: 0.9953 - loss: 0.1820 - val_accuracy: 0.9419 - val_auc: 0.9799 - val_loss: 0.2841 - learning_rate: 1.5000e-04
Epoch 17/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 78s 188ms/step - accuracy: 0.9692 - auc: 0.9961 - loss: 0.1738 - val_accuracy: 0.9322 - val_auc: 0.9793 - val_loss: 0.3096 - learning_rate: 1.5000e-04
Epoch 18/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 78s 188ms/step - accuracy: 0.9688 - auc: 0.9961 - loss: 0.1720 - val_accuracy: 0.9388 - val_auc: 0.9801 - val_loss: 0.2962 - learning_rate: 1.5000e-04
Epoch 19/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 78s 187ms/step - accuracy: 0.9739 - auc: 0.9967 - loss: 0.1632 - val_accuracy: 0.9036 - val_auc: 0.9618 - val_loss: 0.4553 - learning_rate: 1.5000e-04
Epoch 20/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 194ms/step - accuracy: 0.9737 - auc: 0.9967 - loss: 0.1613 - val_accuracy: 0.9183 - val_auc: 0.9715 - val_loss: 0.3576 - learning_rate: 1.5000e-04
Epoch 21/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 80s 192ms/step - accuracy: 0.9814 - auc: 0.9986 - loss: 0.1386 - val_accuracy: 0.9486 - val_auc: 0.9812 - val_loss: 0.2747 - learning_rate: 7.5000e-05
Epoch 22/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 195ms/step - accuracy: 0.9829 - auc: 0.9988 - loss: 0.1351 - val_accuracy: 0.9496 - val_auc: 0.9813 - val_loss: 0.2875 - learning_rate: 7.5000e-05
Epoch 23/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 198ms/step - accuracy: 0.9844 - auc: 0.9988 - loss: 0.1320 - val_accuracy: 0.9227 - val_auc: 0.9729 - val_loss: 0.3719 - learning_rate: 7.5000e-05
Epoch 24/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 79s 190ms/step - accuracy: 0.9824 - auc: 0.9988 - loss: 0.1326 - val_accuracy: 0.9504 - val_auc: 0.9814 - val_loss: 0.2831 - learning_rate: 7.5000e-05
Epoch 25/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 80s 192ms/step - accuracy: 0.9847 - auc: 0.9989 - loss: 0.1265 - val_accuracy: 0.9499 - val_auc: 0.9820 - val_loss: 0.2764 - learning_rate: 7.5000e-05
Epoch 26/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 194ms/step - accuracy: 0.9851 - auc: 0.9990 - loss: 0.1248 - val_accuracy: 0.9370 - val_auc: 0.9765 - val_loss: 0.3430 - learning_rate: 7.5000e-05
Epoch 27/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 196ms/step - accuracy: 0.9853 - auc: 0.9987 - loss: 0.1247 - val_accuracy: 0.9513 - val_auc: 0.9817 - val_loss: 0.2778 - learning_rate: 7.5000e-05
Epoch 28/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 80s 193ms/step - accuracy: 0.9876 - auc: 0.9993 - loss: 0.1176 - val_accuracy: 0.9493 - val_auc: 0.9792 - val_loss: 0.2981 - learning_rate: 7.5000e-05
Epoch 29/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 77s 186ms/step - accuracy: 0.9859 - auc: 0.9992 - loss: 0.1187 - val_accuracy: 0.9511 - val_auc: 0.9814 - val_loss: 0.2813 - learning_rate: 7.5000e-05
Epoch 30/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 78s 187ms/step - accuracy: 0.9868 - auc: 0.9992 - loss: 0.1174 - val_accuracy: 0.9216 - val_auc: 0.9692 - val_loss: 0.4159 - learning_rate: 7.5000e-05
Epoch 31/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 80s 192ms/step - accuracy: 0.9881 - auc: 0.9994 - loss: 0.1132 - val_accuracy: 0.9528 - val_auc: 0.9805 - val_loss: 0.2786 - learning_rate: 7.5000e-05
Epoch 32/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 78s 187ms/step - accuracy: 0.9886 - auc: 0.9993 - loss: 0.1108 - val_accuracy: 0.9480 - val_auc: 0.9770 - val_loss: 0.3089 - learning_rate: 7.5000e-05
Epoch 33/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 80s 191ms/step - accuracy: 0.9884 - auc: 0.9994 - loss: 0.1103 - val_accuracy: 0.9349 - val_auc: 0.9749 - val_loss: 0.3563 - learning_rate: 7.5000e-05
Epoch 34/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 196ms/step - accuracy: 0.9877 - auc: 0.9993 - loss: 0.1119 - val_accuracy: 0.9541 - val_auc: 0.9802 - val_loss: 0.2870 - learning_rate: 7.5000e-05
Epoch 35/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 79s 189ms/step - accuracy: 0.9877 - auc: 0.9991 - loss: 0.1120 - val_accuracy: 0.9298 - val_auc: 0.9674 - val_loss: 0.3947 - learning_rate: 7.5000e-05
Epoch 36/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 79s 189ms/step - accuracy: 0.9897 - auc: 0.9995 - loss: 0.1054 - val_accuracy: 0.9526 - val_auc: 0.9800 - val_loss: 0.2969 - learning_rate: 7.5000e-05
Epoch 37/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 78s 187ms/step - accuracy: 0.9881 - auc: 0.9992 - loss: 0.1081 - val_accuracy: 0.9522 - val_auc: 0.9794 - val_loss: 0.2922 - learning_rate: 7.5000e-05
Epoch 38/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 79s 189ms/step - accuracy: 0.9894 - auc: 0.9992 - loss: 0.1061 - val_accuracy: 0.9320 - val_auc: 0.9718 - val_loss: 0.3819 - learning_rate: 7.5000e-05
Epoch 39/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 78s 187ms/step - accuracy: 0.9926 - auc: 0.9997 - loss: 0.0975 - val_accuracy: 0.9511 - val_auc: 0.9796 - val_loss: 0.2964 - learning_rate: 3.7500e-05
Epoch 40/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 78s 187ms/step - accuracy: 0.9924 - auc: 0.9998 - loss: 0.0952 - val_accuracy: 0.9389 - val_auc: 0.9733 - val_loss: 0.3619 - learning_rate: 3.7500e-05
Epoch 41/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 71s 170ms/step - accuracy: 0.9929 - auc: 0.9998 - loss: 0.0945 - val_accuracy: 0.9528 - val_auc: 0.9795 - val_loss: 0.3001 - learning_rate: 3.7500e-05
Epoch 42/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 47s 113ms/step - accuracy: 0.9926 - auc: 0.9997 - loss: 0.0947 - val_accuracy: 0.9451 - val_auc: 0.9746 - val_loss: 0.3393 - learning_rate: 3.7500e-05
Epoch 43/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 47s 113ms/step - accuracy: 0.9944 - auc: 0.9999 - loss: 0.0909 - val_accuracy: 0.9508 - val_auc: 0.9770 - val_loss: 0.3130 - learning_rate: 1.8750e-05
Epoch 44/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 47s 112ms/step - accuracy: 0.9941 - auc: 0.9998 - loss: 0.0910 - val_accuracy: 0.9537 - val_auc: 0.9803 - val_loss: 0.2942 - learning_rate: 1.8750e-05"""

log_exp2 = r"""
Epoch 1/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 48s 102ms/step - accuracy: 0.6938 - auc: 0.7715 - loss: 0.7460 - val_accuracy: 0.7502 - val_auc: 0.8519 - val_loss: 0.6629 - learning_rate: 3.0000e-04
Epoch 2/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 39s 93ms/step - accuracy: 0.7935 - auc: 0.8805 - loss: 0.6057 - val_accuracy: 0.8109 - val_auc: 0.9036 - val_loss: 0.5693 - learning_rate: 3.0000e-04
Epoch 3/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 39s 93ms/step - accuracy: 0.8459 - auc: 0.9248 - loss: 0.5197 - val_accuracy: 0.8585 - val_auc: 0.9338 - val_loss: 0.4975 - learning_rate: 3.0000e-04
Epoch 4/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 39s 94ms/step - accuracy: 0.8744 - auc: 0.9473 - loss: 0.4633 - val_accuracy: 0.8717 - val_auc: 0.9486 - val_loss: 0.4581 - learning_rate: 3.0000e-04
Epoch 5/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 37s 90ms/step - accuracy: 0.8941 - auc: 0.9624 - loss: 0.4182 - val_accuracy: 0.8679 - val_auc: 0.9441 - val_loss: 0.4904 - learning_rate: 3.0000e-04
Epoch 6/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 38s 90ms/step - accuracy: 0.9108 - auc: 0.9721 - loss: 0.3833 - val_accuracy: 0.8848 - val_auc: 0.9621 - val_loss: 0.4204 - learning_rate: 3.0000e-04
Epoch 7/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 37s 90ms/step - accuracy: 0.9291 - auc: 0.9808 - loss: 0.3461 - val_accuracy: 0.8747 - val_auc: 0.9512 - val_loss: 0.4442 - learning_rate: 3.0000e-04
Epoch 8/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 37s 90ms/step - accuracy: 0.9386 - auc: 0.9850 - loss: 0.3241 - val_accuracy: 0.9057 - val_auc: 0.9692 - val_loss: 0.3940 - learning_rate: 3.0000e-04
Epoch 9/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 37s 89ms/step - accuracy: 0.9483 - auc: 0.9892 - loss: 0.3002 - val_accuracy: 0.8091 - val_auc: 0.9488 - val_loss: 0.6174 - learning_rate: 3.0000e-04
Epoch 10/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 37s 90ms/step - accuracy: 0.9531 - auc: 0.9906 - loss: 0.2878 - val_accuracy: 0.9051 - val_auc: 0.9686 - val_loss: 0.3872 - learning_rate: 3.0000e-04
Epoch 11/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 37s 88ms/step - accuracy: 0.9595 - auc: 0.9926 - loss: 0.2725 - val_accuracy: 0.8924 - val_auc: 0.9711 - val_loss: 0.4301 - learning_rate: 3.0000e-04
Epoch 12/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 37s 89ms/step - accuracy: 0.9628 - auc: 0.9940 - loss: 0.2606 - val_accuracy: 0.8881 - val_auc: 0.9636 - val_loss: 0.4378 - learning_rate: 3.0000e-04
Epoch 13/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 38s 90ms/step - accuracy: 0.9648 - auc: 0.9949 - loss: 0.2521 - val_accuracy: 0.9275 - val_auc: 0.9755 - val_loss: 0.3423 - learning_rate: 3.0000e-04
Epoch 14/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 37s 90ms/step - accuracy: 0.9716 - auc: 0.9964 - loss: 0.2365 - val_accuracy: 0.9323 - val_auc: 0.9805 - val_loss: 0.3176 - learning_rate: 3.0000e-04
Epoch 15/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 37s 89ms/step - accuracy: 0.9716 - auc: 0.9965 - loss: 0.2325 - val_accuracy: 0.9013 - val_auc: 0.9665 - val_loss: 0.3872 - learning_rate: 3.0000e-04
Epoch 16/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 37s 90ms/step - accuracy: 0.9750 - auc: 0.9971 - loss: 0.2235 - val_accuracy: 0.9296 - val_auc: 0.9793 - val_loss: 0.3228 - learning_rate: 3.0000e-04
Epoch 17/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 38s 90ms/step - accuracy: 0.9736 - auc: 0.9968 - loss: 0.2232 - val_accuracy: 0.9245 - val_auc: 0.9768 - val_loss: 0.3311 - learning_rate: 3.0000e-04
Epoch 18/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 37s 90ms/step - accuracy: 0.9758 - auc: 0.9976 - loss: 0.2130 - val_accuracy: 0.8965 - val_auc: 0.9750 - val_loss: 0.3884 - learning_rate: 3.0000e-04
Epoch 19/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 37s 90ms/step - accuracy: 0.9867 - auc: 0.9991 - loss: 0.1916 - val_accuracy: 0.9477 - val_auc: 0.9883 - val_loss: 0.2695 - learning_rate: 1.5000e-04
Epoch 20/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 38s 90ms/step - accuracy: 0.9892 - auc: 0.9994 - loss: 0.1823 - val_accuracy: 0.9392 - val_auc: 0.9877 - val_loss: 0.2948 - learning_rate: 1.5000e-04
Epoch 21/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 37s 89ms/step - accuracy: 0.9905 - auc: 0.9996 - loss: 0.1772 - val_accuracy: 0.9579 - val_auc: 0.9894 - val_loss: 0.2554 - learning_rate: 1.5000e-04
Epoch 22/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 37s 90ms/step - accuracy: 0.9906 - auc: 0.9995 - loss: 0.1744 - val_accuracy: 0.9541 - val_auc: 0.9879 - val_loss: 0.2602 - learning_rate: 1.5000e-04
Epoch 23/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 37s 90ms/step - accuracy: 0.9922 - auc: 0.9996 - loss: 0.1703 - val_accuracy: 0.9316 - val_auc: 0.9841 - val_loss: 0.3046 - learning_rate: 1.5000e-04
Epoch 24/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 38s 91ms/step - accuracy: 0.9894 - auc: 0.9994 - loss: 0.1732 - val_accuracy: 0.9552 - val_auc: 0.9894 - val_loss: 0.2492 - learning_rate: 1.5000e-04
Epoch 25/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 38s 91ms/step - accuracy: 0.9904 - auc: 0.9994 - loss: 0.1682 - val_accuracy: 0.9547 - val_auc: 0.9887 - val_loss: 0.2501 - learning_rate: 1.5000e-04
Epoch 26/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 37s 90ms/step - accuracy: 0.9942 - auc: 0.9998 - loss: 0.1589 - val_accuracy: 0.9532 - val_auc: 0.9888 - val_loss: 0.2508 - learning_rate: 7.5000e-05
Epoch 27/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 38s 91ms/step - accuracy: 0.9947 - auc: 0.9998 - loss: 0.1556 - val_accuracy: 0.9583 - val_auc: 0.9900 - val_loss: 0.2361 - learning_rate: 7.5000e-05
Epoch 28/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 38s 91ms/step - accuracy: 0.9951 - auc: 0.9998 - loss: 0.1535 - val_accuracy: 0.9455 - val_auc: 0.9894 - val_loss: 0.2633 - learning_rate: 7.5000e-05
Epoch 29/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 39s 94ms/step - accuracy: 0.9947 - auc: 0.9999 - loss: 0.1531 - val_accuracy: 0.9496 - val_auc: 0.9896 - val_loss: 0.2518 - learning_rate: 7.5000e-05
Epoch 30/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 41s 98ms/step - accuracy: 0.9950 - auc: 0.9999 - loss: 0.1505 - val_accuracy: 0.9523 - val_auc: 0.9889 - val_loss: 0.2518 - learning_rate: 7.5000e-05
Epoch 31/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 39s 94ms/step - accuracy: 0.9950 - auc: 0.9999 - loss: 0.1494 - val_accuracy: 0.9454 - val_auc: 0.9875 - val_loss: 0.2652 - learning_rate: 7.5000e-05
Epoch 32/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 39s 94ms/step - accuracy: 0.9965 - auc: 0.9999 - loss: 0.1454 - val_accuracy: 0.9588 - val_auc: 0.9913 - val_loss: 0.2289 - learning_rate: 3.7500e-05
Epoch 33/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 39s 94ms/step - accuracy: 0.9968 - auc: 0.9999 - loss: 0.1434 - val_accuracy: 0.9474 - val_auc: 0.9897 - val_loss: 0.2566 - learning_rate: 3.7500e-05
Epoch 34/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 57s 136ms/step - accuracy: 0.9970 - auc: 0.9999 - loss: 0.1425 - val_accuracy: 0.9424 - val_auc: 0.9896 - val_loss: 0.2650 - learning_rate: 3.7500e-05
Epoch 35/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 92s 160ms/step - accuracy: 0.9964 - auc: 0.9999 - loss: 0.1423 - val_accuracy: 0.9505 - val_auc: 0.9902 - val_loss: 0.2473 - learning_rate: 3.7500e-05
Epoch 36/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 67s 160ms/step - accuracy: 0.9962 - auc: 0.9999 - loss: 0.1414 - val_accuracy: 0.9525 - val_auc: 0.9907 - val_loss: 0.2400 - learning_rate: 3.7500e-05
Epoch 37/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 87s 172ms/step - accuracy: 0.9973 - auc: 0.9999 - loss: 0.1388 - val_accuracy: 0.9573 - val_auc: 0.9912 - val_loss: 0.2288 - learning_rate: 1.8750e-05
Epoch 38/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 72s 173ms/step - accuracy: 0.9973 - auc: 1.0000 - loss: 0.1384 - val_accuracy: 0.9553 - val_auc: 0.9909 - val_loss: 0.2363 - learning_rate: 1.8750e-05
Epoch 39/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 77s 160ms/step - accuracy: 0.9973 - auc: 0.9999 - loss: 0.1381 - val_accuracy: 0.9349 - val_auc: 0.9894 - val_loss: 0.2791 - learning_rate: 1.8750e-05
Epoch 40/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 87s 172ms/step - accuracy: 0.9973 - auc: 1.0000 - loss: 0.1373 - val_accuracy: 0.9564 - val_auc: 0.9913 - val_loss: 0.2286 - learning_rate: 1.8750e-05
Epoch 41/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 77s 159ms/step - accuracy: 0.9974 - auc: 1.0000 - loss: 0.1364 - val_accuracy: 0.9562 - val_auc: 0.9912 - val_loss: 0.2307 - learning_rate: 9.3750e-06
Epoch 42/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 67s 160ms/step - accuracy: 0.9979 - auc: 1.0000 - loss: 0.1359 - val_accuracy: 0.9580 - val_auc: 0.9914 - val_loss: 0.2268 - learning_rate: 9.3750e-06
Epoch 43/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 159ms/step - accuracy: 0.9977 - auc: 1.0000 - loss: 0.1355 - val_accuracy: 0.9601 - val_auc: 0.9916 - val_loss: 0.2237 - learning_rate: 9.3750e-06
Epoch 44/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 86s 170ms/step - accuracy: 0.9977 - auc: 1.0000 - loss: 0.1352 - val_accuracy: 0.9562 - val_auc: 0.9913 - val_loss: 0.2313 - learning_rate: 9.3750e-06
Epoch 45/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 78s 160ms/step - accuracy: 0.9975 - auc: 1.0000 - loss: 0.1354 - val_accuracy: 0.9603 - val_auc: 0.9919 - val_loss: 0.2217 - learning_rate: 9.3750e-06
Epoch 46/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 160ms/step - accuracy: 0.9977 - auc: 1.0000 - loss: 0.1347 - val_accuracy: 0.9570 - val_auc: 0.9918 - val_loss: 0.2269 - learning_rate: 9.3750e-06
Epoch 47/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 66s 160ms/step - accuracy: 0.9976 - auc: 1.0000 - loss: 0.1343 - val_accuracy: 0.9577 - val_auc: 0.9917 - val_loss: 0.2273 - learning_rate: 9.3750e-06
Epoch 48/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 88s 173ms/step - accuracy: 0.9976 - auc: 1.0000 - loss: 0.1341 - val_accuracy: 0.9540 - val_auc: 0.9913 - val_loss: 0.2359 - learning_rate: 9.3750e-06
Epoch 49/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 172ms/step - accuracy: 0.9977 - auc: 1.0000 - loss: 0.1339 - val_accuracy: 0.9603 - val_auc: 0.9917 - val_loss: 0.2211 - learning_rate: 9.3750e-06
Epoch 50/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 78s 163ms/step - accuracy: 0.9979 - auc: 1.0000 - loss: 0.1336 - val_accuracy: 0.9580 - val_auc: 0.9914 - val_loss: 0.2250 - learning_rate: 4.6875e-06
Epoch 51/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 67s 161ms/step - accuracy: 0.9979 - auc: 1.0000 - loss: 0.1332 - val_accuracy: 0.9591 - val_auc: 0.9915 - val_loss: 0.2236 - learning_rate: 4.6875e-06
Epoch 52/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 161ms/step - accuracy: 0.9979 - auc: 1.0000 - loss: 0.1331 - val_accuracy: 0.9592 - val_auc: 0.9915 - val_loss: 0.2234 - learning_rate: 4.6875e-06
Epoch 53/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 71s 172ms/step - accuracy: 0.9979 - auc: 1.0000 - loss: 0.1330 - val_accuracy: 0.9574 - val_auc: 0.9914 - val_loss: 0.2271 - learning_rate: 4.6875e-06
Epoch 54/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 83s 173ms/step - accuracy: 0.9977 - auc: 1.0000 - loss: 0.1327 - val_accuracy: 0.9577 - val_auc: 0.9914 - val_loss: 0.2269 - learning_rate: 2.3438e-06
Epoch 55/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 172ms/step - accuracy: 0.9978 - auc: 1.0000 - loss: 0.1326 - val_accuracy: 0.9594 - val_auc: 0.9918 - val_loss: 0.2211 - learning_rate: 2.3438e-06
"""



log_exp3 = r"""
Epoch 1/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 45s 94ms/step - accuracy: 0.5947 - auc: 0.6351 - loss: 0.9710 - val_accuracy: 0.5633 - val_auc: 0.6784 - val_loss: 0.9424 - learning_rate: 2.0000e-04
Epoch 2/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 35s 84ms/step - accuracy: 0.6493 - auc: 0.7039 - loss: 0.8538 - val_accuracy: 0.5766 - val_auc: 0.6350 - val_loss: 0.8755 - learning_rate: 2.0000e-04
Epoch 3/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 35s 85ms/step - accuracy: 0.6766 - auc: 0.7495 - loss: 0.7717 - val_accuracy: 0.5764 - val_auc: 0.6796 - val_loss: 0.8326 - learning_rate: 2.0000e-04
Epoch 4/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 36s 85ms/step - accuracy: 0.6999 - auc: 0.7816 - loss: 0.7125 - val_accuracy: 0.6112 - val_auc: 0.6779 - val_loss: 0.8152 - learning_rate: 2.0000e-04
Epoch 5/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 36s 86ms/step - accuracy: 0.7291 - auc: 0.8109 - loss: 0.6637 - val_accuracy: 0.5439 - val_auc: 0.6601 - val_loss: 0.8126 - learning_rate: 2.0000e-04
Epoch 6/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 36s 85ms/step - accuracy: 0.7451 - auc: 0.8314 - loss: 0.6271 - val_accuracy: 0.6576 - val_auc: 0.7090 - val_loss: 0.7704 - learning_rate: 2.0000e-04
Epoch 7/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 36s 86ms/step - accuracy: 0.7614 - auc: 0.8501 - loss: 0.5913 - val_accuracy: 0.6650 - val_auc: 0.7180 - val_loss: 0.7531 - learning_rate: 2.0000e-04
Epoch 8/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 36s 86ms/step - accuracy: 0.7674 - auc: 0.8613 - loss: 0.5658 - val_accuracy: 0.6807 - val_auc: 0.7428 - val_loss: 0.7219 - learning_rate: 2.0000e-04
Epoch 9/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 36s 86ms/step - accuracy: 0.7774 - auc: 0.8707 - loss: 0.5451 - val_accuracy: 0.6482 - val_auc: 0.7147 - val_loss: 0.7395 - learning_rate: 2.0000e-04
Epoch 10/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 36s 86ms/step - accuracy: 0.7897 - auc: 0.8828 - loss: 0.5221 - val_accuracy: 0.6512 - val_auc: 0.7222 - val_loss: 0.7172 - learning_rate: 2.0000e-04
Epoch 11/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 36s 86ms/step - accuracy: 0.7969 - auc: 0.8930 - loss: 0.5002 - val_accuracy: 0.6779 - val_auc: 0.7275 - val_loss: 0.7224 - learning_rate: 2.0000e-04
Epoch 12/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 36s 86ms/step - accuracy: 0.8157 - auc: 0.9074 - loss: 0.4701 - val_accuracy: 0.6810 - val_auc: 0.7384 - val_loss: 0.7019 - learning_rate: 1.0000e-04
Epoch 13/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 37s 88ms/step - accuracy: 0.8224 - auc: 0.9142 - loss: 0.4543 - val_accuracy: 0.6668 - val_auc: 0.7264 - val_loss: 0.7138 - learning_rate: 1.0000e-04
Epoch 14/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 36s 87ms/step - accuracy: 0.8284 - auc: 0.9165 - loss: 0.4473 - val_accuracy: 0.6564 - val_auc: 0.7344 - val_loss: 0.7303 - learning_rate: 1.0000e-04
Epoch 15/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 36s 87ms/step - accuracy: 0.8386 - auc: 0.9254 - loss: 0.4286 - val_accuracy: 0.6635 - val_auc: 0.7215 - val_loss: 0.7217 - learning_rate: 5.0000e-05
Epoch 16/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 36s 87ms/step - accuracy: 0.8419 - auc: 0.9284 - loss: 0.4196 - val_accuracy: 0.6652 - val_auc: 0.7243 - val_loss: 0.7258 - learning_rate: 5.0000e-05
"""


log_exp4 = r"""
Epoch 1/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 38s 80ms/step - accuracy: 0.6705 - auc: 0.7393 - loss: 0.7672 - val_accuracy: 0.7399 - val_auc: 0.8262 - val_loss: 0.6847 - learning_rate: 3.0000e-04
Epoch 2/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 79ms/step - accuracy: 0.7644 - auc: 0.8458 - loss: 0.6354 - val_accuracy: 0.7852 - val_auc: 0.8777 - val_loss: 0.5855 - learning_rate: 3.0000e-04
Epoch 3/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 79ms/step - accuracy: 0.8051 - auc: 0.8928 - loss: 0.5497 - val_accuracy: 0.7700 - val_auc: 0.8719 - val_loss: 0.6006 - learning_rate: 3.0000e-04
Epoch 4/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 34s 81ms/step - accuracy: 0.8331 - auc: 0.9170 - loss: 0.4950 - val_accuracy: 0.8066 - val_auc: 0.9209 - val_loss: 0.5314 - learning_rate: 3.0000e-04
Epoch 5/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 34s 81ms/step - accuracy: 0.8615 - auc: 0.9406 - loss: 0.4330 - val_accuracy: 0.7748 - val_auc: 0.8945 - val_loss: 0.6198 - learning_rate: 3.0000e-04
Epoch 6/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 79ms/step - accuracy: 0.8797 - auc: 0.9531 - loss: 0.3945 - val_accuracy: 0.8467 - val_auc: 0.9299 - val_loss: 0.4541 - learning_rate: 3.0000e-04
Epoch 7/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 79ms/step - accuracy: 0.8979 - auc: 0.9656 - loss: 0.3509 - val_accuracy: 0.7864 - val_auc: 0.9211 - val_loss: 0.6813 - learning_rate: 3.0000e-04
Epoch 8/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 35s 83ms/step - accuracy: 0.9137 - auc: 0.9738 - loss: 0.3178 - val_accuracy: 0.8521 - val_auc: 0.9572 - val_loss: 0.4730 - learning_rate: 3.0000e-04
Epoch 9/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 80ms/step - accuracy: 0.9259 - auc: 0.9797 - loss: 0.2892 - val_accuracy: 0.9031 - val_auc: 0.9652 - val_loss: 0.3544 - learning_rate: 3.0000e-04
Epoch 10/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 80ms/step - accuracy: 0.9332 - auc: 0.9837 - loss: 0.2671 - val_accuracy: 0.9113 - val_auc: 0.9696 - val_loss: 0.3304 - learning_rate: 3.0000e-04
Epoch 11/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 80ms/step - accuracy: 0.9430 - auc: 0.9873 - loss: 0.2457 - val_accuracy: 0.8991 - val_auc: 0.9701 - val_loss: 0.3583 - learning_rate: 3.0000e-04
Epoch 12/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 80ms/step - accuracy: 0.9474 - auc: 0.9891 - loss: 0.2324 - val_accuracy: 0.9051 - val_auc: 0.9711 - val_loss: 0.3397 - learning_rate: 3.0000e-04
Epoch 13/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 80ms/step - accuracy: 0.9522 - auc: 0.9907 - loss: 0.2211 - val_accuracy: 0.9096 - val_auc: 0.9687 - val_loss: 0.3484 - learning_rate: 3.0000e-04
Epoch 14/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 79ms/step - accuracy: 0.9537 - auc: 0.9920 - loss: 0.2108 - val_accuracy: 0.8983 - val_auc: 0.9664 - val_loss: 0.3693 - learning_rate: 3.0000e-04
Epoch 15/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 79ms/step - accuracy: 0.9592 - auc: 0.9933 - loss: 0.1975 - val_accuracy: 0.9135 - val_auc: 0.9749 - val_loss: 0.3302 - learning_rate: 3.0000e-04
Epoch 16/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 80ms/step - accuracy: 0.9636 - auc: 0.9945 - loss: 0.1850 - val_accuracy: 0.9276 - val_auc: 0.9720 - val_loss: 0.3272 - learning_rate: 3.0000e-04
Epoch 17/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 80ms/step - accuracy: 0.9648 - auc: 0.9945 - loss: 0.1832 - val_accuracy: 0.8991 - val_auc: 0.9700 - val_loss: 0.3902 - learning_rate: 3.0000e-04
Epoch 18/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 80ms/step - accuracy: 0.9690 - auc: 0.9956 - loss: 0.1712 - val_accuracy: 0.9305 - val_auc: 0.9770 - val_loss: 0.2993 - learning_rate: 3.0000e-04
Epoch 19/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 79ms/step - accuracy: 0.9686 - auc: 0.9959 - loss: 0.1688 - val_accuracy: 0.9243 - val_auc: 0.9740 - val_loss: 0.3152 - learning_rate: 3.0000e-04
Epoch 20/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 80ms/step - accuracy: 0.9703 - auc: 0.9963 - loss: 0.1622 - val_accuracy: 0.9282 - val_auc: 0.9773 - val_loss: 0.2937 - learning_rate: 3.0000e-04
Epoch 21/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 79ms/step - accuracy: 0.9715 - auc: 0.9968 - loss: 0.1562 - val_accuracy: 0.8917 - val_auc: 0.9606 - val_loss: 0.4508 - learning_rate: 3.0000e-04
Epoch 22/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 80ms/step - accuracy: 0.9731 - auc: 0.9965 - loss: 0.1569 - val_accuracy: 0.9150 - val_auc: 0.9723 - val_loss: 0.3409 - learning_rate: 3.0000e-04
Epoch 23/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 80ms/step - accuracy: 0.9759 - auc: 0.9972 - loss: 0.1484 - val_accuracy: 0.8923 - val_auc: 0.9642 - val_loss: 0.4376 - learning_rate: 3.0000e-04
Epoch 24/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 80ms/step - accuracy: 0.9746 - auc: 0.9973 - loss: 0.1477 - val_accuracy: 0.9386 - val_auc: 0.9779 - val_loss: 0.2917 - learning_rate: 3.0000e-04
Epoch 25/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 80ms/step - accuracy: 0.9770 - auc: 0.9977 - loss: 0.1408 - val_accuracy: 0.9341 - val_auc: 0.9753 - val_loss: 0.3056 - learning_rate: 3.0000e-04
Epoch 26/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 34s 81ms/step - accuracy: 0.9782 - auc: 0.9975 - loss: 0.1397 - val_accuracy: 0.9085 - val_auc: 0.9714 - val_loss: 0.3679 - learning_rate: 3.0000e-04
Epoch 27/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 80ms/step - accuracy: 0.9781 - auc: 0.9975 - loss: 0.1401 - val_accuracy: 0.9440 - val_auc: 0.9803 - val_loss: 0.2781 - learning_rate: 3.0000e-04
Epoch 28/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 80ms/step - accuracy: 0.9795 - auc: 0.9980 - loss: 0.1347 - val_accuracy: 0.8944 - val_auc: 0.9588 - val_loss: 0.4676 - learning_rate: 3.0000e-04
Epoch 29/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 80ms/step - accuracy: 0.9790 - auc: 0.9982 - loss: 0.1312 - val_accuracy: 0.9263 - val_auc: 0.9738 - val_loss: 0.3191 - learning_rate: 3.0000e-04
Epoch 30/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 80ms/step - accuracy: 0.9791 - auc: 0.9980 - loss: 0.1322 - val_accuracy: 0.9170 - val_auc: 0.9722 - val_loss: 0.3608 - learning_rate: 3.0000e-04
Epoch 31/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 80ms/step - accuracy: 0.9804 - auc: 0.9981 - loss: 0.1278 - val_accuracy: 0.9398 - val_auc: 0.9792 - val_loss: 0.2765 - learning_rate: 3.0000e-04
Epoch 32/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 80ms/step - accuracy: 0.9885 - auc: 0.9995 - loss: 0.1056 - val_accuracy: 0.9481 - val_auc: 0.9809 - val_loss: 0.2744 - learning_rate: 1.5000e-04
Epoch 33/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 80ms/step - accuracy: 0.9919 - auc: 0.9996 - loss: 0.0972 - val_accuracy: 0.9304 - val_auc: 0.9735 - val_loss: 0.3516 - learning_rate: 1.5000e-04
Epoch 34/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 80ms/step - accuracy: 0.9912 - auc: 0.9996 - loss: 0.0970 - val_accuracy: 0.9304 - val_auc: 0.9729 - val_loss: 0.3533 - learning_rate: 1.5000e-04
Epoch 35/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 80ms/step - accuracy: 0.9914 - auc: 0.9996 - loss: 0.0951 - val_accuracy: 0.9425 - val_auc: 0.9768 - val_loss: 0.3066 - learning_rate: 1.5000e-04
Epoch 36/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 80ms/step - accuracy: 0.9917 - auc: 0.9995 - loss: 0.0931 - val_accuracy: 0.9449 - val_auc: 0.9774 - val_loss: 0.2985 - learning_rate: 1.5000e-04
Epoch 37/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 80ms/step - accuracy: 0.9937 - auc: 0.9998 - loss: 0.0867 - val_accuracy: 0.9553 - val_auc: 0.9820 - val_loss: 0.2652 - learning_rate: 7.5000e-05
Epoch 38/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 34s 81ms/step - accuracy: 0.9946 - auc: 0.9999 - loss: 0.0819 - val_accuracy: 0.9468 - val_auc: 0.9769 - val_loss: 0.3015 - learning_rate: 7.5000e-05
Epoch 39/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 80ms/step - accuracy: 0.9950 - auc: 0.9999 - loss: 0.0811 - val_accuracy: 0.9577 - val_auc: 0.9807 - val_loss: 0.2610 - learning_rate: 7.5000e-05
Epoch 40/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 34s 81ms/step - accuracy: 0.9951 - auc: 0.9999 - loss: 0.0791 - val_accuracy: 0.9576 - val_auc: 0.9799 - val_loss: 0.2701 - learning_rate: 7.5000e-05
Epoch 41/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 34s 81ms/step - accuracy: 0.9955 - auc: 0.9999 - loss: 0.0775 - val_accuracy: 0.9481 - val_auc: 0.9781 - val_loss: 0.2944 - learning_rate: 7.5000e-05
Epoch 42/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 34s 81ms/step - accuracy: 0.9958 - auc: 0.9999 - loss: 0.0756 - val_accuracy: 0.9525 - val_auc: 0.9805 - val_loss: 0.2811 - learning_rate: 3.7500e-05
Epoch 43/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 34s 81ms/step - accuracy: 0.9967 - auc: 0.9999 - loss: 0.0731 - val_accuracy: 0.9477 - val_auc: 0.9795 - val_loss: 0.2997 - learning_rate: 3.7500e-05
Epoch 44/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 33s 80ms/step - accuracy: 0.9965 - auc: 0.9999 - loss: 0.0731 - val_accuracy: 0.9514 - val_auc: 0.9806 - val_loss: 0.2851 - learning_rate: 3.7500e-05
Epoch 45/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 34s 81ms/step - accuracy: 0.9966 - auc: 0.9999 - loss: 0.0723 - val_accuracy: 0.9516 - val_auc: 0.9793 - val_loss: 0.2880 - learning_rate: 3.7500e-05
Epoch 46/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 34s 82ms/step - accuracy: 0.9971 - auc: 0.9999 - loss: 0.0705 - val_accuracy: 0.9562 - val_auc: 0.9812 - val_loss: 0.2723 - learning_rate: 1.8750e-05
Epoch 47/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 34s 82ms/step - accuracy: 0.9971 - auc: 1.0000 - loss: 0.0692 - val_accuracy: 0.9526 - val_auc: 0.9785 - val_loss: 0.2873 - learning_rate: 1.8750e-05
"""

log_exp7 = r"""
Epoch 1/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 113ms/step - accuracy: 0.5816 - auc: 0.6191 - loss: 0.8507
Epoch 1: val_auc improved from None to 0.76538, saving model to exp7_cnn_bilstm_best.keras

Epoch 1: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 64s 140ms/step - accuracy: 0.6180 - auc: 0.6693 - loss: 0.8268 - val_accuracy: 0.6820 - val_auc: 0.7654 - val_loss: 0.7654 - learning_rate: 3.0000e-04
Epoch 2/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 166ms/step - accuracy: 0.6882 - auc: 0.7529 - loss: 0.7631
Epoch 2: val_auc improved from 0.76538 to 0.82391, saving model to exp7_cnn_bilstm_best.keras

Epoch 2: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 99s 180ms/step - accuracy: 0.6928 - auc: 0.7616 - loss: 0.7513 - val_accuracy: 0.7459 - val_auc: 0.8239 - val_loss: 0.6946 - learning_rate: 3.0000e-04
Epoch 3/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 168ms/step - accuracy: 0.7259 - auc: 0.8066 - loss: 0.7045
Epoch 3: val_auc improved from 0.82391 to 0.86590, saving model to exp7_cnn_bilstm_best.keras

Epoch 3: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 182ms/step - accuracy: 0.7329 - auc: 0.8154 - loss: 0.6928 - val_accuracy: 0.7712 - val_auc: 0.8659 - val_loss: 0.6495 - learning_rate: 3.0000e-04
Epoch 4/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.7553 - auc: 0.8387 - loss: 0.6621
Epoch 4: val_auc improved from 0.86590 to 0.89040, saving model to exp7_cnn_bilstm_best.keras

Epoch 4: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 83s 184ms/step - accuracy: 0.7598 - auc: 0.8445 - loss: 0.6539 - val_accuracy: 0.8034 - val_auc: 0.8904 - val_loss: 0.6004 - learning_rate: 3.0000e-04
Epoch 5/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 169ms/step - accuracy: 0.7800 - auc: 0.8679 - loss: 0.6234
Epoch 5: val_auc improved from 0.89040 to 0.90495, saving model to exp7_cnn_bilstm_best.keras

Epoch 5: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 86s 194ms/step - accuracy: 0.7891 - auc: 0.8752 - loss: 0.6140 - val_accuracy: 0.8002 - val_auc: 0.9049 - val_loss: 0.5880 - learning_rate: 3.0000e-04
Epoch 6/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.8028 - auc: 0.8886 - loss: 0.5928
Epoch 6: val_auc improved from 0.90495 to 0.93262, saving model to exp7_cnn_bilstm_best.keras

Epoch 6: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 83s 195ms/step - accuracy: 0.8089 - auc: 0.8952 - loss: 0.5844 - val_accuracy: 0.8518 - val_auc: 0.9326 - val_loss: 0.5370 - learning_rate: 3.0000e-04
Epoch 7/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.8214 - auc: 0.9071 - loss: 0.5654
Epoch 7: val_auc improved from 0.93262 to 0.93876, saving model to exp7_cnn_bilstm_best.keras

Epoch 7: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 77s 185ms/step - accuracy: 0.8239 - auc: 0.9103 - loss: 0.5605 - val_accuracy: 0.8533 - val_auc: 0.9388 - val_loss: 0.5226 - learning_rate: 3.0000e-04
Epoch 8/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 172ms/step - accuracy: 0.8333 - auc: 0.9184 - loss: 0.5462
Epoch 8: val_auc improved from 0.93876 to 0.95031, saving model to exp7_cnn_bilstm_best.keras

Epoch 8: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 83s 187ms/step - accuracy: 0.8422 - auc: 0.9239 - loss: 0.5381 - val_accuracy: 0.8724 - val_auc: 0.9503 - val_loss: 0.4972 - learning_rate: 3.0000e-04
Epoch 9/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.8519 - auc: 0.9316 - loss: 0.5242
Epoch 9: val_auc improved from 0.95031 to 0.95590, saving model to exp7_cnn_bilstm_best.keras

Epoch 9: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 185ms/step - accuracy: 0.8563 - auc: 0.9347 - loss: 0.5190 - val_accuracy: 0.8798 - val_auc: 0.9559 - val_loss: 0.4825 - learning_rate: 3.0000e-04
Epoch 10/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 172ms/step - accuracy: 0.8665 - auc: 0.9429 - loss: 0.5043
Epoch 10: val_auc improved from 0.95590 to 0.96319, saving model to exp7_cnn_bilstm_best.keras

Epoch 10: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 87s 197ms/step - accuracy: 0.8678 - auc: 0.9434 - loss: 0.5024 - val_accuracy: 0.8830 - val_auc: 0.9632 - val_loss: 0.4793 - learning_rate: 3.0000e-04
Epoch 11/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.8753 - auc: 0.9493 - loss: 0.4907
Epoch 11: val_auc improved from 0.96319 to 0.96791, saving model to exp7_cnn_bilstm_best.keras

Epoch 11: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 141s 195ms/step - accuracy: 0.8742 - auc: 0.9486 - loss: 0.4909 - val_accuracy: 0.8950 - val_auc: 0.9679 - val_loss: 0.4597 - learning_rate: 3.0000e-04
Epoch 12/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 172ms/step - accuracy: 0.8807 - auc: 0.9533 - loss: 0.4814
Epoch 12: val_auc improved from 0.96791 to 0.97092, saving model to exp7_cnn_bilstm_best.keras

Epoch 12: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 78s 187ms/step - accuracy: 0.8820 - auc: 0.9537 - loss: 0.4798 - val_accuracy: 0.8902 - val_auc: 0.9709 - val_loss: 0.4631 - learning_rate: 3.0000e-04
Epoch 13/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.8904 - auc: 0.9584 - loss: 0.4695
Epoch 13: val_auc improved from 0.97092 to 0.97230, saving model to exp7_cnn_bilstm_best.keras

Epoch 13: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 86s 195ms/step - accuracy: 0.8932 - auc: 0.9592 - loss: 0.4672 - val_accuracy: 0.9006 - val_auc: 0.9723 - val_loss: 0.4452 - learning_rate: 3.0000e-04
Epoch 14/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 171ms/step - accuracy: 0.8932 - auc: 0.9607 - loss: 0.4627
Epoch 14: val_auc improved from 0.97230 to 0.97650, saving model to exp7_cnn_bilstm_best.keras

Epoch 14: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 196ms/step - accuracy: 0.8934 - auc: 0.9618 - loss: 0.4608 - val_accuracy: 0.9076 - val_auc: 0.9765 - val_loss: 0.4368 - learning_rate: 3.0000e-04
Epoch 15/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.8983 - auc: 0.9651 - loss: 0.4525
Epoch 15: val_auc improved from 0.97650 to 0.97887, saving model to exp7_cnn_bilstm_best.keras

Epoch 15: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 77s 185ms/step - accuracy: 0.9018 - auc: 0.9658 - loss: 0.4509 - val_accuracy: 0.9143 - val_auc: 0.9789 - val_loss: 0.4299 - learning_rate: 3.0000e-04
Epoch 16/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 172ms/step - accuracy: 0.9061 - auc: 0.9688 - loss: 0.4444
Epoch 16: val_auc improved from 0.97887 to 0.97912, saving model to exp7_cnn_bilstm_best.keras

Epoch 16: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 87s 197ms/step - accuracy: 0.9049 - auc: 0.9683 - loss: 0.4446 - val_accuracy: 0.9177 - val_auc: 0.9791 - val_loss: 0.4226 - learning_rate: 3.0000e-04
Epoch 17/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9096 - auc: 0.9719 - loss: 0.4359
Epoch 17: val_auc improved from 0.97912 to 0.98240, saving model to exp7_cnn_bilstm_best.keras

Epoch 17: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 196ms/step - accuracy: 0.9095 - auc: 0.9712 - loss: 0.4369 - val_accuracy: 0.9240 - val_auc: 0.9824 - val_loss: 0.4145 - learning_rate: 3.0000e-04
Epoch 18/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 171ms/step - accuracy: 0.9178 - auc: 0.9748 - loss: 0.4291
Epoch 18: val_auc did not improve from 0.98240
416/416 ━━━━━━━━━━━━━━━━━━━━ 78s 186ms/step - accuracy: 0.9156 - auc: 0.9737 - loss: 0.4302 - val_accuracy: 0.9203 - val_auc: 0.9796 - val_loss: 0.4192 - learning_rate: 3.0000e-04
Epoch 19/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9186 - auc: 0.9760 - loss: 0.4247
Epoch 19: val_auc improved from 0.98240 to 0.98273, saving model to exp7_cnn_bilstm_best.keras

Epoch 19: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 77s 184ms/step - accuracy: 0.9200 - auc: 0.9756 - loss: 0.4245 - val_accuracy: 0.9276 - val_auc: 0.9827 - val_loss: 0.4058 - learning_rate: 3.0000e-04
Epoch 20/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 171ms/step - accuracy: 0.9231 - auc: 0.9768 - loss: 0.4204
Epoch 20: val_auc improved from 0.98273 to 0.98365, saving model to exp7_cnn_bilstm_best.keras

Epoch 20: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 87s 197ms/step - accuracy: 0.9220 - auc: 0.9771 - loss: 0.4199 - val_accuracy: 0.9307 - val_auc: 0.9837 - val_loss: 0.4044 - learning_rate: 3.0000e-04
Epoch 21/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9246 - auc: 0.9778 - loss: 0.4164
Epoch 21: val_auc improved from 0.98365 to 0.98501, saving model to exp7_cnn_bilstm_best.keras

Epoch 21: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 77s 185ms/step - accuracy: 0.9229 - auc: 0.9778 - loss: 0.4164 - val_accuracy: 0.9373 - val_auc: 0.9850 - val_loss: 0.3947 - learning_rate: 3.0000e-04
Epoch 22/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9282 - auc: 0.9803 - loss: 0.4104
Epoch 22: val_auc improved from 0.98501 to 0.98513, saving model to exp7_cnn_bilstm_best.keras

Epoch 22: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 195ms/step - accuracy: 0.9287 - auc: 0.9802 - loss: 0.4100 - val_accuracy: 0.9325 - val_auc: 0.9851 - val_loss: 0.3987 - learning_rate: 3.0000e-04
Epoch 23/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 169ms/step - accuracy: 0.9313 - auc: 0.9824 - loss: 0.4045
Epoch 23: val_auc did not improve from 0.98513
416/416 ━━━━━━━━━━━━━━━━━━━━ 77s 184ms/step - accuracy: 0.9293 - auc: 0.9812 - loss: 0.4069 - val_accuracy: 0.9246 - val_auc: 0.9841 - val_loss: 0.4102 - learning_rate: 3.0000e-04
Epoch 24/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 171ms/step - accuracy: 0.9346 - auc: 0.9819 - loss: 0.4034
Epoch 24: val_auc improved from 0.98513 to 0.98595, saving model to exp7_cnn_bilstm_best.keras

Epoch 24: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 77s 186ms/step - accuracy: 0.9342 - auc: 0.9817 - loss: 0.4035 - val_accuracy: 0.9383 - val_auc: 0.9860 - val_loss: 0.3906 - learning_rate: 3.0000e-04
Epoch 25/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 171ms/step - accuracy: 0.9374 - auc: 0.9844 - loss: 0.3969
Epoch 25: val_auc improved from 0.98595 to 0.98667, saving model to exp7_cnn_bilstm_best.keras

Epoch 25: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 186ms/step - accuracy: 0.9356 - auc: 0.9835 - loss: 0.3988 - val_accuracy: 0.9445 - val_auc: 0.9867 - val_loss: 0.3839 - learning_rate: 3.0000e-04
Epoch 26/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 172ms/step - accuracy: 0.9394 - auc: 0.9838 - loss: 0.3967
Epoch 26: val_auc improved from 0.98667 to 0.98729, saving model to exp7_cnn_bilstm_best.keras

Epoch 26: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 187ms/step - accuracy: 0.9375 - auc: 0.9835 - loss: 0.3976 - val_accuracy: 0.9341 - val_auc: 0.9873 - val_loss: 0.3893 - learning_rate: 3.0000e-04
Epoch 27/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9388 - auc: 0.9849 - loss: 0.3933
Epoch 27: val_auc improved from 0.98729 to 0.98921, saving model to exp7_cnn_bilstm_best.keras

Epoch 27: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 195ms/step - accuracy: 0.9381 - auc: 0.9840 - loss: 0.3949 - val_accuracy: 0.9463 - val_auc: 0.9892 - val_loss: 0.3777 - learning_rate: 3.0000e-04
Epoch 28/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 172ms/step - accuracy: 0.9440 - auc: 0.9865 - loss: 0.3887
Epoch 28: val_auc did not improve from 0.98921
416/416 ━━━━━━━━━━━━━━━━━━━━ 79s 187ms/step - accuracy: 0.9428 - auc: 0.9859 - loss: 0.3898 - val_accuracy: 0.9454 - val_auc: 0.9889 - val_loss: 0.3803 - learning_rate: 3.0000e-04
Epoch 29/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9439 - auc: 0.9870 - loss: 0.3858
Epoch 29: val_auc did not improve from 0.98921
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 195ms/step - accuracy: 0.9434 - auc: 0.9864 - loss: 0.3873 - val_accuracy: 0.9398 - val_auc: 0.9890 - val_loss: 0.3879 - learning_rate: 3.0000e-04
Epoch 30/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 171ms/step - accuracy: 0.9416 - auc: 0.9855 - loss: 0.3904
Epoch 30: val_auc did not improve from 0.98921
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 196ms/step - accuracy: 0.9430 - auc: 0.9860 - loss: 0.3888 - val_accuracy: 0.9406 - val_auc: 0.9875 - val_loss: 0.3837 - learning_rate: 3.0000e-04
Epoch 31/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 168ms/step - accuracy: 0.9469 - auc: 0.9880 - loss: 0.3823
Epoch 31: ReduceLROnPlateau reducing learning rate to 0.0001500000071246177.

Epoch 31: val_auc did not improve from 0.98921
416/416 ━━━━━━━━━━━━━━━━━━━━ 77s 183ms/step - accuracy: 0.9471 - auc: 0.9881 - loss: 0.3822 - val_accuracy: 0.9437 - val_auc: 0.9883 - val_loss: 0.3821 - learning_rate: 3.0000e-04
Epoch 32/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 172ms/step - accuracy: 0.9549 - auc: 0.9916 - loss: 0.3708
Epoch 32: val_auc improved from 0.98921 to 0.99115, saving model to exp7_cnn_bilstm_best.keras

Epoch 32: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 197ms/step - accuracy: 0.9569 - auc: 0.9917 - loss: 0.3688 - val_accuracy: 0.9486 - val_auc: 0.9912 - val_loss: 0.3728 - learning_rate: 1.5000e-04
Epoch 33/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9591 - auc: 0.9921 - loss: 0.3662
Epoch 33: val_auc improved from 0.99115 to 0.99213, saving model to exp7_cnn_bilstm_best.keras

Epoch 33: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 137s 185ms/step - accuracy: 0.9591 - auc: 0.9923 - loss: 0.3659 - val_accuracy: 0.9446 - val_auc: 0.9921 - val_loss: 0.3747 - learning_rate: 1.5000e-04
Epoch 34/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9624 - auc: 0.9933 - loss: 0.3619
Epoch 34: val_auc improved from 0.99213 to 0.99229, saving model to exp7_cnn_bilstm_best.keras

Epoch 34: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 86s 195ms/step - accuracy: 0.9598 - auc: 0.9927 - loss: 0.3641 - val_accuracy: 0.9535 - val_auc: 0.9923 - val_loss: 0.3639 - learning_rate: 1.5000e-04
Epoch 35/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9667 - auc: 0.9945 - loss: 0.3564
Epoch 35: val_auc did not improve from 0.99229
416/416 ━━━━━━━━━━━━━━━━━━━━ 77s 184ms/step - accuracy: 0.9637 - auc: 0.9936 - loss: 0.3592 - val_accuracy: 0.9583 - val_auc: 0.9922 - val_loss: 0.3600 - learning_rate: 1.5000e-04
Epoch 36/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9639 - auc: 0.9941 - loss: 0.3578
Epoch 36: val_auc improved from 0.99229 to 0.99286, saving model to exp7_cnn_bilstm_best.keras

Epoch 36: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 185ms/step - accuracy: 0.9646 - auc: 0.9938 - loss: 0.3574 - val_accuracy: 0.9577 - val_auc: 0.9929 - val_loss: 0.3591 - learning_rate: 1.5000e-04
Epoch 37/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 168ms/step - accuracy: 0.9651 - auc: 0.9944 - loss: 0.3538
Epoch 37: val_auc did not improve from 0.99286
416/416 ━━━━━━━━━━━━━━━━━━━━ 86s 193ms/step - accuracy: 0.9646 - auc: 0.9943 - loss: 0.3544 - val_accuracy: 0.9583 - val_auc: 0.9928 - val_loss: 0.3568 - learning_rate: 1.5000e-04
Epoch 38/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 171ms/step - accuracy: 0.9677 - auc: 0.9946 - loss: 0.3513
Epoch 38: val_auc did not improve from 0.99286
416/416 ━━━━━━━━━━━━━━━━━━━━ 83s 196ms/step - accuracy: 0.9661 - auc: 0.9941 - loss: 0.3533 - val_accuracy: 0.9535 - val_auc: 0.9923 - val_loss: 0.3605 - learning_rate: 1.5000e-04
Epoch 39/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 169ms/step - accuracy: 0.9675 - auc: 0.9946 - loss: 0.3516
Epoch 39: val_auc did not improve from 0.99286
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 194ms/step - accuracy: 0.9660 - auc: 0.9944 - loss: 0.3524 - val_accuracy: 0.9567 - val_auc: 0.9927 - val_loss: 0.3541 - learning_rate: 1.5000e-04
Epoch 40/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9679 - auc: 0.9945 - loss: 0.3510
Epoch 40: ReduceLROnPlateau reducing learning rate to 7.500000356230885e-05.

Epoch 40: val_auc did not improve from 0.99286
416/416 ━━━━━━━━━━━━━━━━━━━━ 78s 185ms/step - accuracy: 0.9670 - auc: 0.9944 - loss: 0.3517 - val_accuracy: 0.9552 - val_auc: 0.9927 - val_loss: 0.3584 - learning_rate: 1.5000e-04
Epoch 41/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 169ms/step - accuracy: 0.9700 - auc: 0.9949 - loss: 0.3485
Epoch 41: val_auc improved from 0.99286 to 0.99406, saving model to exp7_cnn_bilstm_best.keras

Epoch 41: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 184ms/step - accuracy: 0.9715 - auc: 0.9957 - loss: 0.3455 - val_accuracy: 0.9619 - val_auc: 0.9941 - val_loss: 0.3509 - learning_rate: 7.5000e-05
Epoch 42/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 173ms/step - accuracy: 0.9727 - auc: 0.9965 - loss: 0.3412
Epoch 42: val_auc did not improve from 0.99406
416/416 ━━━━━━━━━━━━━━━━━━━━ 83s 188ms/step - accuracy: 0.9722 - auc: 0.9960 - loss: 0.3429 - val_accuracy: 0.9610 - val_auc: 0.9937 - val_loss: 0.3505 - learning_rate: 7.5000e-05
Epoch 43/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9761 - auc: 0.9968 - loss: 0.3389
Epoch 43: val_auc did not improve from 0.99406
416/416 ━━━━━━━━━━━━━━━━━━━━ 85s 195ms/step - accuracy: 0.9747 - auc: 0.9962 - loss: 0.3412 - val_accuracy: 0.9604 - val_auc: 0.9940 - val_loss: 0.3515 - learning_rate: 7.5000e-05
Epoch 44/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9745 - auc: 0.9971 - loss: 0.3382
Epoch 44: val_auc did not improve from 0.99406
416/416 ━━━━━━━━━━━━━━━━━━━━ 78s 185ms/step - accuracy: 0.9739 - auc: 0.9968 - loss: 0.3392 - val_accuracy: 0.9669 - val_auc: 0.9938 - val_loss: 0.3448 - learning_rate: 7.5000e-05
Epoch 45/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 171ms/step - accuracy: 0.9728 - auc: 0.9964 - loss: 0.3397
Epoch 45: val_auc improved from 0.99406 to 0.99471, saving model to exp7_cnn_bilstm_best.keras

Epoch 45: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 186ms/step - accuracy: 0.9731 - auc: 0.9965 - loss: 0.3394 - val_accuracy: 0.9639 - val_auc: 0.9947 - val_loss: 0.3452 - learning_rate: 7.5000e-05
Epoch 46/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9778 - auc: 0.9972 - loss: 0.3357
Epoch 46: val_auc did not improve from 0.99471
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 185ms/step - accuracy: 0.9766 - auc: 0.9969 - loss: 0.3365 - val_accuracy: 0.9665 - val_auc: 0.9946 - val_loss: 0.3459 - learning_rate: 7.5000e-05
Epoch 47/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 171ms/step - accuracy: 0.9772 - auc: 0.9971 - loss: 0.3354
Epoch 47: val_auc did not improve from 0.99471
416/416 ━━━━━━━━━━━━━━━━━━━━ 78s 186ms/step - accuracy: 0.9754 - auc: 0.9966 - loss: 0.3376 - val_accuracy: 0.9618 - val_auc: 0.9942 - val_loss: 0.3475 - learning_rate: 7.5000e-05
Epoch 48/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 169ms/step - accuracy: 0.9767 - auc: 0.9973 - loss: 0.3339
Epoch 48: val_auc did not improve from 0.99471
416/416 ━━━━━━━━━━━━━━━━━━━━ 85s 194ms/step - accuracy: 0.9762 - auc: 0.9970 - loss: 0.3353 - val_accuracy: 0.9625 - val_auc: 0.9935 - val_loss: 0.3477 - learning_rate: 7.5000e-05
Epoch 49/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 167ms/step - accuracy: 0.9771 - auc: 0.9971 - loss: 0.3339
Epoch 49: ReduceLROnPlateau reducing learning rate to 3.7500001781154424e-05.

Epoch 49: val_auc did not improve from 0.99471
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 192ms/step - accuracy: 0.9774 - auc: 0.9973 - loss: 0.3337 - val_accuracy: 0.9636 - val_auc: 0.9940 - val_loss: 0.3457 - learning_rate: 7.5000e-05
Epoch 50/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 171ms/step - accuracy: 0.9789 - auc: 0.9980 - loss: 0.3302
Epoch 50: val_auc did not improve from 0.99471
416/416 ━━━━━━━━━━━━━━━━━━━━ 83s 195ms/step - accuracy: 0.9792 - auc: 0.9977 - loss: 0.3311 - val_accuracy: 0.9642 - val_auc: 0.9941 - val_loss: 0.3446 - learning_rate: 3.7500e-05
Epoch 51/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 169ms/step - accuracy: 0.9768 - auc: 0.9975 - loss: 0.3326
Epoch 51: val_auc did not improve from 0.99471
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 193ms/step - accuracy: 0.9782 - auc: 0.9977 - loss: 0.3314 - val_accuracy: 0.9612 - val_auc: 0.9944 - val_loss: 0.3461 - learning_rate: 3.7500e-05
Epoch 52/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 171ms/step - accuracy: 0.9808 - auc: 0.9977 - loss: 0.3302
Epoch 52: val_auc improved from 0.99471 to 0.99475, saving model to exp7_cnn_bilstm_best.keras

Epoch 52: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 196ms/step - accuracy: 0.9800 - auc: 0.9976 - loss: 0.3303 - val_accuracy: 0.9644 - val_auc: 0.9947 - val_loss: 0.3428 - learning_rate: 3.7500e-05
Epoch 53/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 172ms/step - accuracy: 0.9805 - auc: 0.9978 - loss: 0.3285
Epoch 53: ReduceLROnPlateau reducing learning rate to 1.8750000890577212e-05.

Epoch 53: val_auc did not improve from 0.99475
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 197ms/step - accuracy: 0.9800 - auc: 0.9979 - loss: 0.3289 - val_accuracy: 0.9633 - val_auc: 0.9943 - val_loss: 0.3428 - learning_rate: 3.7500e-05
Epoch 54/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 173ms/step - accuracy: 0.9813 - auc: 0.9982 - loss: 0.3279
Epoch 54: val_auc did not improve from 0.99475
416/416 ━━━━━━━━━━━━━━━━━━━━ 78s 188ms/step - accuracy: 0.9807 - auc: 0.9981 - loss: 0.3282 - val_accuracy: 0.9654 - val_auc: 0.9943 - val_loss: 0.3411 - learning_rate: 1.8750e-05
Epoch 55/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 171ms/step - accuracy: 0.9822 - auc: 0.9984 - loss: 0.3266
Epoch 55: val_auc did not improve from 0.99475
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 196ms/step - accuracy: 0.9807 - auc: 0.9982 - loss: 0.3274 - val_accuracy: 0.9644 - val_auc: 0.9942 - val_loss: 0.3431 - learning_rate: 1.8750e-05
Epoch 56/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 171ms/step - accuracy: 0.9825 - auc: 0.9980 - loss: 0.3263
Epoch 56: val_auc did not improve from 0.99475
416/416 ━━━━━━━━━━━━━━━━━━━━ 78s 186ms/step - accuracy: 0.9825 - auc: 0.9979 - loss: 0.3265 - val_accuracy: 0.9637 - val_auc: 0.9945 - val_loss: 0.3425 - learning_rate: 1.8750e-05
Epoch 57/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9829 - auc: 0.9984 - loss: 0.3258
Epoch 57: ReduceLROnPlateau reducing learning rate to 9.375000445288606e-06.

Epoch 57: val_auc improved from 0.99475 to 0.99480, saving model to exp7_cnn_bilstm_best.keras

Epoch 57: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 86s 195ms/step - accuracy: 0.9830 - auc: 0.9983 - loss: 0.3261 - val_accuracy: 0.9659 - val_auc: 0.9948 - val_loss: 0.3398 - learning_rate: 1.8750e-05
Epoch 58/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 172ms/step - accuracy: 0.9823 - auc: 0.9979 - loss: 0.3257
Epoch 58: val_auc did not improve from 0.99480
416/416 ━━━━━━━━━━━━━━━━━━━━ 83s 197ms/step - accuracy: 0.9819 - auc: 0.9980 - loss: 0.3263 - val_accuracy: 0.9642 - val_auc: 0.9946 - val_loss: 0.3405 - learning_rate: 9.3750e-06
Epoch 59/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9817 - auc: 0.9985 - loss: 0.3257
Epoch 59: val_auc did not improve from 0.99480
416/416 ━━━━━━━━━━━━━━━━━━━━ 77s 185ms/step - accuracy: 0.9830 - auc: 0.9985 - loss: 0.3251 - val_accuracy: 0.9636 - val_auc: 0.9946 - val_loss: 0.3414 - learning_rate: 9.3750e-06
Epoch 60/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9825 - auc: 0.9984 - loss: 0.3253
Epoch 60: val_auc did not improve from 0.99480
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 185ms/step - accuracy: 0.9821 - auc: 0.9983 - loss: 0.3258 - val_accuracy: 0.9659 - val_auc: 0.9944 - val_loss: 0.3404 - learning_rate: 9.3750e-06
Epoch 61/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9840 - auc: 0.9986 - loss: 0.3234
Epoch 61: ReduceLROnPlateau reducing learning rate to 4.687500222644303e-06.

Epoch 61: val_auc did not improve from 0.99480
416/416 ━━━━━━━━━━━━━━━━━━━━ 86s 195ms/step - accuracy: 0.9833 - auc: 0.9985 - loss: 0.3243 - val_accuracy: 0.9633 - val_auc: 0.9947 - val_loss: 0.3420 - learning_rate: 9.3750e-06
Epoch 62/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9851 - auc: 0.9982 - loss: 0.3238
Epoch 62: val_auc did not improve from 0.99480
416/416 ━━━━━━━━━━━━━━━━━━━━ 78s 185ms/step - accuracy: 0.9836 - auc: 0.9982 - loss: 0.3244 - val_accuracy: 0.9653 - val_auc: 0.9946 - val_loss: 0.3404 - learning_rate: 4.6875e-06
Epoch 63/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9819 - auc: 0.9987 - loss: 0.3245
Epoch 63: val_auc did not improve from 0.99480
416/416 ━━━━━━━━━━━━━━━━━━━━ 86s 195ms/step - accuracy: 0.9823 - auc: 0.9987 - loss: 0.3241 - val_accuracy: 0.9656 - val_auc: 0.9947 - val_loss: 0.3400 - learning_rate: 4.6875e-06
Epoch 64/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9825 - auc: 0.9982 - loss: 0.3245
Epoch 64: val_auc did not improve from 0.99480
416/416 ━━━━━━━━━━━━━━━━━━━━ 78s 185ms/step - accuracy: 0.9831 - auc: 0.9983 - loss: 0.3247 - val_accuracy: 0.9640 - val_auc: 0.9946 - val_loss: 0.3411 - learning_rate: 4.6875e-06
Epoch 65/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 169ms/step - accuracy: 0.9838 - auc: 0.9984 - loss: 0.3242
Epoch 65: ReduceLROnPlateau reducing learning rate to 2.3437501113221515e-06.

Epoch 65: val_auc did not improve from 0.99480
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 185ms/step - accuracy: 0.9827 - auc: 0.9982 - loss: 0.3253 - val_accuracy: 0.9648 - val_auc: 0.9947 - val_loss: 0.3405 - learning_rate: 4.6875e-06
Epoch 66/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9845 - auc: 0.9987 - loss: 0.3227
Epoch 66: val_auc did not improve from 0.99480
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 186ms/step - accuracy: 0.9838 - auc: 0.9986 - loss: 0.3234 - val_accuracy: 0.9651 - val_auc: 0.9948 - val_loss: 0.3399 - learning_rate: 2.3438e-06
Epoch 67/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 169ms/step - accuracy: 0.9834 - auc: 0.9985 - loss: 0.3232
Epoch 67: val_auc did not improve from 0.99480
416/416 ━━━━━━━━━━━━━━━━━━━━ 86s 194ms/step - accuracy: 0.9832 - auc: 0.9986 - loss: 0.3240 - val_accuracy: 0.9663 - val_auc: 0.9947 - val_loss: 0.3391 - learning_rate: 2.3438e-06
Epoch 68/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9850 - auc: 0.9988 - loss: 0.3222
Epoch 68: val_auc did not improve from 0.99480
416/416 ━━━━━━━━━━━━━━━━━━━━ 78s 186ms/step - accuracy: 0.9839 - auc: 0.9987 - loss: 0.3234 - val_accuracy: 0.9656 - val_auc: 0.9948 - val_loss: 0.3397 - learning_rate: 2.3438e-06
Epoch 69/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 168ms/step - accuracy: 0.9839 - auc: 0.9986 - loss: 0.3230
Epoch 69: val_auc improved from 0.99480 to 0.99482, saving model to exp7_cnn_bilstm_best.keras

Epoch 69: finished saving model to exp7_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 76s 184ms/step - accuracy: 0.9833 - auc: 0.9985 - loss: 0.3237 - val_accuracy: 0.9665 - val_auc: 0.9948 - val_loss: 0.3393 - learning_rate: 2.3438e-06
Epoch 70/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 169ms/step - accuracy: 0.9836 - auc: 0.9985 - loss: 0.3239
Epoch 70: val_auc did not improve from 0.99482
416/416 ━━━━━━━━━━━━━━━━━━━━ 86s 194ms/step - accuracy: 0.9837 - auc: 0.9985 - loss: 0.3238 - val_accuracy: 0.9657 - val_auc: 0.9947 - val_loss: 0.3399 - learning_rate: 2.3438e-06
Epoch 71/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 169ms/step - accuracy: 0.9831 - auc: 0.9983 - loss: 0.3237
Epoch 71: val_auc did not improve from 0.99482
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 194ms/step - accuracy: 0.9827 - auc: 0.9983 - loss: 0.3248 - val_accuracy: 0.9663 - val_auc: 0.9948 - val_loss: 0.3395 - learning_rate: 2.3438e-06
Epoch 72/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9847 - auc: 0.9986 - loss: 0.3230
Epoch 72: val_auc did not improve from 0.99482
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 195ms/step - accuracy: 0.9830 - auc: 0.9984 - loss: 0.3241 - val_accuracy: 0.9654 - val_auc: 0.9948 - val_loss: 0.3395 - learning_rate: 2.3438e-06
Epoch 73/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 168ms/step - accuracy: 0.9832 - auc: 0.9987 - loss: 0.3231
Epoch 73: ReduceLROnPlateau reducing learning rate to 1.1718750556610757e-06.

Epoch 73: val_auc did not improve from 0.99482
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 193ms/step - accuracy: 0.9828 - auc: 0.9985 - loss: 0.3237 - val_accuracy: 0.9654 - val_auc: 0.9945 - val_loss: 0.3405 - learning_rate: 2.3438e-06
Epoch 74/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 169ms/step - accuracy: 0.9838 - auc: 0.9985 - loss: 0.3232
Epoch 74: val_auc did not improve from 0.99482
416/416 ━━━━━━━━━━━━━━━━━━━━ 78s 184ms/step - accuracy: 0.9836 - auc: 0.9985 - loss: 0.3232 - val_accuracy: 0.9653 - val_auc: 0.9947 - val_loss: 0.3399 - learning_rate: 1.1719e-06
Epoch 75/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 169ms/step - accuracy: 0.9851 - auc: 0.9986 - loss: 0.3221
Epoch 75: val_auc did not improve from 0.99482
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 184ms/step - accuracy: 0.9838 - auc: 0.9985 - loss: 0.3236 - val_accuracy: 0.9656 - val_auc: 0.9947 - val_loss: 0.3397 - learning_rate: 1.1719e-06
Epoch 76/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 169ms/step - accuracy: 0.9824 - auc: 0.9985 - loss: 0.3240
Epoch 76: val_auc did not improve from 0.99482
416/416 ━━━━━━━━━━━━━━━━━━━━ 86s 194ms/step - accuracy: 0.9836 - auc: 0.9984 - loss: 0.3240 - val_accuracy: 0.9654 - val_auc: 0.9947 - val_loss: 0.3398 - learning_rate: 1.1719e-06
Epoch 77/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 168ms/step - accuracy: 0.9853 - auc: 0.9987 - loss: 0.3223
Epoch 77: ReduceLROnPlateau reducing learning rate to 1e-06.

Epoch 77: val_auc did not improve from 0.99482
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 193ms/step - accuracy: 0.9844 - auc: 0.9985 - loss: 0.3232 - val_accuracy: 0.9657 - val_auc: 0.9946 - val_loss: 0.3398 - learning_rate: 1.1719e-06
Epoch 78/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 167ms/step - accuracy: 0.9856 - auc: 0.9985 - loss: 0.3223
Epoch 78: val_auc did not improve from 0.99482
416/416 ━━━━━━━━━━━━━━━━━━━━ 76s 182ms/step - accuracy: 0.9855 - auc: 0.9985 - loss: 0.3225 - val_accuracy: 0.9653 - val_auc: 0.9946 - val_loss: 0.3396 - learning_rate: 1.0000e-06
Epoch 79/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9817 - auc: 0.9984 - loss: 0.3245
Epoch 79: val_auc did not improve from 0.99482
416/416 ━━━━━━━━━━━━━━━━━━━━ 87s 195ms/step - accuracy: 0.9826 - auc: 0.9984 - loss: 0.3245 - val_accuracy: 0.9654 - val_auc: 0.9947 - val_loss: 0.3398 - learning_rate: 1.0000e-06
Epoch 80/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 167ms/step - accuracy: 0.9849 - auc: 0.9988 - loss: 0.3224
Epoch 80: val_auc did not improve from 0.99482
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 192ms/step - accuracy: 0.9842 - auc: 0.9986 - loss: 0.3231 - val_accuracy: 0.9665 - val_auc: 0.9947 - val_loss: 0.3396 - learning_rate: 1.0000e-06
Epoch 81/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 169ms/step - accuracy: 0.9841 - auc: 0.9988 - loss: 0.3227
Epoch 81: val_auc did not improve from 0.99482
416/416 ━━━━━━━━━━━━━━━━━━━━ 79s 184ms/step - accuracy: 0.9834 - auc: 0.9986 - loss: 0.3234 - val_accuracy: 0.9663 - val_auc: 0.9947 - val_loss: 0.3393 - learning_rate: 1.0000e-06
Epoch 82/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 167ms/step - accuracy: 0.9847 - auc: 0.9984 - loss: 0.3231
Epoch 82: val_auc did not improve from 0.99482
416/416 ━━━━━━━━━━━━━━━━━━━━ 85s 191ms/step - accuracy: 0.9849 - auc: 0.9986 - loss: 0.3228 - val_accuracy: 0.9663 - val_auc: 0.9947 - val_loss: 0.3392 - learning_rate: 1.0000e-06
Epoch 83/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 171ms/step - accuracy: 0.9833 - auc: 0.9987 - loss: 0.3233
Epoch 83: val_auc did not improve from 0.99482
416/416 ━━━━━━━━━━━━━━━━━━━━ 84s 196ms/step - accuracy: 0.9840 - auc: 0.9987 - loss: 0.3227 - val_accuracy: 0.9669 - val_auc: 0.9948 - val_loss: 0.3392 - learning_rate: 1.0000e-06
Epoch 84/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 169ms/step - accuracy: 0.9834 - auc: 0.9979 - loss: 0.3246
Epoch 84: val_auc did not improve from 0.99482
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 193ms/step - accuracy: 0.9833 - auc: 0.9983 - loss: 0.3240 - val_accuracy: 0.9666 - val_auc: 0.9947 - val_loss: 0.3393 - learning_rate: 1.0000e-06
Epoch 84: early stopping
Restoring model weights from the end of the best epoch: 69.
"""

log_exp8 = r"""
Epoch 1/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 147ms/step - accuracy: 0.5687 - auc: 0.5992 - loss: 0.7684
Epoch 1: val_auc improved from None to 0.75575, saving model to exp8_cnn_bilstm_best.keras

Epoch 1: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 74s 162ms/step - accuracy: 0.6029 - auc: 0.6491 - loss: 0.7489 - val_accuracy: 0.6886 - val_auc: 0.7557 - val_loss: 0.6830 - learning_rate: 3.0000e-04
Epoch 2/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 145ms/step - accuracy: 0.6763 - auc: 0.7351 - loss: 0.6961
Epoch 2: val_auc improved from 0.75575 to 0.82186, saving model to exp8_cnn_bilstm_best.keras

Epoch 2: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 80s 157ms/step - accuracy: 0.6811 - auc: 0.7460 - loss: 0.6854 - val_accuracy: 0.7383 - val_auc: 0.8219 - val_loss: 0.6163 - learning_rate: 3.0000e-04
Epoch 3/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 147ms/step - accuracy: 0.7166 - auc: 0.7965 - loss: 0.6383
Epoch 3: val_auc improved from 0.82186 to 0.84508, saving model to exp8_cnn_bilstm_best.keras

Epoch 3: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 67s 160ms/step - accuracy: 0.7208 - auc: 0.8015 - loss: 0.6317 - val_accuracy: 0.7520 - val_auc: 0.8451 - val_loss: 0.5898 - learning_rate: 3.0000e-04
Epoch 4/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 146ms/step - accuracy: 0.7381 - auc: 0.8256 - loss: 0.6039
Epoch 4: val_auc improved from 0.84508 to 0.87518, saving model to exp8_cnn_bilstm_best.keras

Epoch 4: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 66s 159ms/step - accuracy: 0.7466 - auc: 0.8332 - loss: 0.5948 - val_accuracy: 0.7799 - val_auc: 0.8752 - val_loss: 0.5438 - learning_rate: 3.0000e-04
Epoch 5/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 148ms/step - accuracy: 0.7696 - auc: 0.8562 - loss: 0.5655
Epoch 5: val_auc improved from 0.87518 to 0.89508, saving model to exp8_cnn_bilstm_best.keras

Epoch 5: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 83s 161ms/step - accuracy: 0.7723 - auc: 0.8583 - loss: 0.5616 - val_accuracy: 0.7890 - val_auc: 0.8951 - val_loss: 0.5211 - learning_rate: 3.0000e-04
Epoch 6/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 153ms/step - accuracy: 0.7838 - auc: 0.8710 - loss: 0.5424
Epoch 6: val_auc improved from 0.89508 to 0.91282, saving model to exp8_cnn_bilstm_best.keras

Epoch 6: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 84s 167ms/step - accuracy: 0.7889 - auc: 0.8763 - loss: 0.5349 - val_accuracy: 0.8204 - val_auc: 0.9128 - val_loss: 0.4845 - learning_rate: 3.0000e-04
Epoch 7/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.8019 - auc: 0.8907 - loss: 0.5140
Epoch 7: val_auc improved from 0.91282 to 0.92134, saving model to exp8_cnn_bilstm_best.keras

Epoch 7: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 71s 170ms/step - accuracy: 0.8031 - auc: 0.8909 - loss: 0.5120 - val_accuracy: 0.8344 - val_auc: 0.9213 - val_loss: 0.4651 - learning_rate: 3.0000e-04
Epoch 8/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 156ms/step - accuracy: 0.8153 - auc: 0.9022 - loss: 0.4950
Epoch 8: val_auc improved from 0.92134 to 0.93690, saving model to exp8_cnn_bilstm_best.keras

Epoch 8: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 87s 181ms/step - accuracy: 0.8166 - auc: 0.9050 - loss: 0.4899 - val_accuracy: 0.8472 - val_auc: 0.9369 - val_loss: 0.4352 - learning_rate: 3.0000e-04
Epoch 9/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.8299 - auc: 0.9158 - loss: 0.4708
Epoch 9: val_auc improved from 0.93690 to 0.94479, saving model to exp8_cnn_bilstm_best.keras

Epoch 9: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 179ms/step - accuracy: 0.8315 - auc: 0.9184 - loss: 0.4666 - val_accuracy: 0.8544 - val_auc: 0.9448 - val_loss: 0.4206 - learning_rate: 3.0000e-04
Epoch 10/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.8437 - auc: 0.9266 - loss: 0.4511
Epoch 10: val_auc improved from 0.94479 to 0.94751, saving model to exp8_cnn_bilstm_best.keras

Epoch 10: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 78s 170ms/step - accuracy: 0.8458 - auc: 0.9277 - loss: 0.4488 - val_accuracy: 0.8496 - val_auc: 0.9475 - val_loss: 0.4367 - learning_rate: 3.0000e-04
Epoch 11/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 159ms/step - accuracy: 0.8483 - auc: 0.9328 - loss: 0.4374
Epoch 11: val_auc improved from 0.94751 to 0.95664, saving model to exp8_cnn_bilstm_best.keras

Epoch 11: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 84s 174ms/step - accuracy: 0.8516 - auc: 0.9348 - loss: 0.4338 - val_accuracy: 0.8718 - val_auc: 0.9566 - val_loss: 0.3964 - learning_rate: 3.0000e-04
Epoch 12/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 159ms/step - accuracy: 0.8639 - auc: 0.9421 - loss: 0.4181
Epoch 12: val_auc improved from 0.95664 to 0.96383, saving model to exp8_cnn_bilstm_best.keras

Epoch 12: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 175ms/step - accuracy: 0.8650 - auc: 0.9426 - loss: 0.4171 - val_accuracy: 0.8884 - val_auc: 0.9638 - val_loss: 0.3708 - learning_rate: 3.0000e-04
Epoch 13/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 162ms/step - accuracy: 0.8685 - auc: 0.9457 - loss: 0.4091
Epoch 13: val_auc improved from 0.96383 to 0.96722, saving model to exp8_cnn_bilstm_best.keras

Epoch 13: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 78s 188ms/step - accuracy: 0.8705 - auc: 0.9471 - loss: 0.4058 - val_accuracy: 0.8982 - val_auc: 0.9672 - val_loss: 0.3582 - learning_rate: 3.0000e-04
Epoch 14/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 166ms/step - accuracy: 0.8774 - auc: 0.9523 - loss: 0.3932
Epoch 14: val_auc improved from 0.96722 to 0.96845, saving model to exp8_cnn_bilstm_best.keras

Epoch 14: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 80s 182ms/step - accuracy: 0.8773 - auc: 0.9518 - loss: 0.3944 - val_accuracy: 0.8985 - val_auc: 0.9684 - val_loss: 0.3528 - learning_rate: 3.0000e-04
Epoch 15/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 162ms/step - accuracy: 0.8835 - auc: 0.9567 - loss: 0.3817
Epoch 15: val_auc improved from 0.96845 to 0.97246, saving model to exp8_cnn_bilstm_best.keras

Epoch 15: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 80s 178ms/step - accuracy: 0.8830 - auc: 0.9558 - loss: 0.3837 - val_accuracy: 0.9067 - val_auc: 0.9725 - val_loss: 0.3424 - learning_rate: 3.0000e-04
Epoch 16/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 165ms/step - accuracy: 0.8924 - auc: 0.9605 - loss: 0.3712
Epoch 16: val_auc improved from 0.97246 to 0.97627, saving model to exp8_cnn_bilstm_best.keras

Epoch 16: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 84s 182ms/step - accuracy: 0.8930 - auc: 0.9607 - loss: 0.3704 - val_accuracy: 0.9144 - val_auc: 0.9763 - val_loss: 0.3305 - learning_rate: 3.0000e-04
Epoch 17/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 166ms/step - accuracy: 0.8992 - auc: 0.9645 - loss: 0.3597
Epoch 17: val_auc did not improve from 0.97627
416/416 ━━━━━━━━━━━━━━━━━━━━ 86s 191ms/step - accuracy: 0.8956 - auc: 0.9626 - loss: 0.3642 - val_accuracy: 0.9054 - val_auc: 0.9731 - val_loss: 0.3367 - learning_rate: 3.0000e-04
Epoch 18/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 167ms/step - accuracy: 0.9001 - auc: 0.9666 - loss: 0.3536
Epoch 18: val_auc did not improve from 0.97627
416/416 ━━━━━━━━━━━━━━━━━━━━ 80s 192ms/step - accuracy: 0.9029 - auc: 0.9669 - loss: 0.3520 - val_accuracy: 0.9110 - val_auc: 0.9754 - val_loss: 0.3327 - learning_rate: 3.0000e-04
Epoch 19/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9027 - auc: 0.9682 - loss: 0.3475
Epoch 19: val_auc improved from 0.97627 to 0.97957, saving model to exp8_cnn_bilstm_best.keras

Epoch 19: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 83s 195ms/step - accuracy: 0.9038 - auc: 0.9679 - loss: 0.3478 - val_accuracy: 0.9198 - val_auc: 0.9796 - val_loss: 0.3135 - learning_rate: 3.0000e-04
Epoch 20/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.9109 - auc: 0.9711 - loss: 0.3380
Epoch 20: val_auc improved from 0.97957 to 0.98025, saving model to exp8_cnn_bilstm_best.keras

Epoch 20: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 195ms/step - accuracy: 0.9108 - auc: 0.9705 - loss: 0.3392 - val_accuracy: 0.9194 - val_auc: 0.9802 - val_loss: 0.3158 - learning_rate: 3.0000e-04
Epoch 21/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 173ms/step - accuracy: 0.9124 - auc: 0.9724 - loss: 0.3332
Epoch 21: val_auc improved from 0.98025 to 0.98312, saving model to exp8_cnn_bilstm_best.keras

Epoch 21: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 192ms/step - accuracy: 0.9135 - auc: 0.9731 - loss: 0.3310 - val_accuracy: 0.9275 - val_auc: 0.9831 - val_loss: 0.2967 - learning_rate: 3.0000e-04
Epoch 22/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 174ms/step - accuracy: 0.9183 - auc: 0.9752 - loss: 0.3233
Epoch 22: val_auc improved from 0.98312 to 0.98341, saving model to exp8_cnn_bilstm_best.keras

Epoch 22: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 84s 198ms/step - accuracy: 0.9158 - auc: 0.9742 - loss: 0.3262 - val_accuracy: 0.9311 - val_auc: 0.9834 - val_loss: 0.2951 - learning_rate: 3.0000e-04
Epoch 23/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 178ms/step - accuracy: 0.9192 - auc: 0.9757 - loss: 0.3214
Epoch 23: val_auc improved from 0.98341 to 0.98557, saving model to exp8_cnn_bilstm_best.keras

Epoch 23: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 144s 203ms/step - accuracy: 0.9181 - auc: 0.9756 - loss: 0.3211 - val_accuracy: 0.9322 - val_auc: 0.9856 - val_loss: 0.2884 - learning_rate: 3.0000e-04
Epoch 24/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 179ms/step - accuracy: 0.9227 - auc: 0.9764 - loss: 0.3170
Epoch 24: val_auc did not improve from 0.98557
416/416 ━━━━━━━━━━━━━━━━━━━━ 143s 204ms/step - accuracy: 0.9228 - auc: 0.9771 - loss: 0.3152 - val_accuracy: 0.9340 - val_auc: 0.9844 - val_loss: 0.2911 - learning_rate: 3.0000e-04
Epoch 25/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 178ms/step - accuracy: 0.9233 - auc: 0.9783 - loss: 0.3104
Epoch 25: val_auc improved from 0.98557 to 0.98568, saving model to exp8_cnn_bilstm_best.keras

Epoch 25: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 142s 203ms/step - accuracy: 0.9279 - auc: 0.9797 - loss: 0.3056 - val_accuracy: 0.9365 - val_auc: 0.9857 - val_loss: 0.2839 - learning_rate: 3.0000e-04
Epoch 26/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 182ms/step - accuracy: 0.9273 - auc: 0.9786 - loss: 0.3079
Epoch 26: val_auc improved from 0.98568 to 0.98642, saving model to exp8_cnn_bilstm_best.keras

Epoch 26: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 143s 205ms/step - accuracy: 0.9263 - auc: 0.9788 - loss: 0.3077 - val_accuracy: 0.9407 - val_auc: 0.9864 - val_loss: 0.2792 - learning_rate: 3.0000e-04
Epoch 27/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 190ms/step - accuracy: 0.9294 - auc: 0.9799 - loss: 0.3037
Epoch 27: val_auc did not improve from 0.98642
416/416 ━━━━━━━━━━━━━━━━━━━━ 89s 213ms/step - accuracy: 0.9304 - auc: 0.9805 - loss: 0.3013 - val_accuracy: 0.9314 - val_auc: 0.9856 - val_loss: 0.2887 - learning_rate: 3.0000e-04
Epoch 28/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 187ms/step - accuracy: 0.9346 - auc: 0.9840 - loss: 0.2889
Epoch 28: val_auc improved from 0.98642 to 0.98730, saving model to exp8_cnn_bilstm_best.keras

Epoch 28: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 88s 211ms/step - accuracy: 0.9346 - auc: 0.9831 - loss: 0.2913 - val_accuracy: 0.9422 - val_auc: 0.9873 - val_loss: 0.2739 - learning_rate: 3.0000e-04
Epoch 29/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 189ms/step - accuracy: 0.9376 - auc: 0.9848 - loss: 0.2856
Epoch 29: val_auc improved from 0.98730 to 0.98804, saving model to exp8_cnn_bilstm_best.keras

Epoch 29: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 143s 213ms/step - accuracy: 0.9357 - auc: 0.9837 - loss: 0.2886 - val_accuracy: 0.9335 - val_auc: 0.9880 - val_loss: 0.2819 - learning_rate: 3.0000e-04
Epoch 30/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 188ms/step - accuracy: 0.9345 - auc: 0.9831 - loss: 0.2902
Epoch 30: val_auc did not improve from 0.98804
416/416 ━━━━━━━━━━━━━━━━━━━━ 88s 212ms/step - accuracy: 0.9358 - auc: 0.9841 - loss: 0.2866 - val_accuracy: 0.9407 - val_auc: 0.9878 - val_loss: 0.2715 - learning_rate: 3.0000e-04
Epoch 31/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 199ms/step - accuracy: 0.9394 - auc: 0.9851 - loss: 0.2810
Epoch 31: val_auc improved from 0.98804 to 0.98845, saving model to exp8_cnn_bilstm_best.keras

Epoch 31: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 93s 224ms/step - accuracy: 0.9391 - auc: 0.9849 - loss: 0.2822 - val_accuracy: 0.9451 - val_auc: 0.9884 - val_loss: 0.2693 - learning_rate: 3.0000e-04
Epoch 32/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 205ms/step - accuracy: 0.9430 - auc: 0.9871 - loss: 0.2746
Epoch 32: val_auc improved from 0.98845 to 0.98883, saving model to exp8_cnn_bilstm_best.keras

Epoch 32: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 97s 232ms/step - accuracy: 0.9403 - auc: 0.9863 - loss: 0.2774 - val_accuracy: 0.9451 - val_auc: 0.9888 - val_loss: 0.2667 - learning_rate: 3.0000e-04
Epoch 33/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 192ms/step - accuracy: 0.9405 - auc: 0.9857 - loss: 0.2788
Epoch 33: val_auc improved from 0.98883 to 0.99020, saving model to exp8_cnn_bilstm_best.keras

Epoch 33: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 91s 218ms/step - accuracy: 0.9402 - auc: 0.9857 - loss: 0.2786 - val_accuracy: 0.9469 - val_auc: 0.9902 - val_loss: 0.2635 - learning_rate: 3.0000e-04
Epoch 34/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 204ms/step - accuracy: 0.9454 - auc: 0.9876 - loss: 0.2702
Epoch 34: val_auc did not improve from 0.99020
416/416 ━━━━━━━━━━━━━━━━━━━━ 96s 231ms/step - accuracy: 0.9436 - auc: 0.9871 - loss: 0.2722 - val_accuracy: 0.9475 - val_auc: 0.9902 - val_loss: 0.2579 - learning_rate: 3.0000e-04
Epoch 35/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 197ms/step - accuracy: 0.9487 - auc: 0.9888 - loss: 0.2648
Epoch 35: val_auc improved from 0.99020 to 0.99066, saving model to exp8_cnn_bilstm_best.keras

Epoch 35: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 93s 224ms/step - accuracy: 0.9466 - auc: 0.9878 - loss: 0.2686 - val_accuracy: 0.9499 - val_auc: 0.9907 - val_loss: 0.2539 - learning_rate: 3.0000e-04
Epoch 36/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 199ms/step - accuracy: 0.9445 - auc: 0.9875 - loss: 0.2689
Epoch 36: val_auc did not improve from 0.99066
416/416 ━━━━━━━━━━━━━━━━━━━━ 92s 221ms/step - accuracy: 0.9477 - auc: 0.9882 - loss: 0.2652 - val_accuracy: 0.9457 - val_auc: 0.9894 - val_loss: 0.2646 - learning_rate: 3.0000e-04
Epoch 37/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 201ms/step - accuracy: 0.9478 - auc: 0.9887 - loss: 0.2633
Epoch 37: val_auc improved from 0.99066 to 0.99104, saving model to exp8_cnn_bilstm_best.keras

Epoch 37: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 94s 227ms/step - accuracy: 0.9477 - auc: 0.9884 - loss: 0.2638 - val_accuracy: 0.9511 - val_auc: 0.9910 - val_loss: 0.2522 - learning_rate: 3.0000e-04
Epoch 38/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 203ms/step - accuracy: 0.9526 - auc: 0.9897 - loss: 0.2573
Epoch 38: val_auc did not improve from 0.99104
416/416 ━━━━━━━━━━━━━━━━━━━━ 95s 228ms/step - accuracy: 0.9510 - auc: 0.9897 - loss: 0.2582 - val_accuracy: 0.9481 - val_auc: 0.9910 - val_loss: 0.2565 - learning_rate: 3.0000e-04
Epoch 39/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 201ms/step - accuracy: 0.9540 - auc: 0.9907 - loss: 0.2530
Epoch 39: val_auc improved from 0.99104 to 0.99107, saving model to exp8_cnn_bilstm_best.keras

Epoch 39: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 95s 227ms/step - accuracy: 0.9543 - auc: 0.9905 - loss: 0.2532 - val_accuracy: 0.9523 - val_auc: 0.9911 - val_loss: 0.2519 - learning_rate: 3.0000e-04
Epoch 40/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 200ms/step - accuracy: 0.9510 - auc: 0.9903 - loss: 0.2551
Epoch 40: val_auc did not improve from 0.99107
416/416 ━━━━━━━━━━━━━━━━━━━━ 94s 225ms/step - accuracy: 0.9522 - auc: 0.9906 - loss: 0.2538 - val_accuracy: 0.9478 - val_auc: 0.9906 - val_loss: 0.2632 - learning_rate: 3.0000e-04
Epoch 41/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 181ms/step - accuracy: 0.9528 - auc: 0.9902 - loss: 0.2547
Epoch 41: val_auc improved from 0.99107 to 0.99273, saving model to exp8_cnn_bilstm_best.keras

Epoch 41: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 195ms/step - accuracy: 0.9542 - auc: 0.9903 - loss: 0.2535 - val_accuracy: 0.9520 - val_auc: 0.9927 - val_loss: 0.2513 - learning_rate: 3.0000e-04
Epoch 42/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 204ms/step - accuracy: 0.9561 - auc: 0.9916 - loss: 0.2470
Epoch 42: val_auc improved from 0.99273 to 0.99280, saving model to exp8_cnn_bilstm_best.keras

Epoch 42: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 101s 242ms/step - accuracy: 0.9540 - auc: 0.9911 - loss: 0.2500 - val_accuracy: 0.9580 - val_auc: 0.9928 - val_loss: 0.2373 - learning_rate: 3.0000e-04
Epoch 43/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 326ms/step - accuracy: 0.9574 - auc: 0.9915 - loss: 0.2471
Epoch 43: val_auc did not improve from 0.99280
416/416 ━━━━━━━━━━━━━━━━━━━━ 150s 361ms/step - accuracy: 0.9571 - auc: 0.9913 - loss: 0.2476 - val_accuracy: 0.9559 - val_auc: 0.9922 - val_loss: 0.2465 - learning_rate: 3.0000e-04
Epoch 44/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 322ms/step - accuracy: 0.9576 - auc: 0.9924 - loss: 0.2428
Epoch 44: val_auc did not improve from 0.99280
416/416 ━━━━━━━━━━━━━━━━━━━━ 148s 356ms/step - accuracy: 0.9567 - auc: 0.9919 - loss: 0.2452 - val_accuracy: 0.9600 - val_auc: 0.9924 - val_loss: 0.2414 - learning_rate: 3.0000e-04
Epoch 45/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 322ms/step - accuracy: 0.9562 - auc: 0.9919 - loss: 0.2450
Epoch 45: val_auc did not improve from 0.99280
416/416 ━━━━━━━━━━━━━━━━━━━━ 149s 357ms/step - accuracy: 0.9582 - auc: 0.9919 - loss: 0.2439 - val_accuracy: 0.9588 - val_auc: 0.9917 - val_loss: 0.2413 - learning_rate: 3.0000e-04
Epoch 46/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 315ms/step - accuracy: 0.9598 - auc: 0.9929 - loss: 0.2398
Epoch 46: ReduceLROnPlateau reducing learning rate to 0.0001500000071246177.

Epoch 46: val_auc did not improve from 0.99280
416/416 ━━━━━━━━━━━━━━━━━━━━ 152s 365ms/step - accuracy: 0.9585 - auc: 0.9923 - loss: 0.2416 - val_accuracy: 0.9510 - val_auc: 0.9924 - val_loss: 0.2524 - learning_rate: 3.0000e-04
Epoch 47/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 287ms/step - accuracy: 0.9655 - auc: 0.9947 - loss: 0.2280
Epoch 47: val_auc improved from 0.99280 to 0.99381, saving model to exp8_cnn_bilstm_best.keras

Epoch 47: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 135s 325ms/step - accuracy: 0.9651 - auc: 0.9944 - loss: 0.2289 - val_accuracy: 0.9589 - val_auc: 0.9938 - val_loss: 0.2385 - learning_rate: 1.5000e-04
Epoch 48/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 322ms/step - accuracy: 0.9712 - auc: 0.9958 - loss: 0.2202
Epoch 48: val_auc did not improve from 0.99381
416/416 ━━━━━━━━━━━━━━━━━━━━ 148s 357ms/step - accuracy: 0.9712 - auc: 0.9959 - loss: 0.2197 - val_accuracy: 0.9633 - val_auc: 0.9934 - val_loss: 0.2329 - learning_rate: 1.5000e-04
Epoch 49/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 322ms/step - accuracy: 0.9692 - auc: 0.9958 - loss: 0.2205
Epoch 49: val_auc improved from 0.99381 to 0.99416, saving model to exp8_cnn_bilstm_best.keras

Epoch 49: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 148s 357ms/step - accuracy: 0.9693 - auc: 0.9954 - loss: 0.2217 - val_accuracy: 0.9621 - val_auc: 0.9942 - val_loss: 0.2297 - learning_rate: 1.5000e-04
Epoch 50/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 324ms/step - accuracy: 0.9732 - auc: 0.9963 - loss: 0.2157
Epoch 50: val_auc improved from 0.99416 to 0.99441, saving model to exp8_cnn_bilstm_best.keras

Epoch 50: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 150s 359ms/step - accuracy: 0.9710 - auc: 0.9959 - loss: 0.2178 - val_accuracy: 0.9642 - val_auc: 0.9944 - val_loss: 0.2233 - learning_rate: 1.5000e-04
Epoch 51/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 326ms/step - accuracy: 0.9716 - auc: 0.9962 - loss: 0.2157
Epoch 51: val_auc did not improve from 0.99441
416/416 ━━━━━━━━━━━━━━━━━━━━ 150s 361ms/step - accuracy: 0.9719 - auc: 0.9960 - loss: 0.2162 - val_accuracy: 0.9631 - val_auc: 0.9943 - val_loss: 0.2299 - learning_rate: 1.5000e-04
Epoch 52/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 326ms/step - accuracy: 0.9740 - auc: 0.9965 - loss: 0.2127
Epoch 52: val_auc did not improve from 0.99441
416/416 ━━━━━━━━━━━━━━━━━━━━ 150s 361ms/step - accuracy: 0.9719 - auc: 0.9959 - loss: 0.2160 - val_accuracy: 0.9630 - val_auc: 0.9940 - val_loss: 0.2277 - learning_rate: 1.5000e-04
Epoch 53/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 274ms/step - accuracy: 0.9715 - auc: 0.9958 - loss: 0.2159
Epoch 53: val_auc did not improve from 0.99441
416/416 ━━━━━━━━━━━━━━━━━━━━ 126s 303ms/step - accuracy: 0.9718 - auc: 0.9960 - loss: 0.2153 - val_accuracy: 0.9637 - val_auc: 0.9941 - val_loss: 0.2286 - learning_rate: 1.5000e-04
Epoch 54/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 306ms/step - accuracy: 0.9762 - auc: 0.9970 - loss: 0.2079
Epoch 54: val_auc improved from 0.99441 to 0.99480, saving model to exp8_cnn_bilstm_best.keras

Epoch 54: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 142s 341ms/step - accuracy: 0.9733 - auc: 0.9965 - loss: 0.2116 - val_accuracy: 0.9663 - val_auc: 0.9948 - val_loss: 0.2223 - learning_rate: 1.5000e-04
Epoch 55/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 325ms/step - accuracy: 0.9754 - auc: 0.9970 - loss: 0.2084
Epoch 55: val_auc did not improve from 0.99480
416/416 ━━━━━━━━━━━━━━━━━━━━ 150s 361ms/step - accuracy: 0.9741 - auc: 0.9968 - loss: 0.2098 - val_accuracy: 0.9693 - val_auc: 0.9945 - val_loss: 0.2215 - learning_rate: 1.5000e-04
Epoch 56/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 327ms/step - accuracy: 0.9764 - auc: 0.9970 - loss: 0.2069
Epoch 56: val_auc did not improve from 0.99480
416/416 ━━━━━━━━━━━━━━━━━━━━ 151s 362ms/step - accuracy: 0.9754 - auc: 0.9968 - loss: 0.2078 - val_accuracy: 0.9656 - val_auc: 0.9943 - val_loss: 0.2275 - learning_rate: 1.5000e-04
Epoch 57/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 325ms/step - accuracy: 0.9773 - auc: 0.9968 - loss: 0.2068
Epoch 57: val_auc did not improve from 0.99480
416/416 ━━━━━━━━━━━━━━━━━━━━ 149s 359ms/step - accuracy: 0.9754 - auc: 0.9967 - loss: 0.2088 - val_accuracy: 0.9675 - val_auc: 0.9947 - val_loss: 0.2196 - learning_rate: 1.5000e-04
Epoch 58/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 326ms/step - accuracy: 0.9765 - auc: 0.9968 - loss: 0.2071
Epoch 58: val_auc improved from 0.99480 to 0.99520, saving model to exp8_cnn_bilstm_best.keras

Epoch 58: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 150s 360ms/step - accuracy: 0.9749 - auc: 0.9966 - loss: 0.2082 - val_accuracy: 0.9689 - val_auc: 0.9952 - val_loss: 0.2156 - learning_rate: 1.5000e-04
Epoch 59/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 327ms/step - accuracy: 0.9760 - auc: 0.9973 - loss: 0.2042
Epoch 59: val_auc did not improve from 0.99520
416/416 ━━━━━━━━━━━━━━━━━━━━ 151s 362ms/step - accuracy: 0.9756 - auc: 0.9971 - loss: 0.2053 - val_accuracy: 0.9650 - val_auc: 0.9944 - val_loss: 0.2242 - learning_rate: 1.5000e-04
Epoch 60/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 310ms/step - accuracy: 0.9740 - auc: 0.9964 - loss: 0.2080
Epoch 60: val_auc did not improve from 0.99520
416/416 ━━━━━━━━━━━━━━━━━━━━ 140s 336ms/step - accuracy: 0.9750 - auc: 0.9967 - loss: 0.2064 - val_accuracy: 0.9648 - val_auc: 0.9950 - val_loss: 0.2203 - learning_rate: 1.5000e-04
Epoch 61/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 295ms/step - accuracy: 0.9759 - auc: 0.9972 - loss: 0.2043
Epoch 61: val_auc did not improve from 0.99520
416/416 ━━━━━━━━━━━━━━━━━━━━ 137s 330ms/step - accuracy: 0.9755 - auc: 0.9970 - loss: 0.2047 - val_accuracy: 0.9665 - val_auc: 0.9945 - val_loss: 0.2220 - learning_rate: 1.5000e-04
Epoch 62/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 328ms/step - accuracy: 0.9785 - auc: 0.9973 - loss: 0.2013
Epoch 62: val_auc did not improve from 0.99520
416/416 ━━━━━━━━━━━━━━━━━━━━ 151s 363ms/step - accuracy: 0.9772 - auc: 0.9971 - loss: 0.2030 - val_accuracy: 0.9671 - val_auc: 0.9948 - val_loss: 0.2177 - learning_rate: 1.5000e-04
Epoch 63/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 327ms/step - accuracy: 0.9783 - auc: 0.9976 - loss: 0.1993
Epoch 63: ReduceLROnPlateau reducing learning rate to 7.500000356230885e-05.

Epoch 63: val_auc did not improve from 0.99520
416/416 ━━━━━━━━━━━━━━━━━━━━ 150s 361ms/step - accuracy: 0.9776 - auc: 0.9975 - loss: 0.2000 - val_accuracy: 0.9665 - val_auc: 0.9950 - val_loss: 0.2192 - learning_rate: 1.5000e-04
Epoch 64/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 314ms/step - accuracy: 0.9797 - auc: 0.9976 - loss: 0.1958
Epoch 64: val_auc did not improve from 0.99520
416/416 ━━━━━━━━━━━━━━━━━━━━ 145s 350ms/step - accuracy: 0.9794 - auc: 0.9977 - loss: 0.1960 - val_accuracy: 0.9665 - val_auc: 0.9950 - val_loss: 0.2209 - learning_rate: 7.5000e-05
Epoch 65/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 273ms/step - accuracy: 0.9802 - auc: 0.9979 - loss: 0.1961
Epoch 65: val_auc improved from 0.99520 to 0.99526, saving model to exp8_cnn_bilstm_best.keras

Epoch 65: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 128s 308ms/step - accuracy: 0.9804 - auc: 0.9979 - loss: 0.1954 - val_accuracy: 0.9710 - val_auc: 0.9953 - val_loss: 0.2110 - learning_rate: 7.5000e-05
Epoch 66/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 327ms/step - accuracy: 0.9829 - auc: 0.9985 - loss: 0.1906
Epoch 66: val_auc did not improve from 0.99526
416/416 ━━━━━━━━━━━━━━━━━━━━ 151s 362ms/step - accuracy: 0.9813 - auc: 0.9983 - loss: 0.1920 - val_accuracy: 0.9677 - val_auc: 0.9951 - val_loss: 0.2152 - learning_rate: 7.5000e-05
Epoch 67/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 327ms/step - accuracy: 0.9827 - auc: 0.9984 - loss: 0.1904
Epoch 67: val_auc did not improve from 0.99526
416/416 ━━━━━━━━━━━━━━━━━━━━ 151s 362ms/step - accuracy: 0.9817 - auc: 0.9982 - loss: 0.1915 - val_accuracy: 0.9699 - val_auc: 0.9952 - val_loss: 0.2138 - learning_rate: 7.5000e-05
Epoch 68/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 328ms/step - accuracy: 0.9823 - auc: 0.9983 - loss: 0.1909
Epoch 68: val_auc improved from 0.99526 to 0.99535, saving model to exp8_cnn_bilstm_best.keras

Epoch 68: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 151s 363ms/step - accuracy: 0.9821 - auc: 0.9982 - loss: 0.1912 - val_accuracy: 0.9687 - val_auc: 0.9953 - val_loss: 0.2115 - learning_rate: 7.5000e-05
Epoch 69/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 248ms/step - accuracy: 0.9824 - auc: 0.9983 - loss: 0.1904
Epoch 69: val_auc improved from 0.99535 to 0.99585, saving model to exp8_cnn_bilstm_best.keras

Epoch 69: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 114s 274ms/step - accuracy: 0.9821 - auc: 0.9983 - loss: 0.1902 - val_accuracy: 0.9707 - val_auc: 0.9958 - val_loss: 0.2112 - learning_rate: 7.5000e-05
Epoch 70/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 201ms/step - accuracy: 0.9843 - auc: 0.9987 - loss: 0.1875
Epoch 70: val_auc did not improve from 0.99585
416/416 ━━━━━━━━━━━━━━━━━━━━ 95s 228ms/step - accuracy: 0.9836 - auc: 0.9986 - loss: 0.1881 - val_accuracy: 0.9663 - val_auc: 0.9955 - val_loss: 0.2149 - learning_rate: 7.5000e-05
Epoch 71/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 218ms/step - accuracy: 0.9849 - auc: 0.9984 - loss: 0.1866
Epoch 71: val_auc did not improve from 0.99585
416/416 ━━━━━━━━━━━━━━━━━━━━ 101s 244ms/step - accuracy: 0.9840 - auc: 0.9984 - loss: 0.1878 - val_accuracy: 0.9693 - val_auc: 0.9957 - val_loss: 0.2113 - learning_rate: 7.5000e-05
Epoch 72/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 216ms/step - accuracy: 0.9838 - auc: 0.9985 - loss: 0.1874
Epoch 72: val_auc did not improve from 0.99585
416/416 ━━━━━━━━━━━━━━━━━━━━ 101s 243ms/step - accuracy: 0.9833 - auc: 0.9984 - loss: 0.1881 - val_accuracy: 0.9681 - val_auc: 0.9957 - val_loss: 0.2107 - learning_rate: 7.5000e-05
Epoch 73/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 211ms/step - accuracy: 0.9823 - auc: 0.9984 - loss: 0.1892
Epoch 73: val_auc did not improve from 0.99585
416/416 ━━━━━━━━━━━━━━━━━━━━ 108s 260ms/step - accuracy: 0.9818 - auc: 0.9984 - loss: 0.1889 - val_accuracy: 0.9702 - val_auc: 0.9956 - val_loss: 0.2118 - learning_rate: 7.5000e-05
Epoch 74/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 214ms/step - accuracy: 0.9842 - auc: 0.9988 - loss: 0.1856
Epoch 74: ReduceLROnPlateau reducing learning rate to 3.7500001781154424e-05.

Epoch 74: val_auc did not improve from 0.99585
416/416 ━━━━━━━━━━━━━━━━━━━━ 143s 263ms/step - accuracy: 0.9835 - auc: 0.9985 - loss: 0.1864 - val_accuracy: 0.9714 - val_auc: 0.9955 - val_loss: 0.2073 - learning_rate: 7.5000e-05
Epoch 75/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 211ms/step - accuracy: 0.9840 - auc: 0.9985 - loss: 0.1857
Epoch 75: val_auc did not improve from 0.99585
416/416 ━━━━━━━━━━━━━━━━━━━━ 109s 261ms/step - accuracy: 0.9842 - auc: 0.9986 - loss: 0.1850 - val_accuracy: 0.9708 - val_auc: 0.9956 - val_loss: 0.2079 - learning_rate: 3.7500e-05
Epoch 76/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 214ms/step - accuracy: 0.9863 - auc: 0.9985 - loss: 0.1840
Epoch 76: val_auc did not improve from 0.99585
416/416 ━━━━━━━━━━━━━━━━━━━━ 134s 241ms/step - accuracy: 0.9856 - auc: 0.9987 - loss: 0.1838 - val_accuracy: 0.9696 - val_auc: 0.9957 - val_loss: 0.2085 - learning_rate: 3.7500e-05
Epoch 77/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 219ms/step - accuracy: 0.9855 - auc: 0.9990 - loss: 0.1811
Epoch 77: val_auc did not improve from 0.99585
416/416 ━━━━━━━━━━━━━━━━━━━━ 112s 269ms/step - accuracy: 0.9842 - auc: 0.9987 - loss: 0.1839 - val_accuracy: 0.9681 - val_auc: 0.9954 - val_loss: 0.2134 - learning_rate: 3.7500e-05
Epoch 78/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 217ms/step - accuracy: 0.9861 - auc: 0.9988 - loss: 0.1828
Epoch 78: val_auc improved from 0.99585 to 0.99592, saving model to exp8_cnn_bilstm_best.keras

Epoch 78: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 132s 245ms/step - accuracy: 0.9850 - auc: 0.9989 - loss: 0.1834 - val_accuracy: 0.9711 - val_auc: 0.9959 - val_loss: 0.2068 - learning_rate: 3.7500e-05
Epoch 79/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 210ms/step - accuracy: 0.9873 - auc: 0.9991 - loss: 0.1797
Epoch 79: ReduceLROnPlateau reducing learning rate to 1.8750000890577212e-05.

Epoch 79: val_auc improved from 0.99592 to 0.99594, saving model to exp8_cnn_bilstm_best.keras

Epoch 79: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 139s 238ms/step - accuracy: 0.9864 - auc: 0.9990 - loss: 0.1805 - val_accuracy: 0.9714 - val_auc: 0.9959 - val_loss: 0.2068 - learning_rate: 3.7500e-05
Epoch 80/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 210ms/step - accuracy: 0.9876 - auc: 0.9990 - loss: 0.1797
Epoch 80: val_auc improved from 0.99594 to 0.99611, saving model to exp8_cnn_bilstm_best.keras

Epoch 80: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 142s 238ms/step - accuracy: 0.9875 - auc: 0.9991 - loss: 0.1795 - val_accuracy: 0.9726 - val_auc: 0.9961 - val_loss: 0.2061 - learning_rate: 1.8750e-05
Epoch 81/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 213ms/step - accuracy: 0.9856 - auc: 0.9989 - loss: 0.1821
Epoch 81: val_auc improved from 0.99611 to 0.99611, saving model to exp8_cnn_bilstm_best.keras

Epoch 81: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 143s 241ms/step - accuracy: 0.9865 - auc: 0.9989 - loss: 0.1813 - val_accuracy: 0.9710 - val_auc: 0.9961 - val_loss: 0.2072 - learning_rate: 1.8750e-05
Epoch 82/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 213ms/step - accuracy: 0.9861 - auc: 0.9990 - loss: 0.1801
Epoch 82: val_auc did not improve from 0.99611
416/416 ━━━━━━━━━━━━━━━━━━━━ 151s 263ms/step - accuracy: 0.9861 - auc: 0.9989 - loss: 0.1806 - val_accuracy: 0.9714 - val_auc: 0.9961 - val_loss: 0.2068 - learning_rate: 1.8750e-05
Epoch 83/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 213ms/step - accuracy: 0.9878 - auc: 0.9991 - loss: 0.1787
Epoch 83: val_auc did not improve from 0.99611
416/416 ━━━━━━━━━━━━━━━━━━━━ 133s 241ms/step - accuracy: 0.9877 - auc: 0.9991 - loss: 0.1787 - val_accuracy: 0.9717 - val_auc: 0.9960 - val_loss: 0.2056 - learning_rate: 1.8750e-05
Epoch 84/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 219ms/step - accuracy: 0.9878 - auc: 0.9991 - loss: 0.1783
Epoch 84: val_auc did not improve from 0.99611
416/416 ━━━━━━━━━━━━━━━━━━━━ 153s 268ms/step - accuracy: 0.9876 - auc: 0.9990 - loss: 0.1787 - val_accuracy: 0.9726 - val_auc: 0.9961 - val_loss: 0.2056 - learning_rate: 1.8750e-05
Epoch 85/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 216ms/step - accuracy: 0.9871 - auc: 0.9988 - loss: 0.1804
Epoch 85: ReduceLROnPlateau reducing learning rate to 9.375000445288606e-06.

Epoch 85: val_auc did not improve from 0.99611
416/416 ━━━━━━━━━━━━━━━━━━━━ 110s 265ms/step - accuracy: 0.9865 - auc: 0.9990 - loss: 0.1798 - val_accuracy: 0.9710 - val_auc: 0.9960 - val_loss: 0.2054 - learning_rate: 1.8750e-05
Epoch 86/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 212ms/step - accuracy: 0.9868 - auc: 0.9993 - loss: 0.1775
Epoch 86: val_auc improved from 0.99611 to 0.99614, saving model to exp8_cnn_bilstm_best.keras

Epoch 86: finished saving model to exp8_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 140s 261ms/step - accuracy: 0.9876 - auc: 0.9993 - loss: 0.1775 - val_accuracy: 0.9725 - val_auc: 0.9961 - val_loss: 0.2047 - learning_rate: 9.3750e-06
Epoch 87/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 212ms/step - accuracy: 0.9873 - auc: 0.9990 - loss: 0.1780
Epoch 87: val_auc did not improve from 0.99614
416/416 ━━━━━━━━━━━━━━━━━━━━ 142s 261ms/step - accuracy: 0.9874 - auc: 0.9992 - loss: 0.1773 - val_accuracy: 0.9717 - val_auc: 0.9960 - val_loss: 0.2052 - learning_rate: 9.3750e-06
Epoch 88/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 211ms/step - accuracy: 0.9865 - auc: 0.9990 - loss: 0.1794
Epoch 88: val_auc did not improve from 0.99614
416/416 ━━━━━━━━━━━━━━━━━━━━ 142s 261ms/step - accuracy: 0.9867 - auc: 0.9989 - loss: 0.1796 - val_accuracy: 0.9723 - val_auc: 0.9960 - val_loss: 0.2050 - learning_rate: 9.3750e-06
Epoch 89/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 210ms/step - accuracy: 0.9888 - auc: 0.9992 - loss: 0.1763
Epoch 89: val_auc did not improve from 0.99614
416/416 ━━━━━━━━━━━━━━━━━━━━ 142s 260ms/step - accuracy: 0.9872 - auc: 0.9990 - loss: 0.1781 - val_accuracy: 0.9722 - val_auc: 0.9960 - val_loss: 0.2043 - learning_rate: 9.3750e-06
Epoch 90/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 213ms/step - accuracy: 0.9874 - auc: 0.9992 - loss: 0.1773
Epoch 90: ReduceLROnPlateau reducing learning rate to 4.687500222644303e-06.

Epoch 90: val_auc did not improve from 0.99614
416/416 ━━━━━━━━━━━━━━━━━━━━ 143s 263ms/step - accuracy: 0.9877 - auc: 0.9992 - loss: 0.1768 - val_accuracy: 0.9722 - val_auc: 0.9960 - val_loss: 0.2054 - learning_rate: 9.3750e-06
Epoch 91/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 211ms/step - accuracy: 0.9882 - auc: 0.9992 - loss: 0.1767
Epoch 91: val_auc did not improve from 0.99614
416/416 ━━━━━━━━━━━━━━━━━━━━ 141s 260ms/step - accuracy: 0.9877 - auc: 0.9992 - loss: 0.1768 - val_accuracy: 0.9726 - val_auc: 0.9960 - val_loss: 0.2050 - learning_rate: 4.6875e-06
Epoch 92/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 210ms/step - accuracy: 0.9880 - auc: 0.9992 - loss: 0.1776
Epoch 92: val_auc did not improve from 0.99614
416/416 ━━━━━━━━━━━━━━━━━━━━ 133s 238ms/step - accuracy: 0.9883 - auc: 0.9991 - loss: 0.1775 - val_accuracy: 0.9720 - val_auc: 0.9960 - val_loss: 0.2056 - learning_rate: 4.6875e-06
Epoch 93/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 210ms/step - accuracy: 0.9884 - auc: 0.9993 - loss: 0.1765
Epoch 93: val_auc did not improve from 0.99614
416/416 ━━━━━━━━━━━━━━━━━━━━ 108s 259ms/step - accuracy: 0.9880 - auc: 0.9992 - loss: 0.1768 - val_accuracy: 0.9723 - val_auc: 0.9961 - val_loss: 0.2045 - learning_rate: 4.6875e-06
Epoch 94/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 211ms/step - accuracy: 0.9886 - auc: 0.9993 - loss: 0.1768
Epoch 94: val_auc did not improve from 0.99614
416/416 ━━━━━━━━━━━━━━━━━━━━ 108s 260ms/step - accuracy: 0.9887 - auc: 0.9993 - loss: 0.1767 - val_accuracy: 0.9720 - val_auc: 0.9961 - val_loss: 0.2052 - learning_rate: 4.6875e-06
Epoch 95/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 209ms/step - accuracy: 0.9888 - auc: 0.9991 - loss: 0.1765
Epoch 95: ReduceLROnPlateau reducing learning rate to 2.3437501113221515e-06.

Epoch 95: val_auc did not improve from 0.99614
416/416 ━━━━━━━━━━━━━━━━━━━━ 141s 259ms/step - accuracy: 0.9883 - auc: 0.9992 - loss: 0.1770 - val_accuracy: 0.9717 - val_auc: 0.9961 - val_loss: 0.2052 - learning_rate: 4.6875e-06
Epoch 96/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 210ms/step - accuracy: 0.9892 - auc: 0.9994 - loss: 0.1759
Epoch 96: val_auc did not improve from 0.99614
416/416 ━━━━━━━━━━━━━━━━━━━━ 142s 260ms/step - accuracy: 0.9892 - auc: 0.9993 - loss: 0.1763 - val_accuracy: 0.9723 - val_auc: 0.9961 - val_loss: 0.2045 - learning_rate: 2.3438e-06
Epoch 96: early stopping
Restoring model weights from the end of the best epoch: 86.
"""


log_exp9 = r"""
Epoch 1/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 364ms/step - accuracy: 0.5508 - auc: 0.5735 - loss: 0.8719
Epoch 1: val_auc improved from None to 0.71006, saving model to exp9_cnn_bilstm_best.keras

Epoch 1: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 91s 397ms/step - accuracy: 0.5862 - auc: 0.6251 - loss: 0.8547 - val_accuracy: 0.6417 - val_auc: 0.7101 - val_loss: 0.8195 - learning_rate: 2.0000e-04
Epoch 2/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 367ms/step - accuracy: 0.6557 - auc: 0.7129 - loss: 0.8121
Epoch 2: val_auc improved from 0.71006 to 0.79477, saving model to exp9_cnn_bilstm_best.keras

Epoch 2: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 141s 392ms/step - accuracy: 0.6654 - auc: 0.7264 - loss: 0.8021 - val_accuracy: 0.7088 - val_auc: 0.7948 - val_loss: 0.7587 - learning_rate: 2.0000e-04
Epoch 3/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 370ms/step - accuracy: 0.7022 - auc: 0.7736 - loss: 0.7676
Epoch 3: val_auc improved from 0.79477 to 0.83940, saving model to exp9_cnn_bilstm_best.keras

Epoch 3: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 83s 396ms/step - accuracy: 0.7101 - auc: 0.7814 - loss: 0.7595 - val_accuracy: 0.7643 - val_auc: 0.8394 - val_loss: 0.7159 - learning_rate: 2.0000e-04
Epoch 4/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 380ms/step - accuracy: 0.7283 - auc: 0.8061 - loss: 0.7358
Epoch 4: val_auc improved from 0.83940 to 0.85799, saving model to exp9_cnn_bilstm_best.keras

Epoch 4: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 144s 404ms/step - accuracy: 0.7365 - auc: 0.8151 - loss: 0.7267 - val_accuracy: 0.7726 - val_auc: 0.8580 - val_loss: 0.6880 - learning_rate: 2.0000e-04
Epoch 5/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 379ms/step - accuracy: 0.7548 - auc: 0.8366 - loss: 0.7038
Epoch 5: val_auc improved from 0.85799 to 0.87051, saving model to exp9_cnn_bilstm_best.keras

Epoch 5: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 142s 404ms/step - accuracy: 0.7594 - auc: 0.8423 - loss: 0.6981 - val_accuracy: 0.7774 - val_auc: 0.8705 - val_loss: 0.6791 - learning_rate: 2.0000e-04
Epoch 6/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 375ms/step - accuracy: 0.7703 - auc: 0.8556 - loss: 0.6810
Epoch 6: val_auc improved from 0.87051 to 0.89626, saving model to exp9_cnn_bilstm_best.keras

Epoch 6: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 83s 401ms/step - accuracy: 0.7744 - auc: 0.8603 - loss: 0.6761 - val_accuracy: 0.8040 - val_auc: 0.8963 - val_loss: 0.6500 - learning_rate: 2.0000e-04
Epoch 7/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 368ms/step - accuracy: 0.7952 - auc: 0.8814 - loss: 0.6541
Epoch 7: val_auc improved from 0.89626 to 0.91156, saving model to exp9_cnn_bilstm_best.keras

Epoch 7: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 140s 392ms/step - accuracy: 0.7975 - auc: 0.8834 - loss: 0.6506 - val_accuracy: 0.8216 - val_auc: 0.9116 - val_loss: 0.6264 - learning_rate: 2.0000e-04
Epoch 8/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 373ms/step - accuracy: 0.8127 - auc: 0.8960 - loss: 0.6356
Epoch 8: val_auc improved from 0.91156 to 0.93030, saving model to exp9_cnn_bilstm_best.keras

Epoch 8: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 83s 398ms/step - accuracy: 0.8176 - auc: 0.9010 - loss: 0.6289 - val_accuracy: 0.8439 - val_auc: 0.9303 - val_loss: 0.5969 - learning_rate: 2.0000e-04
Epoch 9/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 377ms/step - accuracy: 0.8292 - auc: 0.9119 - loss: 0.6135
Epoch 9: val_auc improved from 0.93030 to 0.94052, saving model to exp9_cnn_bilstm_best.keras

Epoch 9: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 143s 402ms/step - accuracy: 0.8310 - auc: 0.9143 - loss: 0.6099 - val_accuracy: 0.8440 - val_auc: 0.9405 - val_loss: 0.5849 - learning_rate: 2.0000e-04
Epoch 10/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 375ms/step - accuracy: 0.8464 - auc: 0.9257 - loss: 0.5942
Epoch 10: val_auc improved from 0.94052 to 0.95395, saving model to exp9_cnn_bilstm_best.keras

Epoch 10: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 141s 399ms/step - accuracy: 0.8468 - auc: 0.9265 - loss: 0.5923 - val_accuracy: 0.8758 - val_auc: 0.9539 - val_loss: 0.5546 - learning_rate: 2.0000e-04
Epoch 11/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 377ms/step - accuracy: 0.8545 - auc: 0.9350 - loss: 0.5796
Epoch 11: val_auc did not improve from 0.95395
208/208 ━━━━━━━━━━━━━━━━━━━━ 142s 401ms/step - accuracy: 0.8575 - auc: 0.9369 - loss: 0.5764 - val_accuracy: 0.8706 - val_auc: 0.9522 - val_loss: 0.5543 - learning_rate: 2.0000e-04
Epoch 12/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 375ms/step - accuracy: 0.8685 - auc: 0.9415 - loss: 0.5665
Epoch 12: val_auc improved from 0.95395 to 0.96391, saving model to exp9_cnn_bilstm_best.keras

Epoch 12: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 83s 399ms/step - accuracy: 0.8694 - auc: 0.9435 - loss: 0.5634 - val_accuracy: 0.8920 - val_auc: 0.9639 - val_loss: 0.5302 - learning_rate: 2.0000e-04
Epoch 13/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 374ms/step - accuracy: 0.8733 - auc: 0.9490 - loss: 0.5536
Epoch 13: val_auc did not improve from 0.96391
208/208 ━━━━━━━━━━━━━━━━━━━━ 83s 397ms/step - accuracy: 0.8772 - auc: 0.9506 - loss: 0.5507 - val_accuracy: 0.8879 - val_auc: 0.9625 - val_loss: 0.5294 - learning_rate: 2.0000e-04
Epoch 14/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 375ms/step - accuracy: 0.8873 - auc: 0.9541 - loss: 0.5425
Epoch 14: val_auc improved from 0.96391 to 0.96429, saving model to exp9_cnn_bilstm_best.keras

Epoch 14: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 142s 400ms/step - accuracy: 0.8874 - auc: 0.9547 - loss: 0.5410 - val_accuracy: 0.8970 - val_auc: 0.9643 - val_loss: 0.5230 - learning_rate: 2.0000e-04
Epoch 15/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 374ms/step - accuracy: 0.8897 - auc: 0.9582 - loss: 0.5338
Epoch 15: val_auc improved from 0.96429 to 0.97039, saving model to exp9_cnn_bilstm_best.keras

Epoch 15: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 83s 399ms/step - accuracy: 0.8924 - auc: 0.9591 - loss: 0.5315 - val_accuracy: 0.9067 - val_auc: 0.9704 - val_loss: 0.5096 - learning_rate: 2.0000e-04
Epoch 16/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 386ms/step - accuracy: 0.9018 - auc: 0.9653 - loss: 0.5200
Epoch 16: val_auc improved from 0.97039 to 0.97584, saving model to exp9_cnn_bilstm_best.keras

Epoch 16: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 144s 411ms/step - accuracy: 0.8994 - auc: 0.9640 - loss: 0.5212 - val_accuracy: 0.9108 - val_auc: 0.9758 - val_loss: 0.5003 - learning_rate: 2.0000e-04
Epoch 17/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 384ms/step - accuracy: 0.9070 - auc: 0.9681 - loss: 0.5125
Epoch 17: val_auc improved from 0.97584 to 0.97839, saving model to exp9_cnn_bilstm_best.keras

Epoch 17: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 143s 415ms/step - accuracy: 0.9046 - auc: 0.9669 - loss: 0.5137 - val_accuracy: 0.9185 - val_auc: 0.9784 - val_loss: 0.4890 - learning_rate: 2.0000e-04
Epoch 18/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 490ms/step - accuracy: 0.9091 - auc: 0.9695 - loss: 0.5074
Epoch 18: val_auc did not improve from 0.97839
208/208 ━━━━━━━━━━━━━━━━━━━━ 108s 521ms/step - accuracy: 0.9118 - auc: 0.9701 - loss: 0.5053 - val_accuracy: 0.9228 - val_auc: 0.9780 - val_loss: 0.4859 - learning_rate: 2.0000e-04
Epoch 19/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 477ms/step - accuracy: 0.9131 - auc: 0.9720 - loss: 0.5006
Epoch 19: val_auc improved from 0.97839 to 0.97940, saving model to exp9_cnn_bilstm_best.keras

Epoch 19: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 106s 509ms/step - accuracy: 0.9150 - auc: 0.9720 - loss: 0.4997 - val_accuracy: 0.9222 - val_auc: 0.9794 - val_loss: 0.4846 - learning_rate: 2.0000e-04
Epoch 20/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 479ms/step - accuracy: 0.9210 - auc: 0.9755 - loss: 0.4913
Epoch 20: val_auc did not improve from 0.97940
208/208 ━━━━━━━━━━━━━━━━━━━━ 106s 511ms/step - accuracy: 0.9197 - auc: 0.9748 - loss: 0.4922 - val_accuracy: 0.9161 - val_auc: 0.9793 - val_loss: 0.4855 - learning_rate: 2.0000e-04
Epoch 21/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 476ms/step - accuracy: 0.9194 - auc: 0.9746 - loss: 0.4907
Epoch 21: val_auc improved from 0.97940 to 0.98385, saving model to exp9_cnn_bilstm_best.keras

Epoch 21: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 106s 508ms/step - accuracy: 0.9216 - auc: 0.9761 - loss: 0.4875 - val_accuracy: 0.9311 - val_auc: 0.9839 - val_loss: 0.4690 - learning_rate: 2.0000e-04
Epoch 22/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 482ms/step - accuracy: 0.9317 - auc: 0.9816 - loss: 0.4767
Epoch 22: val_auc did not improve from 0.98385
208/208 ━━━━━━━━━━━━━━━━━━━━ 107s 514ms/step - accuracy: 0.9283 - auc: 0.9803 - loss: 0.4785 - val_accuracy: 0.9298 - val_auc: 0.9826 - val_loss: 0.4698 - learning_rate: 2.0000e-04
Epoch 23/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 481ms/step - accuracy: 0.9294 - auc: 0.9787 - loss: 0.4777
Epoch 23: val_auc did not improve from 0.98385
208/208 ━━━━━━━━━━━━━━━━━━━━ 107s 513ms/step - accuracy: 0.9292 - auc: 0.9792 - loss: 0.4767 - val_accuracy: 0.9325 - val_auc: 0.9821 - val_loss: 0.4665 - learning_rate: 2.0000e-04
Epoch 24/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 477ms/step - accuracy: 0.9281 - auc: 0.9802 - loss: 0.4743
Epoch 24: val_auc did not improve from 0.98385
208/208 ━━━━━━━━━━━━━━━━━━━━ 106s 509ms/step - accuracy: 0.9324 - auc: 0.9817 - loss: 0.4708 - val_accuracy: 0.9243 - val_auc: 0.9827 - val_loss: 0.4740 - learning_rate: 2.0000e-04
Epoch 25/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 475ms/step - accuracy: 0.9387 - auc: 0.9831 - loss: 0.4659
Epoch 25: val_auc did not improve from 0.98385
208/208 ━━━━━━━━━━━━━━━━━━━━ 106s 507ms/step - accuracy: 0.9382 - auc: 0.9831 - loss: 0.4651 - val_accuracy: 0.9228 - val_auc: 0.9821 - val_loss: 0.4742 - learning_rate: 2.0000e-04
Epoch 26/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 455ms/step - accuracy: 0.9381 - auc: 0.9837 - loss: 0.4632
Epoch 26: val_auc improved from 0.98385 to 0.98435, saving model to exp9_cnn_bilstm_best.keras

Epoch 26: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 100s 482ms/step - accuracy: 0.9380 - auc: 0.9835 - loss: 0.4629 - val_accuracy: 0.9329 - val_auc: 0.9843 - val_loss: 0.4592 - learning_rate: 2.0000e-04
Epoch 27/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 420ms/step - accuracy: 0.9437 - auc: 0.9858 - loss: 0.4558
Epoch 27: val_auc improved from 0.98435 to 0.98634, saving model to exp9_cnn_bilstm_best.keras

Epoch 27: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 94s 453ms/step - accuracy: 0.9443 - auc: 0.9857 - loss: 0.4555 - val_accuracy: 0.9403 - val_auc: 0.9863 - val_loss: 0.4500 - learning_rate: 2.0000e-04
Epoch 28/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 479ms/step - accuracy: 0.9407 - auc: 0.9851 - loss: 0.4554
Epoch 28: val_auc did not improve from 0.98634
208/208 ━━━━━━━━━━━━━━━━━━━━ 106s 510ms/step - accuracy: 0.9426 - auc: 0.9857 - loss: 0.4543 - val_accuracy: 0.9335 - val_auc: 0.9856 - val_loss: 0.4553 - learning_rate: 2.0000e-04
Epoch 29/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 474ms/step - accuracy: 0.9507 - auc: 0.9887 - loss: 0.4458
Epoch 29: val_auc did not improve from 0.98634
208/208 ━━━━━━━━━━━━━━━━━━━━ 105s 506ms/step - accuracy: 0.9471 - auc: 0.9876 - loss: 0.4484 - val_accuracy: 0.9365 - val_auc: 0.9842 - val_loss: 0.4507 - learning_rate: 2.0000e-04
Epoch 30/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 474ms/step - accuracy: 0.9456 - auc: 0.9866 - loss: 0.4488
Epoch 30: val_auc improved from 0.98634 to 0.98760, saving model to exp9_cnn_bilstm_best.keras

Epoch 30: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 105s 507ms/step - accuracy: 0.9467 - auc: 0.9874 - loss: 0.4473 - val_accuracy: 0.9487 - val_auc: 0.9876 - val_loss: 0.4401 - learning_rate: 2.0000e-04
Epoch 31/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 477ms/step - accuracy: 0.9517 - auc: 0.9892 - loss: 0.4419
Epoch 31: val_auc improved from 0.98760 to 0.98819, saving model to exp9_cnn_bilstm_best.keras

Epoch 31: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 106s 510ms/step - accuracy: 0.9489 - auc: 0.9886 - loss: 0.4434 - val_accuracy: 0.9463 - val_auc: 0.9882 - val_loss: 0.4404 - learning_rate: 2.0000e-04
Epoch 32/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 477ms/step - accuracy: 0.9496 - auc: 0.9890 - loss: 0.4410
Epoch 32: val_auc improved from 0.98819 to 0.98853, saving model to exp9_cnn_bilstm_best.keras

Epoch 32: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 106s 509ms/step - accuracy: 0.9496 - auc: 0.9892 - loss: 0.4407 - val_accuracy: 0.9430 - val_auc: 0.9885 - val_loss: 0.4423 - learning_rate: 2.0000e-04
Epoch 33/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 469ms/step - accuracy: 0.9548 - auc: 0.9909 - loss: 0.4359
Epoch 33: val_auc did not improve from 0.98853
208/208 ━━━━━━━━━━━━━━━━━━━━ 103s 494ms/step - accuracy: 0.9531 - auc: 0.9900 - loss: 0.4372 - val_accuracy: 0.9385 - val_auc: 0.9864 - val_loss: 0.4453 - learning_rate: 2.0000e-04
Epoch 34/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 384ms/step - accuracy: 0.9542 - auc: 0.9903 - loss: 0.4342
Epoch 34: val_auc improved from 0.98853 to 0.98899, saving model to exp9_cnn_bilstm_best.keras

Epoch 34: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 85s 409ms/step - accuracy: 0.9552 - auc: 0.9908 - loss: 0.4335 - val_accuracy: 0.9508 - val_auc: 0.9890 - val_loss: 0.4311 - learning_rate: 2.0000e-04
Epoch 35/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 465ms/step - accuracy: 0.9567 - auc: 0.9913 - loss: 0.4308
Epoch 35: val_auc did not improve from 0.98899
208/208 ━━━━━━━━━━━━━━━━━━━━ 103s 498ms/step - accuracy: 0.9558 - auc: 0.9909 - loss: 0.4317 - val_accuracy: 0.9517 - val_auc: 0.9883 - val_loss: 0.4314 - learning_rate: 2.0000e-04
Epoch 36/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 477ms/step - accuracy: 0.9550 - auc: 0.9903 - loss: 0.4322
Epoch 36: val_auc improved from 0.98899 to 0.99064, saving model to exp9_cnn_bilstm_best.keras

Epoch 36: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 106s 510ms/step - accuracy: 0.9568 - auc: 0.9911 - loss: 0.4304 - val_accuracy: 0.9508 - val_auc: 0.9906 - val_loss: 0.4280 - learning_rate: 2.0000e-04
Epoch 37/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 479ms/step - accuracy: 0.9593 - auc: 0.9916 - loss: 0.4269
Epoch 37: val_auc did not improve from 0.99064
208/208 ━━━━━━━━━━━━━━━━━━━━ 106s 512ms/step - accuracy: 0.9610 - auc: 0.9921 - loss: 0.4252 - val_accuracy: 0.9519 - val_auc: 0.9891 - val_loss: 0.4286 - learning_rate: 2.0000e-04
Epoch 38/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 477ms/step - accuracy: 0.9619 - auc: 0.9934 - loss: 0.4212
Epoch 38: val_auc did not improve from 0.99064
208/208 ━━━━━━━━━━━━━━━━━━━━ 106s 509ms/step - accuracy: 0.9615 - auc: 0.9927 - loss: 0.4224 - val_accuracy: 0.9523 - val_auc: 0.9895 - val_loss: 0.4257 - learning_rate: 2.0000e-04
Epoch 39/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 474ms/step - accuracy: 0.9644 - auc: 0.9936 - loss: 0.4199
Epoch 39: val_auc did not improve from 0.99064
208/208 ━━━━━━━━━━━━━━━━━━━━ 105s 506ms/step - accuracy: 0.9638 - auc: 0.9933 - loss: 0.4203 - val_accuracy: 0.9507 - val_auc: 0.9890 - val_loss: 0.4284 - learning_rate: 2.0000e-04
Epoch 40/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 463ms/step - accuracy: 0.9649 - auc: 0.9939 - loss: 0.4190
Epoch 40: val_auc did not improve from 0.99064
208/208 ━━━━━━━━━━━━━━━━━━━━ 102s 489ms/step - accuracy: 0.9640 - auc: 0.9934 - loss: 0.4193 - val_accuracy: 0.9508 - val_auc: 0.9895 - val_loss: 0.4241 - learning_rate: 2.0000e-04
Epoch 41/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 467ms/step - accuracy: 0.9662 - auc: 0.9939 - loss: 0.4163
Epoch 41: ReduceLROnPlateau reducing learning rate to 9.999999747378752e-05.

Epoch 41: val_auc did not improve from 0.99064
208/208 ━━━━━━━━━━━━━━━━━━━━ 144s 499ms/step - accuracy: 0.9641 - auc: 0.9933 - loss: 0.4179 - val_accuracy: 0.9534 - val_auc: 0.9903 - val_loss: 0.4217 - learning_rate: 2.0000e-04
Epoch 42/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 481ms/step - accuracy: 0.9718 - auc: 0.9955 - loss: 0.4098
Epoch 42: val_auc improved from 0.99064 to 0.99094, saving model to exp9_cnn_bilstm_best.keras

Epoch 42: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 107s 513ms/step - accuracy: 0.9719 - auc: 0.9958 - loss: 0.4089 - val_accuracy: 0.9556 - val_auc: 0.9909 - val_loss: 0.4179 - learning_rate: 1.0000e-04
Epoch 43/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 475ms/step - accuracy: 0.9742 - auc: 0.9963 - loss: 0.4059
Epoch 43: val_auc improved from 0.99094 to 0.99218, saving model to exp9_cnn_bilstm_best.keras

Epoch 43: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 105s 507ms/step - accuracy: 0.9728 - auc: 0.9959 - loss: 0.4066 - val_accuracy: 0.9582 - val_auc: 0.9922 - val_loss: 0.4128 - learning_rate: 1.0000e-04
Epoch 44/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 466ms/step - accuracy: 0.9750 - auc: 0.9967 - loss: 0.4044
Epoch 44: val_auc did not improve from 0.99218
208/208 ━━━━━━━━━━━━━━━━━━━━ 103s 497ms/step - accuracy: 0.9742 - auc: 0.9964 - loss: 0.4053 - val_accuracy: 0.9601 - val_auc: 0.9921 - val_loss: 0.4121 - learning_rate: 1.0000e-04
Epoch 45/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 472ms/step - accuracy: 0.9738 - auc: 0.9961 - loss: 0.4048
Epoch 45: val_auc did not improve from 0.99218
208/208 ━━━━━━━━━━━━━━━━━━━━ 105s 505ms/step - accuracy: 0.9738 - auc: 0.9964 - loss: 0.4042 - val_accuracy: 0.9562 - val_auc: 0.9912 - val_loss: 0.4152 - learning_rate: 1.0000e-04
Epoch 46/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 475ms/step - accuracy: 0.9764 - auc: 0.9966 - loss: 0.4018
Epoch 46: val_auc did not improve from 0.99218
208/208 ━━━━━━━━━━━━━━━━━━━━ 106s 508ms/step - accuracy: 0.9751 - auc: 0.9967 - loss: 0.4025 - val_accuracy: 0.9582 - val_auc: 0.9917 - val_loss: 0.4151 - learning_rate: 1.0000e-04
Epoch 47/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 475ms/step - accuracy: 0.9774 - auc: 0.9972 - loss: 0.3998
Epoch 47: val_auc improved from 0.99218 to 0.99281, saving model to exp9_cnn_bilstm_best.keras

Epoch 47: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 106s 508ms/step - accuracy: 0.9750 - auc: 0.9965 - loss: 0.4021 - val_accuracy: 0.9586 - val_auc: 0.9928 - val_loss: 0.4109 - learning_rate: 1.0000e-04
Epoch 48/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 397ms/step - accuracy: 0.9758 - auc: 0.9969 - loss: 0.4002
Epoch 48: val_auc did not improve from 0.99281
208/208 ━━━━━━━━━━━━━━━━━━━━ 89s 426ms/step - accuracy: 0.9769 - auc: 0.9968 - loss: 0.3998 - val_accuracy: 0.9576 - val_auc: 0.9915 - val_loss: 0.4132 - learning_rate: 1.0000e-04
Epoch 49/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 469ms/step - accuracy: 0.9771 - auc: 0.9966 - loss: 0.3999
Epoch 49: val_auc did not improve from 0.99281
208/208 ━━━━━━━━━━━━━━━━━━━━ 104s 501ms/step - accuracy: 0.9768 - auc: 0.9970 - loss: 0.3997 - val_accuracy: 0.9606 - val_auc: 0.9917 - val_loss: 0.4105 - learning_rate: 1.0000e-04
Epoch 50/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 471ms/step - accuracy: 0.9766 - auc: 0.9966 - loss: 0.3991
Epoch 50: val_auc did not improve from 0.99281
208/208 ━━━━━━━━━━━━━━━━━━━━ 105s 504ms/step - accuracy: 0.9762 - auc: 0.9968 - loss: 0.3991 - val_accuracy: 0.9556 - val_auc: 0.9922 - val_loss: 0.4133 - learning_rate: 1.0000e-04
Epoch 51/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 474ms/step - accuracy: 0.9795 - auc: 0.9973 - loss: 0.3962
Epoch 51: val_auc did not improve from 0.99281
208/208 ━━━━━━━━━━━━━━━━━━━━ 105s 507ms/step - accuracy: 0.9789 - auc: 0.9972 - loss: 0.3968 - val_accuracy: 0.9583 - val_auc: 0.9921 - val_loss: 0.4120 - learning_rate: 1.0000e-04
Epoch 52/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 475ms/step - accuracy: 0.9772 - auc: 0.9972 - loss: 0.3972
Epoch 52: ReduceLROnPlateau reducing learning rate to 4.999999873689376e-05.

Epoch 52: val_auc did not improve from 0.99281
208/208 ━━━━━━━━━━━━━━━━━━━━ 106s 507ms/step - accuracy: 0.9773 - auc: 0.9971 - loss: 0.3973 - val_accuracy: 0.9561 - val_auc: 0.9917 - val_loss: 0.4130 - learning_rate: 1.0000e-04
Epoch 53/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 475ms/step - accuracy: 0.9813 - auc: 0.9976 - loss: 0.3935
Epoch 53: val_auc did not improve from 0.99281
208/208 ━━━━━━━━━━━━━━━━━━━━ 106s 507ms/step - accuracy: 0.9807 - auc: 0.9977 - loss: 0.3934 - val_accuracy: 0.9585 - val_auc: 0.9927 - val_loss: 0.4078 - learning_rate: 5.0000e-05
Epoch 54/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 472ms/step - accuracy: 0.9826 - auc: 0.9980 - loss: 0.3916
Epoch 54: val_auc did not improve from 0.99281
208/208 ━━━━━━━━━━━━━━━━━━━━ 105s 504ms/step - accuracy: 0.9809 - auc: 0.9978 - loss: 0.3925 - val_accuracy: 0.9634 - val_auc: 0.9926 - val_loss: 0.4045 - learning_rate: 5.0000e-05
Epoch 55/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 474ms/step - accuracy: 0.9818 - auc: 0.9978 - loss: 0.3925
Epoch 55: val_auc did not improve from 0.99281
208/208 ━━━━━━━━━━━━━━━━━━━━ 105s 505ms/step - accuracy: 0.9810 - auc: 0.9978 - loss: 0.3921 - val_accuracy: 0.9612 - val_auc: 0.9928 - val_loss: 0.4059 - learning_rate: 5.0000e-05
Epoch 56/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 381ms/step - accuracy: 0.9836 - auc: 0.9982 - loss: 0.3902
Epoch 56: val_auc did not improve from 0.99281
208/208 ━━━━━━━━━━━━━━━━━━━━ 84s 406ms/step - accuracy: 0.9818 - auc: 0.9979 - loss: 0.3913 - val_accuracy: 0.9612 - val_auc: 0.9926 - val_loss: 0.4050 - learning_rate: 5.0000e-05
Epoch 57/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 465ms/step - accuracy: 0.9826 - auc: 0.9981 - loss: 0.3896
Epoch 57: ReduceLROnPlateau reducing learning rate to 2.499999936844688e-05.

Epoch 57: val_auc did not improve from 0.99281
208/208 ━━━━━━━━━━━━━━━━━━━━ 103s 497ms/step - accuracy: 0.9811 - auc: 0.9980 - loss: 0.3908 - val_accuracy: 0.9597 - val_auc: 0.9927 - val_loss: 0.4065 - learning_rate: 5.0000e-05
Epoch 58/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 473ms/step - accuracy: 0.9812 - auc: 0.9981 - loss: 0.3904
Epoch 58: val_auc did not improve from 0.99281
208/208 ━━━━━━━━━━━━━━━━━━━━ 105s 505ms/step - accuracy: 0.9828 - auc: 0.9981 - loss: 0.3893 - val_accuracy: 0.9612 - val_auc: 0.9926 - val_loss: 0.4056 - learning_rate: 2.5000e-05
Epoch 59/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 474ms/step - accuracy: 0.9839 - auc: 0.9984 - loss: 0.3883
Epoch 59: val_auc improved from 0.99281 to 0.99312, saving model to exp9_cnn_bilstm_best.keras

Epoch 59: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 105s 507ms/step - accuracy: 0.9841 - auc: 0.9986 - loss: 0.3877 - val_accuracy: 0.9628 - val_auc: 0.9931 - val_loss: 0.4029 - learning_rate: 2.5000e-05
Epoch 60/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 472ms/step - accuracy: 0.9851 - auc: 0.9987 - loss: 0.3865
Epoch 60: val_auc improved from 0.99312 to 0.99337, saving model to exp9_cnn_bilstm_best.keras

Epoch 60: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 105s 505ms/step - accuracy: 0.9847 - auc: 0.9986 - loss: 0.3868 - val_accuracy: 0.9624 - val_auc: 0.9934 - val_loss: 0.4027 - learning_rate: 2.5000e-05
Epoch 61/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 472ms/step - accuracy: 0.9835 - auc: 0.9981 - loss: 0.3880
Epoch 61: val_auc improved from 0.99337 to 0.99342, saving model to exp9_cnn_bilstm_best.keras

Epoch 61: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 105s 505ms/step - accuracy: 0.9842 - auc: 0.9984 - loss: 0.3873 - val_accuracy: 0.9624 - val_auc: 0.9934 - val_loss: 0.4026 - learning_rate: 2.5000e-05
Epoch 62/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 450ms/step - accuracy: 0.9844 - auc: 0.9985 - loss: 0.3859
Epoch 62: val_auc did not improve from 0.99342
208/208 ━━━━━━━━━━━━━━━━━━━━ 136s 476ms/step - accuracy: 0.9841 - auc: 0.9985 - loss: 0.3867 - val_accuracy: 0.9618 - val_auc: 0.9931 - val_loss: 0.4027 - learning_rate: 2.5000e-05
Epoch 63/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 420ms/step - accuracy: 0.9846 - auc: 0.9985 - loss: 0.3870
Epoch 63: val_auc did not improve from 0.99342
208/208 ━━━━━━━━━━━━━━━━━━━━ 94s 451ms/step - accuracy: 0.9844 - auc: 0.9984 - loss: 0.3870 - val_accuracy: 0.9631 - val_auc: 0.9933 - val_loss: 0.4020 - learning_rate: 2.5000e-05
Epoch 64/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 469ms/step - accuracy: 0.9860 - auc: 0.9988 - loss: 0.3847
Epoch 64: val_auc did not improve from 0.99342
208/208 ━━━━━━━━━━━━━━━━━━━━ 104s 500ms/step - accuracy: 0.9858 - auc: 0.9987 - loss: 0.3851 - val_accuracy: 0.9622 - val_auc: 0.9932 - val_loss: 0.4018 - learning_rate: 2.5000e-05
Epoch 65/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 466ms/step - accuracy: 0.9864 - auc: 0.9985 - loss: 0.3846
Epoch 65: ReduceLROnPlateau reducing learning rate to 1.249999968422344e-05.

Epoch 65: val_auc did not improve from 0.99342
208/208 ━━━━━━━━━━━━━━━━━━━━ 103s 497ms/step - accuracy: 0.9857 - auc: 0.9986 - loss: 0.3849 - val_accuracy: 0.9634 - val_auc: 0.9934 - val_loss: 0.4007 - learning_rate: 2.5000e-05
Epoch 66/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 467ms/step - accuracy: 0.9854 - auc: 0.9988 - loss: 0.3850
Epoch 66: val_auc did not improve from 0.99342
208/208 ━━━━━━━━━━━━━━━━━━━━ 104s 499ms/step - accuracy: 0.9854 - auc: 0.9988 - loss: 0.3847 - val_accuracy: 0.9639 - val_auc: 0.9934 - val_loss: 0.4008 - learning_rate: 1.2500e-05
Epoch 67/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 461ms/step - accuracy: 0.9870 - auc: 0.9990 - loss: 0.3834
Epoch 67: val_auc improved from 0.99342 to 0.99354, saving model to exp9_cnn_bilstm_best.keras

Epoch 67: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 101s 487ms/step - accuracy: 0.9865 - auc: 0.9988 - loss: 0.3843 - val_accuracy: 0.9628 - val_auc: 0.9935 - val_loss: 0.4018 - learning_rate: 1.2500e-05
Epoch 68/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 399ms/step - accuracy: 0.9867 - auc: 0.9990 - loss: 0.3834
Epoch 68: val_auc did not improve from 0.99354
208/208 ━━━━━━━━━━━━━━━━━━━━ 131s 432ms/step - accuracy: 0.9860 - auc: 0.9988 - loss: 0.3842 - val_accuracy: 0.9642 - val_auc: 0.9933 - val_loss: 0.4012 - learning_rate: 1.2500e-05
Epoch 69/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 474ms/step - accuracy: 0.9868 - auc: 0.9987 - loss: 0.3845
Epoch 69: val_auc did not improve from 0.99354
208/208 ━━━━━━━━━━━━━━━━━━━━ 105s 507ms/step - accuracy: 0.9863 - auc: 0.9988 - loss: 0.3842 - val_accuracy: 0.9636 - val_auc: 0.9933 - val_loss: 0.4009 - learning_rate: 1.2500e-05
Epoch 70/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 470ms/step - accuracy: 0.9856 - auc: 0.9986 - loss: 0.3843
Epoch 70: val_auc did not improve from 0.99354
208/208 ━━━━━━━━━━━━━━━━━━━━ 104s 501ms/step - accuracy: 0.9859 - auc: 0.9987 - loss: 0.3842 - val_accuracy: 0.9627 - val_auc: 0.9931 - val_loss: 0.4014 - learning_rate: 1.2500e-05
Epoch 71/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 468ms/step - accuracy: 0.9876 - auc: 0.9991 - loss: 0.3824
Epoch 71: val_auc did not improve from 0.99354
208/208 ━━━━━━━━━━━━━━━━━━━━ 104s 500ms/step - accuracy: 0.9867 - auc: 0.9989 - loss: 0.3832 - val_accuracy: 0.9644 - val_auc: 0.9935 - val_loss: 0.3998 - learning_rate: 1.2500e-05
Epoch 72/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 468ms/step - accuracy: 0.9861 - auc: 0.9990 - loss: 0.3834
Epoch 72: ReduceLROnPlateau reducing learning rate to 6.24999984211172e-06.

Epoch 72: val_auc did not improve from 0.99354
208/208 ━━━━━━━━━━━━━━━━━━━━ 104s 500ms/step - accuracy: 0.9860 - auc: 0.9989 - loss: 0.3832 - val_accuracy: 0.9628 - val_auc: 0.9934 - val_loss: 0.4015 - learning_rate: 1.2500e-05
Epoch 73/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 381ms/step - accuracy: 0.9858 - auc: 0.9990 - loss: 0.3829
Epoch 73: val_auc did not improve from 0.99354
208/208 ━━━━━━━━━━━━━━━━━━━━ 90s 431ms/step - accuracy: 0.9856 - auc: 0.9987 - loss: 0.3838 - val_accuracy: 0.9631 - val_auc: 0.9934 - val_loss: 0.4008 - learning_rate: 6.2500e-06
Epoch 74/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 372ms/step - accuracy: 0.9862 - auc: 0.9990 - loss: 0.3833
Epoch 74: val_auc did not improve from 0.99354
208/208 ━━━━━━━━━━━━━━━━━━━━ 135s 397ms/step - accuracy: 0.9870 - auc: 0.9990 - loss: 0.3826 - val_accuracy: 0.9636 - val_auc: 0.9934 - val_loss: 0.4008 - learning_rate: 6.2500e-06
Epoch 75/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 432ms/step - accuracy: 0.9882 - auc: 0.9991 - loss: 0.3815
Epoch 75: val_auc did not improve from 0.99354
208/208 ━━━━━━━━━━━━━━━━━━━━ 96s 463ms/step - accuracy: 0.9874 - auc: 0.9990 - loss: 0.3824 - val_accuracy: 0.9647 - val_auc: 0.9934 - val_loss: 0.4005 - learning_rate: 6.2500e-06
Epoch 76/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 471ms/step - accuracy: 0.9856 - auc: 0.9988 - loss: 0.3834
Epoch 76: val_auc did not improve from 0.99354
208/208 ━━━━━━━━━━━━━━━━━━━━ 105s 503ms/step - accuracy: 0.9860 - auc: 0.9988 - loss: 0.3834 - val_accuracy: 0.9647 - val_auc: 0.9935 - val_loss: 0.4001 - learning_rate: 6.2500e-06
Epoch 77/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 468ms/step - accuracy: 0.9875 - auc: 0.9991 - loss: 0.3819
Epoch 77: ReduceLROnPlateau reducing learning rate to 3.12499992105586e-06.

Epoch 77: val_auc did not improve from 0.99354
208/208 ━━━━━━━━━━━━━━━━━━━━ 104s 499ms/step - accuracy: 0.9863 - auc: 0.9990 - loss: 0.3829 - val_accuracy: 0.9628 - val_auc: 0.9934 - val_loss: 0.4011 - learning_rate: 6.2500e-06
Epoch 78/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 468ms/step - accuracy: 0.9871 - auc: 0.9991 - loss: 0.3818
Epoch 78: val_auc improved from 0.99354 to 0.99354, saving model to exp9_cnn_bilstm_best.keras

Epoch 78: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 104s 501ms/step - accuracy: 0.9863 - auc: 0.9990 - loss: 0.3827 - val_accuracy: 0.9640 - val_auc: 0.9935 - val_loss: 0.4001 - learning_rate: 3.1250e-06
Epoch 79/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 472ms/step - accuracy: 0.9872 - auc: 0.9989 - loss: 0.3821
Epoch 79: val_auc did not improve from 0.99354
208/208 ━━━━━━━━━━━━━━━━━━━━ 105s 505ms/step - accuracy: 0.9868 - auc: 0.9989 - loss: 0.3825 - val_accuracy: 0.9640 - val_auc: 0.9935 - val_loss: 0.3998 - learning_rate: 3.1250e-06
Epoch 80/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 466ms/step - accuracy: 0.9874 - auc: 0.9989 - loss: 0.3824
Epoch 80: val_auc did not improve from 0.99354
208/208 ━━━━━━━━━━━━━━━━━━━━ 104s 498ms/step - accuracy: 0.9875 - auc: 0.9988 - loss: 0.3825 - val_accuracy: 0.9645 - val_auc: 0.9935 - val_loss: 0.3999 - learning_rate: 3.1250e-06
Epoch 81/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 468ms/step - accuracy: 0.9871 - auc: 0.9989 - loss: 0.3820
Epoch 81: val_auc did not improve from 0.99354
208/208 ━━━━━━━━━━━━━━━━━━━━ 104s 500ms/step - accuracy: 0.9866 - auc: 0.9989 - loss: 0.3825 - val_accuracy: 0.9639 - val_auc: 0.9935 - val_loss: 0.3997 - learning_rate: 3.1250e-06
Epoch 82/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 469ms/step - accuracy: 0.9866 - auc: 0.9990 - loss: 0.3824
Epoch 82: ReduceLROnPlateau reducing learning rate to 1.56249996052793e-06.

Epoch 82: val_auc improved from 0.99354 to 0.99360, saving model to exp9_cnn_bilstm_best.keras

Epoch 82: finished saving model to exp9_cnn_bilstm_best.keras
208/208 ━━━━━━━━━━━━━━━━━━━━ 105s 502ms/step - accuracy: 0.9867 - auc: 0.9991 - loss: 0.3820 - val_accuracy: 0.9637 - val_auc: 0.9936 - val_loss: 0.3998 - learning_rate: 3.1250e-06
Epoch 83/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 470ms/step - accuracy: 0.9867 - auc: 0.9991 - loss: 0.3818
Epoch 83: val_auc did not improve from 0.99360
208/208 ━━━━━━━━━━━━━━━━━━━━ 104s 498ms/step - accuracy: 0.9867 - auc: 0.9990 - loss: 0.3820 - val_accuracy: 0.9644 - val_auc: 0.9936 - val_loss: 0.3995 - learning_rate: 1.5625e-06
Epoch 84/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 366ms/step - accuracy: 0.9870 - auc: 0.9991 - loss: 0.3818
Epoch 84: val_auc did not improve from 0.99360
208/208 ━━━━━━━━━━━━━━━━━━━━ 120s 391ms/step - accuracy: 0.9867 - auc: 0.9990 - loss: 0.3823 - val_accuracy: 0.9640 - val_auc: 0.9935 - val_loss: 0.3995 - learning_rate: 1.5625e-06
Epoch 85/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 366ms/step - accuracy: 0.9884 - auc: 0.9992 - loss: 0.3810
Epoch 85: val_auc did not improve from 0.99360
208/208 ━━━━━━━━━━━━━━━━━━━━ 82s 391ms/step - accuracy: 0.9879 - auc: 0.9992 - loss: 0.3813 - val_accuracy: 0.9645 - val_auc: 0.9936 - val_loss: 0.3992 - learning_rate: 1.5625e-06
Epoch 86/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 365ms/step - accuracy: 0.9863 - auc: 0.9989 - loss: 0.3825
Epoch 86: val_auc did not improve from 0.99360
208/208 ━━━━━━━━━━━━━━━━━━━━ 82s 390ms/step - accuracy: 0.9869 - auc: 0.9990 - loss: 0.3822 - val_accuracy: 0.9642 - val_auc: 0.9936 - val_loss: 0.3995 - learning_rate: 1.5625e-06
Epoch 87/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 366ms/step - accuracy: 0.9864 - auc: 0.9989 - loss: 0.3827
Epoch 87: ReduceLROnPlateau reducing learning rate to 1e-06.

Epoch 87: val_auc did not improve from 0.99360
208/208 ━━━━━━━━━━━━━━━━━━━━ 82s 391ms/step - accuracy: 0.9871 - auc: 0.9990 - loss: 0.3821 - val_accuracy: 0.9645 - val_auc: 0.9935 - val_loss: 0.3997 - learning_rate: 1.5625e-06
Epoch 88/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 365ms/step - accuracy: 0.9877 - auc: 0.9991 - loss: 0.3811
Epoch 88: val_auc did not improve from 0.99360
208/208 ━━━━━━━━━━━━━━━━━━━━ 82s 390ms/step - accuracy: 0.9870 - auc: 0.9990 - loss: 0.3819 - val_accuracy: 0.9639 - val_auc: 0.9935 - val_loss: 0.3995 - learning_rate: 1.0000e-06
Epoch 89/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 367ms/step - accuracy: 0.9871 - auc: 0.9989 - loss: 0.3821
Epoch 89: val_auc did not improve from 0.99360
208/208 ━━━━━━━━━━━━━━━━━━━━ 82s 392ms/step - accuracy: 0.9872 - auc: 0.9990 - loss: 0.3822 - val_accuracy: 0.9647 - val_auc: 0.9936 - val_loss: 0.3996 - learning_rate: 1.0000e-06
Epoch 90/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 367ms/step - accuracy: 0.9866 - auc: 0.9986 - loss: 0.3821
Epoch 90: val_auc did not improve from 0.99360
208/208 ━━━━━━━━━━━━━━━━━━━━ 87s 417ms/step - accuracy: 0.9879 - auc: 0.9989 - loss: 0.3813 - val_accuracy: 0.9644 - val_auc: 0.9936 - val_loss: 0.3996 - learning_rate: 1.0000e-06
Epoch 91/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 365ms/step - accuracy: 0.9865 - auc: 0.9989 - loss: 0.3829
Epoch 91: val_auc did not improve from 0.99360
208/208 ━━━━━━━━━━━━━━━━━━━━ 136s 390ms/step - accuracy: 0.9877 - auc: 0.9990 - loss: 0.3818 - val_accuracy: 0.9639 - val_auc: 0.9936 - val_loss: 0.3995 - learning_rate: 1.0000e-06
Epoch 92/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 366ms/step - accuracy: 0.9859 - auc: 0.9986 - loss: 0.3828
Epoch 92: val_auc did not improve from 0.99360
208/208 ━━━━━━━━━━━━━━━━━━━━ 81s 391ms/step - accuracy: 0.9868 - auc: 0.9988 - loss: 0.3822 - val_accuracy: 0.9636 - val_auc: 0.9935 - val_loss: 0.3997 - learning_rate: 1.0000e-06
Epoch 93/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 366ms/step - accuracy: 0.9865 - auc: 0.9990 - loss: 0.3823
Epoch 93: val_auc did not improve from 0.99360
208/208 ━━━━━━━━━━━━━━━━━━━━ 87s 416ms/step - accuracy: 0.9868 - auc: 0.9990 - loss: 0.3819 - val_accuracy: 0.9639 - val_auc: 0.9936 - val_loss: 0.3998 - learning_rate: 1.0000e-06
Epoch 94/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 365ms/step - accuracy: 0.9870 - auc: 0.9988 - loss: 0.3817
Epoch 94: val_auc did not improve from 0.99360
208/208 ━━━━━━━━━━━━━━━━━━━━ 142s 415ms/step - accuracy: 0.9873 - auc: 0.9988 - loss: 0.3823 - val_accuracy: 0.9645 - val_auc: 0.9936 - val_loss: 0.3996 - learning_rate: 1.0000e-06
Epoch 95/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 362ms/step - accuracy: 0.9875 - auc: 0.9988 - loss: 0.3817
Epoch 95: val_auc did not improve from 0.99360
208/208 ━━━━━━━━━━━━━━━━━━━━ 136s 388ms/step - accuracy: 0.9873 - auc: 0.9989 - loss: 0.3821 - val_accuracy: 0.9639 - val_auc: 0.9936 - val_loss: 0.3997 - learning_rate: 1.0000e-06
Epoch 96/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 359ms/step - accuracy: 0.9871 - auc: 0.9988 - loss: 0.3818
Epoch 96: val_auc did not improve from 0.99360
208/208 ━━━━━━━━━━━━━━━━━━━━ 81s 384ms/step - accuracy: 0.9866 - auc: 0.9987 - loss: 0.3823 - val_accuracy: 0.9642 - val_auc: 0.9935 - val_loss: 0.3998 - learning_rate: 1.0000e-06
Epoch 97/150
208/208 ━━━━━━━━━━━━━━━━━━━━ 0s 361ms/step - accuracy: 0.9883 - auc: 0.9991 - loss: 0.3815
Epoch 97: val_auc did not improve from 0.99360
208/208 ━━━━━━━━━━━━━━━━━━━━ 82s 385ms/step - accuracy: 0.9870 - auc: 0.9990 - loss: 0.3822 - val_accuracy: 0.9642 - val_auc: 0.9935 - val_loss: 0.3996 - learning_rate: 1.0000e-06
Epoch 97: early stopping
Restoring model weights from the end of the best epoch: 82.
"""


log_exp10 = r"""
Epoch 1/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 157ms/step - accuracy: 0.5941 - auc: 0.6390 - loss: 0.8392
Epoch 1: val_auc improved from None to 0.80567, saving model to exp10_cnn_bilstm_best.keras

Epoch 1: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 80s 171ms/step - accuracy: 0.6383 - auc: 0.6996 - loss: 0.8051 - val_accuracy: 0.7204 - val_auc: 0.8057 - val_loss: 0.7253 - learning_rate: 3.0000e-04
Epoch 2/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.7060 - auc: 0.7801 - loss: 0.7298
Epoch 2: val_auc improved from 0.80567 to 0.85221, saving model to exp10_cnn_bilstm_best.keras

Epoch 2: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 80s 166ms/step - accuracy: 0.7152 - auc: 0.7929 - loss: 0.7128 - val_accuracy: 0.7658 - val_auc: 0.8522 - val_loss: 0.6521 - learning_rate: 3.0000e-04
Epoch 3/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.7463 - auc: 0.8321 - loss: 0.6634
Epoch 3: val_auc improved from 0.85221 to 0.87923, saving model to exp10_cnn_bilstm_best.keras

Epoch 3: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 83s 168ms/step - accuracy: 0.7572 - auc: 0.8416 - loss: 0.6503 - val_accuracy: 0.7786 - val_auc: 0.8792 - val_loss: 0.6172 - learning_rate: 3.0000e-04
Epoch 4/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.7822 - auc: 0.8682 - loss: 0.6118
Epoch 4: val_auc improved from 0.87923 to 0.91330, saving model to exp10_cnn_bilstm_best.keras

Epoch 4: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 168ms/step - accuracy: 0.7898 - auc: 0.8770 - loss: 0.5994 - val_accuracy: 0.8181 - val_auc: 0.9133 - val_loss: 0.5598 - learning_rate: 3.0000e-04
Epoch 5/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.8164 - auc: 0.9011 - loss: 0.5627
Epoch 5: val_auc improved from 0.91330 to 0.93303, saving model to exp10_cnn_bilstm_best.keras

Epoch 5: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 168ms/step - accuracy: 0.8207 - auc: 0.9057 - loss: 0.5544 - val_accuracy: 0.8394 - val_auc: 0.9330 - val_loss: 0.5201 - learning_rate: 3.0000e-04
Epoch 6/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 156ms/step - accuracy: 0.8292 - auc: 0.9158 - loss: 0.5345
Epoch 6: val_auc improved from 0.93303 to 0.94613, saving model to exp10_cnn_bilstm_best.keras

Epoch 6: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 168ms/step - accuracy: 0.8396 - auc: 0.9223 - loss: 0.5241 - val_accuracy: 0.8636 - val_auc: 0.9461 - val_loss: 0.4871 - learning_rate: 3.0000e-04
Epoch 7/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.8544 - auc: 0.9347 - loss: 0.5003
Epoch 7: val_auc improved from 0.94613 to 0.95684, saving model to exp10_cnn_bilstm_best.keras

Epoch 7: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 167ms/step - accuracy: 0.8577 - auc: 0.9359 - loss: 0.4971 - val_accuracy: 0.8851 - val_auc: 0.9568 - val_loss: 0.4553 - learning_rate: 3.0000e-04
Epoch 8/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.8664 - auc: 0.9455 - loss: 0.4772
Epoch 8: val_auc improved from 0.95684 to 0.96225, saving model to exp10_cnn_bilstm_best.keras

Epoch 8: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 70s 168ms/step - accuracy: 0.8727 - auc: 0.9482 - loss: 0.4708 - val_accuracy: 0.8914 - val_auc: 0.9622 - val_loss: 0.4410 - learning_rate: 3.0000e-04
Epoch 9/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.8810 - auc: 0.9507 - loss: 0.4622
Epoch 9: val_auc improved from 0.96225 to 0.96599, saving model to exp10_cnn_bilstm_best.keras

Epoch 9: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 168ms/step - accuracy: 0.8838 - auc: 0.9530 - loss: 0.4568 - val_accuracy: 0.8911 - val_auc: 0.9660 - val_loss: 0.4345 - learning_rate: 3.0000e-04
Epoch 10/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 156ms/step - accuracy: 0.8936 - auc: 0.9596 - loss: 0.4406
Epoch 10: val_auc improved from 0.96599 to 0.96657, saving model to exp10_cnn_bilstm_best.keras

Epoch 10: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 169ms/step - accuracy: 0.8927 - auc: 0.9599 - loss: 0.4391 - val_accuracy: 0.8840 - val_auc: 0.9666 - val_loss: 0.4394 - learning_rate: 3.0000e-04
Epoch 11/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 156ms/step - accuracy: 0.8952 - auc: 0.9636 - loss: 0.4287
Epoch 11: val_auc improved from 0.96657 to 0.97464, saving model to exp10_cnn_bilstm_best.keras

Epoch 11: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 168ms/step - accuracy: 0.8976 - auc: 0.9646 - loss: 0.4262 - val_accuracy: 0.9096 - val_auc: 0.9746 - val_loss: 0.4020 - learning_rate: 3.0000e-04
Epoch 12/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 156ms/step - accuracy: 0.9036 - auc: 0.9676 - loss: 0.4166
Epoch 12: val_auc improved from 0.97464 to 0.97495, saving model to exp10_cnn_bilstm_best.keras

Epoch 12: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 169ms/step - accuracy: 0.9061 - auc: 0.9688 - loss: 0.4130 - val_accuracy: 0.9076 - val_auc: 0.9749 - val_loss: 0.4043 - learning_rate: 3.0000e-04
Epoch 13/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 156ms/step - accuracy: 0.9100 - auc: 0.9708 - loss: 0.4055
Epoch 13: val_auc improved from 0.97495 to 0.97659, saving model to exp10_cnn_bilstm_best.keras

Epoch 13: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 70s 169ms/step - accuracy: 0.9119 - auc: 0.9718 - loss: 0.4026 - val_accuracy: 0.9152 - val_auc: 0.9766 - val_loss: 0.3939 - learning_rate: 3.0000e-04
Epoch 14/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 157ms/step - accuracy: 0.9164 - auc: 0.9740 - loss: 0.3950
Epoch 14: val_auc improved from 0.97659 to 0.97922, saving model to exp10_cnn_bilstm_best.keras

Epoch 14: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 169ms/step - accuracy: 0.9151 - auc: 0.9733 - loss: 0.3961 - val_accuracy: 0.9224 - val_auc: 0.9792 - val_loss: 0.3809 - learning_rate: 3.0000e-04
Epoch 15/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 156ms/step - accuracy: 0.9244 - auc: 0.9790 - loss: 0.3808
Epoch 15: val_auc improved from 0.97922 to 0.97930, saving model to exp10_cnn_bilstm_best.keras

Epoch 15: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 169ms/step - accuracy: 0.9238 - auc: 0.9780 - loss: 0.3823 - val_accuracy: 0.9102 - val_auc: 0.9793 - val_loss: 0.3903 - learning_rate: 3.0000e-04
Epoch 16/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 158ms/step - accuracy: 0.9280 - auc: 0.9809 - loss: 0.3730
Epoch 16: val_auc improved from 0.97930 to 0.98032, saving model to exp10_cnn_bilstm_best.keras

Epoch 16: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 83s 170ms/step - accuracy: 0.9281 - auc: 0.9805 - loss: 0.3727 - val_accuracy: 0.9147 - val_auc: 0.9803 - val_loss: 0.3851 - learning_rate: 3.0000e-04
Epoch 17/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 156ms/step - accuracy: 0.9300 - auc: 0.9809 - loss: 0.3690
Epoch 17: val_auc improved from 0.98032 to 0.98444, saving model to exp10_cnn_bilstm_best.keras

Epoch 17: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 168ms/step - accuracy: 0.9295 - auc: 0.9804 - loss: 0.3702 - val_accuracy: 0.9322 - val_auc: 0.9844 - val_loss: 0.3611 - learning_rate: 3.0000e-04
Epoch 18/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.9363 - auc: 0.9820 - loss: 0.3644
Epoch 18: val_auc improved from 0.98444 to 0.98525, saving model to exp10_cnn_bilstm_best.keras

Epoch 18: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 168ms/step - accuracy: 0.9373 - auc: 0.9832 - loss: 0.3603 - val_accuracy: 0.9380 - val_auc: 0.9852 - val_loss: 0.3524 - learning_rate: 3.0000e-04
Epoch 19/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 156ms/step - accuracy: 0.9367 - auc: 0.9841 - loss: 0.3564
Epoch 19: val_auc did not improve from 0.98525
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 168ms/step - accuracy: 0.9380 - auc: 0.9848 - loss: 0.3544 - val_accuracy: 0.9285 - val_auc: 0.9841 - val_loss: 0.3606 - learning_rate: 3.0000e-04
Epoch 20/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.9436 - auc: 0.9860 - loss: 0.3483
Epoch 20: val_auc improved from 0.98525 to 0.98565, saving model to exp10_cnn_bilstm_best.keras

Epoch 20: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 70s 168ms/step - accuracy: 0.9418 - auc: 0.9859 - loss: 0.3485 - val_accuracy: 0.9370 - val_auc: 0.9856 - val_loss: 0.3518 - learning_rate: 3.0000e-04
Epoch 21/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 156ms/step - accuracy: 0.9436 - auc: 0.9865 - loss: 0.3464
Epoch 21: val_auc improved from 0.98565 to 0.98582, saving model to exp10_cnn_bilstm_best.keras

Epoch 21: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 168ms/step - accuracy: 0.9435 - auc: 0.9870 - loss: 0.3444 - val_accuracy: 0.9356 - val_auc: 0.9858 - val_loss: 0.3552 - learning_rate: 3.0000e-04
Epoch 22/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 156ms/step - accuracy: 0.9467 - auc: 0.9882 - loss: 0.3395
Epoch 22: val_auc improved from 0.98582 to 0.98659, saving model to exp10_cnn_bilstm_best.keras

Epoch 22: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 168ms/step - accuracy: 0.9466 - auc: 0.9880 - loss: 0.3391 - val_accuracy: 0.9428 - val_auc: 0.9866 - val_loss: 0.3434 - learning_rate: 3.0000e-04
Epoch 23/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.9474 - auc: 0.9890 - loss: 0.3357
Epoch 23: val_auc improved from 0.98659 to 0.98756, saving model to exp10_cnn_bilstm_best.keras

Epoch 23: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 167ms/step - accuracy: 0.9479 - auc: 0.9885 - loss: 0.3364 - val_accuracy: 0.9394 - val_auc: 0.9876 - val_loss: 0.3421 - learning_rate: 3.0000e-04
Epoch 24/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 156ms/step - accuracy: 0.9542 - auc: 0.9900 - loss: 0.3293
Epoch 24: val_auc did not improve from 0.98756
416/416 ━━━━━━━━━━━━━━━━━━━━ 83s 169ms/step - accuracy: 0.9524 - auc: 0.9894 - loss: 0.3311 - val_accuracy: 0.9409 - val_auc: 0.9865 - val_loss: 0.3411 - learning_rate: 3.0000e-04
Epoch 25/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9516 - auc: 0.9907 - loss: 0.3271
Epoch 25: val_auc did not improve from 0.98756
416/416 ━━━━━━━━━━━━━━━━━━━━ 69s 166ms/step - accuracy: 0.9519 - auc: 0.9901 - loss: 0.3281 - val_accuracy: 0.9486 - val_auc: 0.9869 - val_loss: 0.3333 - learning_rate: 3.0000e-04
Epoch 26/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 156ms/step - accuracy: 0.9526 - auc: 0.9900 - loss: 0.3263
Epoch 26: val_auc improved from 0.98756 to 0.98943, saving model to exp10_cnn_bilstm_best.keras

Epoch 26: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 83s 168ms/step - accuracy: 0.9542 - auc: 0.9907 - loss: 0.3240 - val_accuracy: 0.9440 - val_auc: 0.9894 - val_loss: 0.3328 - learning_rate: 3.0000e-04
Epoch 27/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.9544 - auc: 0.9916 - loss: 0.3216
Epoch 27: val_auc did not improve from 0.98943
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 167ms/step - accuracy: 0.9553 - auc: 0.9914 - loss: 0.3220 - val_accuracy: 0.9448 - val_auc: 0.9887 - val_loss: 0.3315 - learning_rate: 3.0000e-04
Epoch 28/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 157ms/step - accuracy: 0.9610 - auc: 0.9929 - loss: 0.3144
Epoch 28: val_auc improved from 0.98943 to 0.98997, saving model to exp10_cnn_bilstm_best.keras

Epoch 28: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 83s 170ms/step - accuracy: 0.9574 - auc: 0.9919 - loss: 0.3181 - val_accuracy: 0.9466 - val_auc: 0.9900 - val_loss: 0.3280 - learning_rate: 3.0000e-04
Epoch 29/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9588 - auc: 0.9932 - loss: 0.3128
Epoch 29: val_auc did not improve from 0.98997
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 166ms/step - accuracy: 0.9577 - auc: 0.9928 - loss: 0.3146 - val_accuracy: 0.9469 - val_auc: 0.9896 - val_loss: 0.3278 - learning_rate: 3.0000e-04
Epoch 30/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 156ms/step - accuracy: 0.9600 - auc: 0.9928 - loss: 0.3126
Epoch 30: val_auc did not improve from 0.98997
416/416 ━━━━━━━━━━━━━━━━━━━━ 70s 168ms/step - accuracy: 0.9604 - auc: 0.9926 - loss: 0.3128 - val_accuracy: 0.9449 - val_auc: 0.9882 - val_loss: 0.3291 - learning_rate: 3.0000e-04
Epoch 31/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 156ms/step - accuracy: 0.9631 - auc: 0.9933 - loss: 0.3085
Epoch 31: val_auc did not improve from 0.98997
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 169ms/step - accuracy: 0.9622 - auc: 0.9933 - loss: 0.3092 - val_accuracy: 0.9448 - val_auc: 0.9895 - val_loss: 0.3330 - learning_rate: 3.0000e-04
Epoch 32/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 156ms/step - accuracy: 0.9630 - auc: 0.9935 - loss: 0.3071
Epoch 32: val_auc improved from 0.98997 to 0.99010, saving model to exp10_cnn_bilstm_best.keras

Epoch 32: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 169ms/step - accuracy: 0.9624 - auc: 0.9933 - loss: 0.3083 - val_accuracy: 0.9507 - val_auc: 0.9901 - val_loss: 0.3244 - learning_rate: 3.0000e-04
Epoch 33/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.9645 - auc: 0.9942 - loss: 0.3040
Epoch 33: val_auc improved from 0.99010 to 0.99034, saving model to exp10_cnn_bilstm_best.keras

Epoch 33: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 167ms/step - accuracy: 0.9641 - auc: 0.9939 - loss: 0.3054 - val_accuracy: 0.9421 - val_auc: 0.9903 - val_loss: 0.3337 - learning_rate: 3.0000e-04
Epoch 34/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 156ms/step - accuracy: 0.9637 - auc: 0.9941 - loss: 0.3041
Epoch 34: val_auc improved from 0.99034 to 0.99107, saving model to exp10_cnn_bilstm_best.keras

Epoch 34: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 83s 169ms/step - accuracy: 0.9643 - auc: 0.9940 - loss: 0.3044 - val_accuracy: 0.9519 - val_auc: 0.9911 - val_loss: 0.3192 - learning_rate: 3.0000e-04
Epoch 35/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 157ms/step - accuracy: 0.9657 - auc: 0.9945 - loss: 0.3026
Epoch 35: val_auc improved from 0.99107 to 0.99140, saving model to exp10_cnn_bilstm_best.keras

Epoch 35: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 70s 169ms/step - accuracy: 0.9644 - auc: 0.9940 - loss: 0.3041 - val_accuracy: 0.9516 - val_auc: 0.9914 - val_loss: 0.3189 - learning_rate: 3.0000e-04
Epoch 36/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.9667 - auc: 0.9944 - loss: 0.3017
Epoch 36: val_auc did not improve from 0.99140
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 167ms/step - accuracy: 0.9681 - auc: 0.9947 - loss: 0.2991 - val_accuracy: 0.9424 - val_auc: 0.9909 - val_loss: 0.3344 - learning_rate: 3.0000e-04
Epoch 37/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 156ms/step - accuracy: 0.9668 - auc: 0.9944 - loss: 0.3001
Epoch 37: val_auc did not improve from 0.99140
416/416 ━━━━━━━━━━━━━━━━━━━━ 70s 169ms/step - accuracy: 0.9668 - auc: 0.9944 - loss: 0.2993 - val_accuracy: 0.9532 - val_auc: 0.9906 - val_loss: 0.3165 - learning_rate: 3.0000e-04
Epoch 38/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.9705 - auc: 0.9958 - loss: 0.2927
Epoch 38: val_auc improved from 0.99140 to 0.99200, saving model to exp10_cnn_bilstm_best.keras

Epoch 38: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 87s 181ms/step - accuracy: 0.9694 - auc: 0.9951 - loss: 0.2948 - val_accuracy: 0.9598 - val_auc: 0.9920 - val_loss: 0.3071 - learning_rate: 3.0000e-04
Epoch 39/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 156ms/step - accuracy: 0.9698 - auc: 0.9952 - loss: 0.2940
Epoch 39: val_auc did not improve from 0.99200
416/416 ━━━━━━━━━━━━━━━━━━━━ 77s 168ms/step - accuracy: 0.9686 - auc: 0.9948 - loss: 0.2956 - val_accuracy: 0.9580 - val_auc: 0.9913 - val_loss: 0.3108 - learning_rate: 3.0000e-04
Epoch 40/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.9697 - auc: 0.9958 - loss: 0.2933
Epoch 40: val_auc did not improve from 0.99200
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 168ms/step - accuracy: 0.9684 - auc: 0.9954 - loss: 0.2947 - val_accuracy: 0.9555 - val_auc: 0.9899 - val_loss: 0.3130 - learning_rate: 3.0000e-04
Epoch 41/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 156ms/step - accuracy: 0.9707 - auc: 0.9957 - loss: 0.2915
Epoch 41: val_auc did not improve from 0.99200
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 168ms/step - accuracy: 0.9697 - auc: 0.9957 - loss: 0.2922 - val_accuracy: 0.9549 - val_auc: 0.9911 - val_loss: 0.3115 - learning_rate: 3.0000e-04
Epoch 42/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9721 - auc: 0.9963 - loss: 0.2897
Epoch 42: val_auc did not improve from 0.99200
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 166ms/step - accuracy: 0.9707 - auc: 0.9958 - loss: 0.2914 - val_accuracy: 0.9502 - val_auc: 0.9897 - val_loss: 0.3213 - learning_rate: 3.0000e-04
Epoch 43/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 168ms/step - accuracy: 0.9744 - auc: 0.9967 - loss: 0.2857
Epoch 43: ReduceLROnPlateau reducing learning rate to 0.0001500000071246177.

Epoch 43: val_auc did not improve from 0.99200
416/416 ━━━━━━━━━━━━━━━━━━━━ 75s 180ms/step - accuracy: 0.9726 - auc: 0.9964 - loss: 0.2877 - val_accuracy: 0.9561 - val_auc: 0.9914 - val_loss: 0.3079 - learning_rate: 3.0000e-04
Epoch 44/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 158ms/step - accuracy: 0.9816 - auc: 0.9981 - loss: 0.2759
Epoch 44: val_auc improved from 0.99200 to 0.99325, saving model to exp10_cnn_bilstm_best.keras

Epoch 44: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 71s 170ms/step - accuracy: 0.9811 - auc: 0.9981 - loss: 0.2749 - val_accuracy: 0.9651 - val_auc: 0.9933 - val_loss: 0.2971 - learning_rate: 1.5000e-04
Epoch 45/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.9836 - auc: 0.9983 - loss: 0.2716
Epoch 45: val_auc improved from 0.99325 to 0.99347, saving model to exp10_cnn_bilstm_best.keras

Epoch 45: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 167ms/step - accuracy: 0.9830 - auc: 0.9983 - loss: 0.2709 - val_accuracy: 0.9637 - val_auc: 0.9935 - val_loss: 0.2944 - learning_rate: 1.5000e-04
Epoch 46/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9837 - auc: 0.9987 - loss: 0.2686
Epoch 46: val_auc did not improve from 0.99347
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 166ms/step - accuracy: 0.9842 - auc: 0.9986 - loss: 0.2678 - val_accuracy: 0.9628 - val_auc: 0.9931 - val_loss: 0.2969 - learning_rate: 1.5000e-04
Epoch 47/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9838 - auc: 0.9985 - loss: 0.2674
Epoch 47: val_auc improved from 0.99347 to 0.99382, saving model to exp10_cnn_bilstm_best.keras

Epoch 47: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 166ms/step - accuracy: 0.9833 - auc: 0.9984 - loss: 0.2682 - val_accuracy: 0.9678 - val_auc: 0.9938 - val_loss: 0.2893 - learning_rate: 1.5000e-04
Epoch 48/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9848 - auc: 0.9987 - loss: 0.2650
Epoch 48: val_auc did not improve from 0.99382
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 167ms/step - accuracy: 0.9844 - auc: 0.9986 - loss: 0.2656 - val_accuracy: 0.9607 - val_auc: 0.9937 - val_loss: 0.2966 - learning_rate: 1.5000e-04
Epoch 49/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9855 - auc: 0.9988 - loss: 0.2639
Epoch 49: val_auc did not improve from 0.99382
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 166ms/step - accuracy: 0.9860 - auc: 0.9988 - loss: 0.2636 - val_accuracy: 0.9595 - val_auc: 0.9935 - val_loss: 0.3010 - learning_rate: 1.5000e-04
Epoch 50/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9853 - auc: 0.9989 - loss: 0.2628
Epoch 50: val_auc did not improve from 0.99382
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 166ms/step - accuracy: 0.9846 - auc: 0.9987 - loss: 0.2639 - val_accuracy: 0.9653 - val_auc: 0.9938 - val_loss: 0.2891 - learning_rate: 1.5000e-04
Epoch 51/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.9846 - auc: 0.9989 - loss: 0.2617
Epoch 51: val_auc did not improve from 0.99382
416/416 ━━━━━━━━━━━━━━━━━━━━ 83s 167ms/step - accuracy: 0.9846 - auc: 0.9988 - loss: 0.2621 - val_accuracy: 0.9659 - val_auc: 0.9932 - val_loss: 0.2922 - learning_rate: 1.5000e-04
Epoch 52/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 153ms/step - accuracy: 0.9864 - auc: 0.9990 - loss: 0.2603
Epoch 52: ReduceLROnPlateau reducing learning rate to 7.500000356230885e-05.

Epoch 52: val_auc did not improve from 0.99382
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 165ms/step - accuracy: 0.9852 - auc: 0.9988 - loss: 0.2619 - val_accuracy: 0.9613 - val_auc: 0.9924 - val_loss: 0.2944 - learning_rate: 1.5000e-04
Epoch 53/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 157ms/step - accuracy: 0.9883 - auc: 0.9992 - loss: 0.2573
Epoch 53: val_auc improved from 0.99382 to 0.99415, saving model to exp10_cnn_bilstm_best.keras

Epoch 53: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 83s 169ms/step - accuracy: 0.9898 - auc: 0.9993 - loss: 0.2555 - val_accuracy: 0.9644 - val_auc: 0.9941 - val_loss: 0.2900 - learning_rate: 7.5000e-05
Epoch 54/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9905 - auc: 0.9994 - loss: 0.2523
Epoch 54: val_auc did not improve from 0.99415
416/416 ━━━━━━━━━━━━━━━━━━━━ 69s 166ms/step - accuracy: 0.9903 - auc: 0.9994 - loss: 0.2529 - val_accuracy: 0.9648 - val_auc: 0.9940 - val_loss: 0.2905 - learning_rate: 7.5000e-05
Epoch 55/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9911 - auc: 0.9995 - loss: 0.2518
Epoch 55: val_auc improved from 0.99415 to 0.99428, saving model to exp10_cnn_bilstm_best.keras

Epoch 55: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 166ms/step - accuracy: 0.9903 - auc: 0.9994 - loss: 0.2522 - val_accuracy: 0.9683 - val_auc: 0.9943 - val_loss: 0.2825 - learning_rate: 7.5000e-05
Epoch 56/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 153ms/step - accuracy: 0.9905 - auc: 0.9993 - loss: 0.2513
Epoch 56: val_auc did not improve from 0.99428
416/416 ━━━━━━━━━━━━━━━━━━━━ 69s 166ms/step - accuracy: 0.9904 - auc: 0.9993 - loss: 0.2515 - val_accuracy: 0.9672 - val_auc: 0.9941 - val_loss: 0.2839 - learning_rate: 7.5000e-05
Epoch 57/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9905 - auc: 0.9994 - loss: 0.2505
Epoch 57: val_auc improved from 0.99428 to 0.99441, saving model to exp10_cnn_bilstm_best.keras

Epoch 57: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 166ms/step - accuracy: 0.9911 - auc: 0.9995 - loss: 0.2502 - val_accuracy: 0.9671 - val_auc: 0.9944 - val_loss: 0.2816 - learning_rate: 7.5000e-05
Epoch 58/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 153ms/step - accuracy: 0.9926 - auc: 0.9997 - loss: 0.2475
Epoch 58: val_auc did not improve from 0.99441
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 165ms/step - accuracy: 0.9908 - auc: 0.9996 - loss: 0.2490 - val_accuracy: 0.9648 - val_auc: 0.9936 - val_loss: 0.2863 - learning_rate: 7.5000e-05
Epoch 59/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 153ms/step - accuracy: 0.9904 - auc: 0.9995 - loss: 0.2495
Epoch 59: val_auc improved from 0.99441 to 0.99443, saving model to exp10_cnn_bilstm_best.keras

Epoch 59: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 165ms/step - accuracy: 0.9904 - auc: 0.9994 - loss: 0.2498 - val_accuracy: 0.9672 - val_auc: 0.9944 - val_loss: 0.2815 - learning_rate: 7.5000e-05
Epoch 60/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9913 - auc: 0.9996 - loss: 0.2487
Epoch 60: val_auc improved from 0.99443 to 0.99470, saving model to exp10_cnn_bilstm_best.keras

Epoch 60: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 166ms/step - accuracy: 0.9907 - auc: 0.9995 - loss: 0.2492 - val_accuracy: 0.9671 - val_auc: 0.9947 - val_loss: 0.2818 - learning_rate: 7.5000e-05
Epoch 61/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 153ms/step - accuracy: 0.9911 - auc: 0.9996 - loss: 0.2476
Epoch 61: val_auc did not improve from 0.99470
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 165ms/step - accuracy: 0.9914 - auc: 0.9996 - loss: 0.2472 - val_accuracy: 0.9654 - val_auc: 0.9943 - val_loss: 0.2833 - learning_rate: 7.5000e-05
Epoch 62/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 153ms/step - accuracy: 0.9910 - auc: 0.9996 - loss: 0.2465
Epoch 62: val_auc did not improve from 0.99470
416/416 ━━━━━━━━━━━━━━━━━━━━ 69s 166ms/step - accuracy: 0.9909 - auc: 0.9996 - loss: 0.2471 - val_accuracy: 0.9675 - val_auc: 0.9945 - val_loss: 0.2807 - learning_rate: 7.5000e-05
Epoch 63/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 153ms/step - accuracy: 0.9902 - auc: 0.9995 - loss: 0.2475
Epoch 63: val_auc did not improve from 0.99470
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 166ms/step - accuracy: 0.9903 - auc: 0.9995 - loss: 0.2475 - val_accuracy: 0.9657 - val_auc: 0.9943 - val_loss: 0.2839 - learning_rate: 7.5000e-05
Epoch 64/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9917 - auc: 0.9997 - loss: 0.2456
Epoch 64: val_auc did not improve from 0.99470
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 167ms/step - accuracy: 0.9914 - auc: 0.9996 - loss: 0.2459 - val_accuracy: 0.9663 - val_auc: 0.9942 - val_loss: 0.2845 - learning_rate: 7.5000e-05
Epoch 65/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9921 - auc: 0.9995 - loss: 0.2451
Epoch 65: ReduceLROnPlateau reducing learning rate to 3.7500001781154424e-05.

Epoch 65: val_auc did not improve from 0.99470
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 166ms/step - accuracy: 0.9911 - auc: 0.9995 - loss: 0.2460 - val_accuracy: 0.9671 - val_auc: 0.9940 - val_loss: 0.2802 - learning_rate: 7.5000e-05
Epoch 66/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 153ms/step - accuracy: 0.9929 - auc: 0.9997 - loss: 0.2430
Epoch 66: val_auc improved from 0.99470 to 0.99477, saving model to exp10_cnn_bilstm_best.keras

Epoch 66: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 166ms/step - accuracy: 0.9929 - auc: 0.9996 - loss: 0.2430 - val_accuracy: 0.9693 - val_auc: 0.9948 - val_loss: 0.2774 - learning_rate: 3.7500e-05
Epoch 67/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 153ms/step - accuracy: 0.9938 - auc: 0.9997 - loss: 0.2418
Epoch 67: val_auc improved from 0.99477 to 0.99481, saving model to exp10_cnn_bilstm_best.keras

Epoch 67: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 166ms/step - accuracy: 0.9933 - auc: 0.9997 - loss: 0.2422 - val_accuracy: 0.9689 - val_auc: 0.9948 - val_loss: 0.2787 - learning_rate: 3.7500e-05
Epoch 68/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 153ms/step - accuracy: 0.9939 - auc: 0.9998 - loss: 0.2411
Epoch 68: val_auc did not improve from 0.99481
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 165ms/step - accuracy: 0.9930 - auc: 0.9997 - loss: 0.2420 - val_accuracy: 0.9678 - val_auc: 0.9948 - val_loss: 0.2768 - learning_rate: 3.7500e-05
Epoch 69/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9944 - auc: 0.9998 - loss: 0.2402
Epoch 69: val_auc did not improve from 0.99481
416/416 ━━━━━━━━━━━━━━━━━━━━ 69s 167ms/step - accuracy: 0.9935 - auc: 0.9998 - loss: 0.2409 - val_accuracy: 0.9690 - val_auc: 0.9946 - val_loss: 0.2774 - learning_rate: 3.7500e-05
Epoch 70/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.9930 - auc: 0.9997 - loss: 0.2410
Epoch 70: val_auc did not improve from 0.99481
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 167ms/step - accuracy: 0.9932 - auc: 0.9997 - loss: 0.2407 - val_accuracy: 0.9675 - val_auc: 0.9944 - val_loss: 0.2778 - learning_rate: 3.7500e-05
Epoch 71/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 153ms/step - accuracy: 0.9945 - auc: 0.9998 - loss: 0.2396
Epoch 71: val_auc improved from 0.99481 to 0.99497, saving model to exp10_cnn_bilstm_best.keras

Epoch 71: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 166ms/step - accuracy: 0.9941 - auc: 0.9998 - loss: 0.2400 - val_accuracy: 0.9710 - val_auc: 0.9950 - val_loss: 0.2739 - learning_rate: 3.7500e-05
Epoch 72/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.9939 - auc: 0.9997 - loss: 0.2400
Epoch 72: val_auc improved from 0.99497 to 0.99497, saving model to exp10_cnn_bilstm_best.keras

Epoch 72: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 167ms/step - accuracy: 0.9930 - auc: 0.9997 - loss: 0.2404 - val_accuracy: 0.9705 - val_auc: 0.9950 - val_loss: 0.2731 - learning_rate: 3.7500e-05
Epoch 73/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 153ms/step - accuracy: 0.9941 - auc: 0.9998 - loss: 0.2387
Epoch 73: val_auc did not improve from 0.99497
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 166ms/step - accuracy: 0.9934 - auc: 0.9998 - loss: 0.2396 - val_accuracy: 0.9677 - val_auc: 0.9947 - val_loss: 0.2750 - learning_rate: 3.7500e-05
Epoch 74/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 156ms/step - accuracy: 0.9936 - auc: 0.9997 - loss: 0.2394
Epoch 74: val_auc did not improve from 0.99497
416/416 ━━━━━━━━━━━━━━━━━━━━ 70s 168ms/step - accuracy: 0.9938 - auc: 0.9998 - loss: 0.2392 - val_accuracy: 0.9690 - val_auc: 0.9943 - val_loss: 0.2756 - learning_rate: 3.7500e-05
Epoch 75/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 153ms/step - accuracy: 0.9941 - auc: 0.9998 - loss: 0.2390
Epoch 75: val_auc did not improve from 0.99497
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 166ms/step - accuracy: 0.9935 - auc: 0.9998 - loss: 0.2392 - val_accuracy: 0.9678 - val_auc: 0.9943 - val_loss: 0.2749 - learning_rate: 3.7500e-05
Epoch 76/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.9942 - auc: 0.9998 - loss: 0.2388
Epoch 76: ReduceLROnPlateau reducing learning rate to 1.8750000890577212e-05.

Epoch 76: val_auc did not improve from 0.99497
416/416 ━━━━━━━━━━━━━━━━━━━━ 70s 167ms/step - accuracy: 0.9942 - auc: 0.9998 - loss: 0.2386 - val_accuracy: 0.9666 - val_auc: 0.9946 - val_loss: 0.2780 - learning_rate: 3.7500e-05
Epoch 77/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 153ms/step - accuracy: 0.9949 - auc: 0.9999 - loss: 0.2368
Epoch 77: val_auc did not improve from 0.99497
416/416 ━━━━━━━━━━━━━━━━━━━━ 86s 178ms/step - accuracy: 0.9942 - auc: 0.9998 - loss: 0.2374 - val_accuracy: 0.9692 - val_auc: 0.9946 - val_loss: 0.2743 - learning_rate: 1.8750e-05
Epoch 78/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 156ms/step - accuracy: 0.9948 - auc: 0.9998 - loss: 0.2367
Epoch 78: val_auc did not improve from 0.99497
416/416 ━━━━━━━━━━━━━━━━━━━━ 78s 168ms/step - accuracy: 0.9943 - auc: 0.9998 - loss: 0.2372 - val_accuracy: 0.9674 - val_auc: 0.9946 - val_loss: 0.2750 - learning_rate: 1.8750e-05
Epoch 79/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9947 - auc: 0.9998 - loss: 0.2364
Epoch 79: val_auc did not improve from 0.99497
416/416 ━━━━━━━━━━━━━━━━━━━━ 81s 166ms/step - accuracy: 0.9949 - auc: 0.9998 - loss: 0.2366 - val_accuracy: 0.9690 - val_auc: 0.9949 - val_loss: 0.2735 - learning_rate: 1.8750e-05
Epoch 80/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.9956 - auc: 0.9999 - loss: 0.2358
Epoch 80: val_auc did not improve from 0.99497
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 167ms/step - accuracy: 0.9951 - auc: 0.9999 - loss: 0.2357 - val_accuracy: 0.9695 - val_auc: 0.9950 - val_loss: 0.2739 - learning_rate: 1.8750e-05
Epoch 81/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.9952 - auc: 0.9999 - loss: 0.2357
Epoch 81: val_auc improved from 0.99497 to 0.99514, saving model to exp10_cnn_bilstm_best.keras

Epoch 81: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 168ms/step - accuracy: 0.9950 - auc: 0.9999 - loss: 0.2364 - val_accuracy: 0.9710 - val_auc: 0.9951 - val_loss: 0.2719 - learning_rate: 1.8750e-05
Epoch 82/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.9941 - auc: 0.9998 - loss: 0.2363
Epoch 82: val_auc did not improve from 0.99514
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 167ms/step - accuracy: 0.9946 - auc: 0.9998 - loss: 0.2361 - val_accuracy: 0.9680 - val_auc: 0.9949 - val_loss: 0.2747 - learning_rate: 1.8750e-05
Epoch 83/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9951 - auc: 0.9998 - loss: 0.2361
Epoch 83: val_auc did not improve from 0.99514
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 167ms/step - accuracy: 0.9949 - auc: 0.9998 - loss: 0.2361 - val_accuracy: 0.9684 - val_auc: 0.9950 - val_loss: 0.2722 - learning_rate: 1.8750e-05
Epoch 84/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9955 - auc: 0.9999 - loss: 0.2354
Epoch 84: val_auc improved from 0.99514 to 0.99519, saving model to exp10_cnn_bilstm_best.keras

Epoch 84: finished saving model to exp10_cnn_bilstm_best.keras
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 167ms/step - accuracy: 0.9954 - auc: 0.9999 - loss: 0.2355 - val_accuracy: 0.9698 - val_auc: 0.9952 - val_loss: 0.2712 - learning_rate: 1.8750e-05
Epoch 85/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9953 - auc: 0.9999 - loss: 0.2356
Epoch 85: val_auc did not improve from 0.99519
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 166ms/step - accuracy: 0.9947 - auc: 0.9998 - loss: 0.2361 - val_accuracy: 0.9701 - val_auc: 0.9951 - val_loss: 0.2724 - learning_rate: 1.8750e-05
Epoch 86/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.9953 - auc: 0.9999 - loss: 0.2350
Epoch 86: ReduceLROnPlateau reducing learning rate to 9.375000445288606e-06.

Epoch 86: val_auc did not improve from 0.99519
416/416 ━━━━━━━━━━━━━━━━━━━━ 69s 167ms/step - accuracy: 0.9951 - auc: 0.9999 - loss: 0.2352 - val_accuracy: 0.9693 - val_auc: 0.9949 - val_loss: 0.2735 - learning_rate: 1.8750e-05
Epoch 87/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9949 - auc: 0.9998 - loss: 0.2348
Epoch 87: val_auc did not improve from 0.99519
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 166ms/step - accuracy: 0.9951 - auc: 0.9998 - loss: 0.2347 - val_accuracy: 0.9695 - val_auc: 0.9949 - val_loss: 0.2719 - learning_rate: 9.3750e-06
Epoch 88/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.9960 - auc: 0.9999 - loss: 0.2338
Epoch 88: val_auc did not improve from 0.99519
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 167ms/step - accuracy: 0.9955 - auc: 0.9999 - loss: 0.2344 - val_accuracy: 0.9698 - val_auc: 0.9947 - val_loss: 0.2741 - learning_rate: 9.3750e-06
Epoch 89/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9954 - auc: 0.9999 - loss: 0.2341
Epoch 89: val_auc did not improve from 0.99519
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 166ms/step - accuracy: 0.9949 - auc: 0.9999 - loss: 0.2349 - val_accuracy: 0.9699 - val_auc: 0.9949 - val_loss: 0.2724 - learning_rate: 9.3750e-06
Epoch 90/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.9957 - auc: 0.9999 - loss: 0.2336
Epoch 90: val_auc did not improve from 0.99519
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 168ms/step - accuracy: 0.9958 - auc: 0.9999 - loss: 0.2338 - val_accuracy: 0.9693 - val_auc: 0.9949 - val_loss: 0.2726 - learning_rate: 9.3750e-06
Epoch 91/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9952 - auc: 0.9999 - loss: 0.2342
Epoch 91: ReduceLROnPlateau reducing learning rate to 4.687500222644303e-06.

Epoch 91: val_auc did not improve from 0.99519
416/416 ━━━━━━━━━━━━━━━━━━━━ 70s 167ms/step - accuracy: 0.9958 - auc: 0.9999 - loss: 0.2339 - val_accuracy: 0.9696 - val_auc: 0.9948 - val_loss: 0.2721 - learning_rate: 9.3750e-06
Epoch 92/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.9943 - auc: 0.9998 - loss: 0.2354
Epoch 92: val_auc did not improve from 0.99519
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 167ms/step - accuracy: 0.9955 - auc: 0.9999 - loss: 0.2343 - val_accuracy: 0.9695 - val_auc: 0.9949 - val_loss: 0.2719 - learning_rate: 4.6875e-06
Epoch 93/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9952 - auc: 0.9999 - loss: 0.2345
Epoch 93: val_auc did not improve from 0.99519
416/416 ━━━━━━━━━━━━━━━━━━━━ 69s 167ms/step - accuracy: 0.9955 - auc: 0.9999 - loss: 0.2342 - val_accuracy: 0.9707 - val_auc: 0.9950 - val_loss: 0.2713 - learning_rate: 4.6875e-06
Epoch 94/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.9949 - auc: 0.9993 - loss: 0.2352
Epoch 94: val_auc did not improve from 0.99519
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 167ms/step - accuracy: 0.9950 - auc: 0.9998 - loss: 0.2345 - val_accuracy: 0.9699 - val_auc: 0.9951 - val_loss: 0.2712 - learning_rate: 4.6875e-06
Epoch 95/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9948 - auc: 0.9999 - loss: 0.2347
Epoch 95: val_auc did not improve from 0.99519
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 166ms/step - accuracy: 0.9949 - auc: 0.9999 - loss: 0.2345 - val_accuracy: 0.9702 - val_auc: 0.9950 - val_loss: 0.2717 - learning_rate: 4.6875e-06
Epoch 96/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 155ms/step - accuracy: 0.9954 - auc: 0.9999 - loss: 0.2337
Epoch 96: ReduceLROnPlateau reducing learning rate to 2.3437501113221515e-06.

Epoch 96: val_auc did not improve from 0.99519
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 167ms/step - accuracy: 0.9954 - auc: 0.9999 - loss: 0.2334 - val_accuracy: 0.9701 - val_auc: 0.9949 - val_loss: 0.2705 - learning_rate: 4.6875e-06
Epoch 97/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9952 - auc: 0.9999 - loss: 0.2339
Epoch 97: val_auc did not improve from 0.99519
416/416 ━━━━━━━━━━━━━━━━━━━━ 69s 166ms/step - accuracy: 0.9952 - auc: 0.9999 - loss: 0.2338 - val_accuracy: 0.9699 - val_auc: 0.9949 - val_loss: 0.2715 - learning_rate: 2.3438e-06
Epoch 98/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9951 - auc: 0.9999 - loss: 0.2337
Epoch 98: val_auc did not improve from 0.99519
416/416 ━━━━━━━━━━━━━━━━━━━━ 69s 167ms/step - accuracy: 0.9954 - auc: 0.9999 - loss: 0.2337 - val_accuracy: 0.9699 - val_auc: 0.9949 - val_loss: 0.2720 - learning_rate: 2.3438e-06
Epoch 99/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step - accuracy: 0.9954 - auc: 0.9999 - loss: 0.2341
Epoch 99: val_auc did not improve from 0.99519
416/416 ━━━━━━━━━━━━━━━━━━━━ 82s 167ms/step - accuracy: 0.9955 - auc: 0.9999 - loss: 0.2337 - val_accuracy: 0.9702 - val_auc: 0.9949 - val_loss: 0.2713 - learning_rate: 2.3438e-06
Epoch 99: early stopping
Restoring model weights from the end of the best epoch: 84.
"""

log_exp11 = r"""
Epoch 1/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 206s 454ms/step - accuracy: 0.6468 - auc: 0.7065 - loss: 0.7935 - val_accuracy: 0.7655 - val_auc: 0.8574 - val_loss: 0.6277 - learning_rate: 3.0000e-04
Epoch 2/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 177s 426ms/step - accuracy: 0.7794 - auc: 0.8686 - loss: 0.6066 - val_accuracy: 0.7844 - val_auc: 0.8733 - val_loss: 0.5922 - learning_rate: 3.0000e-04
Epoch 3/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 172s 415ms/step - accuracy: 0.8329 - auc: 0.9172 - loss: 0.5085 - val_accuracy: 0.7978 - val_auc: 0.9216 - val_loss: 0.5700 - learning_rate: 3.0000e-04
Epoch 4/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 172s 414ms/step - accuracy: 0.8643 - auc: 0.9412 - loss: 0.4455 - val_accuracy: 0.8747 - val_auc: 0.9474 - val_loss: 0.4245 - learning_rate: 3.0000e-04
Epoch 5/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 172s 414ms/step - accuracy: 0.8937 - auc: 0.9624 - loss: 0.3802 - val_accuracy: 0.8758 - val_auc: 0.9501 - val_loss: 0.4238 - learning_rate: 3.0000e-04
Epoch 6/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 173s 417ms/step - accuracy: 0.8988 - auc: 0.9676 - loss: 0.3569 - val_accuracy: 0.8970 - val_auc: 0.9645 - val_loss: 0.3649 - learning_rate: 3.0000e-04
Epoch 7/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 175s 420ms/step - accuracy: 0.9238 - auc: 0.9794 - loss: 0.3070 - val_accuracy: 0.9052 - val_auc: 0.9677 - val_loss: 0.3585 - learning_rate: 3.0000e-04
Epoch 8/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 173s 416ms/step - accuracy: 0.9316 - auc: 0.9833 - loss: 0.2852 - val_accuracy: 0.8736 - val_auc: 0.9644 - val_loss: 0.4386 - learning_rate: 3.0000e-04
Epoch 9/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 176s 424ms/step - accuracy: 0.9417 - auc: 0.9874 - loss: 0.2594 - val_accuracy: 0.8855 - val_auc: 0.9515 - val_loss: 0.4225 - learning_rate: 3.0000e-04
Epoch 10/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 177s 426ms/step - accuracy: 0.9411 - auc: 0.9869 - loss: 0.2585 - val_accuracy: 0.8473 - val_auc: 0.9346 - val_loss: 0.6208 - learning_rate: 3.0000e-04
Epoch 11/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 179s 431ms/step - accuracy: 0.9504 - auc: 0.9905 - loss: 0.2341 - val_accuracy: 0.9225 - val_auc: 0.9748 - val_loss: 0.3243 - learning_rate: 3.0000e-04
Epoch 12/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 193s 464ms/step - accuracy: 0.9586 - auc: 0.9935 - loss: 0.2098 - val_accuracy: 0.8946 - val_auc: 0.9660 - val_loss: 0.4117 - learning_rate: 3.0000e-04
Epoch 13/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 195s 469ms/step - accuracy: 0.9654 - auc: 0.9951 - loss: 0.1948 - val_accuracy: 0.9152 - val_auc: 0.9714 - val_loss: 0.3251 - learning_rate: 3.0000e-04
Epoch 14/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 196s 471ms/step - accuracy: 0.9612 - auc: 0.9938 - loss: 0.2020 - val_accuracy: 0.9326 - val_auc: 0.9779 - val_loss: 0.2970 - learning_rate: 3.0000e-04
Epoch 15/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 198s 476ms/step - accuracy: 0.9650 - auc: 0.9949 - loss: 0.1891 - val_accuracy: 0.8894 - val_auc: 0.9647 - val_loss: 0.4477 - learning_rate: 3.0000e-04
Epoch 16/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 198s 476ms/step - accuracy: 0.9719 - auc: 0.9964 - loss: 0.1726 - val_accuracy: 0.9210 - val_auc: 0.9749 - val_loss: 0.3291 - learning_rate: 3.0000e-04
Epoch 17/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 199s 479ms/step - accuracy: 0.9717 - auc: 0.9969 - loss: 0.1668 - val_accuracy: 0.9216 - val_auc: 0.9736 - val_loss: 0.3280 - learning_rate: 3.0000e-04
Epoch 18/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 203s 488ms/step - accuracy: 0.9751 - auc: 0.9971 - loss: 0.1620 - val_accuracy: 0.9356 - val_auc: 0.9807 - val_loss: 0.2893 - learning_rate: 3.0000e-04
Epoch 19/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 204s 491ms/step - accuracy: 0.9753 - auc: 0.9968 - loss: 0.1596 - val_accuracy: 0.9188 - val_auc: 0.9697 - val_loss: 0.3590 - learning_rate: 3.0000e-04
Epoch 20/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 203s 489ms/step - accuracy: 0.9746 - auc: 0.9966 - loss: 0.1603 - val_accuracy: 0.9424 - val_auc: 0.9835 - val_loss: 0.2604 - learning_rate: 3.0000e-04
Epoch 21/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 204s 491ms/step - accuracy: 0.9748 - auc: 0.9971 - loss: 0.1550 - val_accuracy: 0.8413 - val_auc: 0.9416 - val_loss: 0.6694 - learning_rate: 3.0000e-04
Epoch 22/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 205s 494ms/step - accuracy: 0.9754 - auc: 0.9974 - loss: 0.1517 - val_accuracy: 0.9322 - val_auc: 0.9767 - val_loss: 0.3037 - learning_rate: 3.0000e-04
Epoch 23/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 212s 510ms/step - accuracy: 0.9751 - auc: 0.9972 - loss: 0.1519 - val_accuracy: 0.9189 - val_auc: 0.9723 - val_loss: 0.3460 - learning_rate: 3.0000e-04
Epoch 24/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 207s 498ms/step - accuracy: 0.9762 - auc: 0.9976 - loss: 0.1472 - val_accuracy: 0.9317 - val_auc: 0.9778 - val_loss: 0.3034 - learning_rate: 3.0000e-04
Epoch 25/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 206s 495ms/step - accuracy: 0.9873 - auc: 0.9992 - loss: 0.1209 - val_accuracy: 0.9436 - val_auc: 0.9835 - val_loss: 0.2840 - learning_rate: 1.5000e-04
Epoch 26/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 204s 490ms/step - accuracy: 0.9909 - auc: 0.9996 - loss: 0.1086 - val_accuracy: 0.9301 - val_auc: 0.9738 - val_loss: 0.3492 - learning_rate: 1.5000e-04
Epoch 27/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 208s 500ms/step - accuracy: 0.9910 - auc: 0.9997 - loss: 0.1046 - val_accuracy: 0.9412 - val_auc: 0.9790 - val_loss: 0.3081 - learning_rate: 1.5000e-04
Epoch 28/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 208s 499ms/step - accuracy: 0.9916 - auc: 0.9997 - loss: 0.1013 - val_accuracy: 0.9412 - val_auc: 0.9787 - val_loss: 0.3032 - learning_rate: 1.5000e-04
Epoch 29/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 208s 499ms/step - accuracy: 0.9893 - auc: 0.9994 - loss: 0.1051 - val_accuracy: 0.9430 - val_auc: 0.9762 - val_loss: 0.3065 - learning_rate: 1.5000e-04
Epoch 30/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 211s 506ms/step - accuracy: 0.9925 - auc: 0.9998 - loss: 0.0970 - val_accuracy: 0.9477 - val_auc: 0.9811 - val_loss: 0.2819 - learning_rate: 7.5000e-05
Epoch 31/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 214s 514ms/step - accuracy: 0.9945 - auc: 0.9999 - loss: 0.0901 - val_accuracy: 0.9580 - val_auc: 0.9851 - val_loss: 0.2451 - learning_rate: 7.5000e-05
Epoch 32/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 212s 508ms/step - accuracy: 0.9948 - auc: 0.9997 - loss: 0.0877 - val_accuracy: 0.9535 - val_auc: 0.9823 - val_loss: 0.2675 - learning_rate: 7.5000e-05
Epoch 33/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 212s 510ms/step - accuracy: 0.9956 - auc: 0.9999 - loss: 0.0859 - val_accuracy: 0.9471 - val_auc: 0.9794 - val_loss: 0.3054 - learning_rate: 7.5000e-05
Epoch 34/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 213s 512ms/step - accuracy: 0.9955 - auc: 0.9999 - loss: 0.0845 - val_accuracy: 0.9236 - val_auc: 0.9658 - val_loss: 0.4315 - learning_rate: 7.5000e-05
Epoch 35/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 213s 513ms/step - accuracy: 0.9945 - auc: 0.9999 - loss: 0.0849 - val_accuracy: 0.9568 - val_auc: 0.9825 - val_loss: 0.2679 - learning_rate: 7.5000e-05
Epoch 36/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 210s 505ms/step - accuracy: 0.9966 - auc: 0.9999 - loss: 0.0795 - val_accuracy: 0.9558 - val_auc: 0.9827 - val_loss: 0.2696 - learning_rate: 3.7500e-05
Epoch 37/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 211s 508ms/step - accuracy: 0.9968 - auc: 0.9999 - loss: 0.0777 - val_accuracy: 0.9564 - val_auc: 0.9834 - val_loss: 0.2619 - learning_rate: 3.7500e-05
Epoch 38/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 212s 511ms/step - accuracy: 0.9972 - auc: 1.0000 - loss: 0.0765 - val_accuracy: 0.9598 - val_auc: 0.9829 - val_loss: 0.2652 - learning_rate: 3.7500e-05
Epoch 39/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 212s 510ms/step - accuracy: 0.9964 - auc: 1.0000 - loss: 0.0767 - val_accuracy: 0.9517 - val_auc: 0.9796 - val_loss: 0.2949 - learning_rate: 3.7500e-05
Epoch 40/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 212s 511ms/step - accuracy: 0.9973 - auc: 1.0000 - loss: 0.0750 - val_accuracy: 0.9546 - val_auc: 0.9818 - val_loss: 0.2739 - learning_rate: 3.7500e-05
Epoch 41/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 214s 514ms/step - accuracy: 0.9961 - auc: 0.9999 - loss: 0.0759 - val_accuracy: 0.9525 - val_auc: 0.9804 - val_loss: 0.2853 - learning_rate: 3.7500e-05
Epoch 42/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 214s 515ms/step - accuracy: 0.9969 - auc: 1.0000 - loss: 0.0737 - val_accuracy: 0.9600 - val_auc: 0.9839 - val_loss: 0.2533 - learning_rate: 3.7500e-05
Epoch 43/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 215s 517ms/step - accuracy: 0.9975 - auc: 1.0000 - loss: 0.0718 - val_accuracy: 0.9461 - val_auc: 0.9767 - val_loss: 0.3237 - learning_rate: 3.7500e-05
Epoch 44/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 217s 521ms/step - accuracy: 0.9968 - auc: 1.0000 - loss: 0.0721 - val_accuracy: 0.9532 - val_auc: 0.9809 - val_loss: 0.2842 - learning_rate: 3.7500e-05
Epoch 45/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 218s 524ms/step - accuracy: 0.9970 - auc: 0.9999 - loss: 0.0719 - val_accuracy: 0.9514 - val_auc: 0.9785 - val_loss: 0.2981 - learning_rate: 3.7500e-05
Epoch 46/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 216s 520ms/step - accuracy: 0.9970 - auc: 1.0000 - loss: 0.0705 - val_accuracy: 0.9492 - val_auc: 0.9781 - val_loss: 0.3035 - learning_rate: 3.7500e-05
Epoch 47/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 270s 538ms/step - accuracy: 0.9970 - auc: 1.0000 - loss: 0.0702 - val_accuracy: 0.9543 - val_auc: 0.9809 - val_loss: 0.2790 - learning_rate: 1.8750e-05
Epoch 48/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 232s 557ms/step - accuracy: 0.9972 - auc: 1.0000 - loss: 0.0686 - val_accuracy: 0.9541 - val_auc: 0.9808 - val_loss: 0.2770 - learning_rate: 1.8750e-05
Epoch 49/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 217s 522ms/step - accuracy: 0.9973 - auc: 1.0000 - loss: 0.0674 - val_accuracy: 0.9580 - val_auc: 0.9830 - val_loss: 0.2635 - learning_rate: 1.8750e-05
Epoch 50/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 218s 525ms/step - accuracy: 0.9976 - auc: 1.0000 - loss: 0.0668 - val_accuracy: 0.9570 - val_auc: 0.9819 - val_loss: 0.2640 - learning_rate: 1.8750e-05
Epoch 51/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 220s 529ms/step - accuracy: 0.9979 - auc: 0.9999 - loss: 0.0664 - val_accuracy: 0.9564 - val_auc: 0.9823 - val_loss: 0.2710 - learning_rate: 9.3750e-06
Epoch 52/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 220s 528ms/step - accuracy: 0.9980 - auc: 1.0000 - loss: 0.0659 - val_accuracy: 0.9564 - val_auc: 0.9821 - val_loss: 0.2731 - learning_rate: 9.3750e-06
Epoch 53/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 220s 530ms/step - accuracy: 0.9978 - auc: 1.0000 - loss: 0.0658 - val_accuracy: 0.9537 - val_auc: 0.9808 - val_loss: 0.2824 - learning_rate: 9.3750e-06
Epoch 54/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 245s 588ms/step - accuracy: 0.9984 - auc: 1.0000 - loss: 0.0647 - val_accuracy: 0.9580 - val_auc: 0.9827 - val_loss: 0.2618 - learning_rate: 9.3750e-06
Epoch 55/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 224s 539ms/step - accuracy: 0.9978 - auc: 1.0000 - loss: 0.0658 - val_accuracy: 0.9552 - val_auc: 0.9818 - val_loss: 0.2758 - learning_rate: 4.6875e-06
Epoch 56/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 247s 595ms/step - accuracy: 0.9981 - auc: 1.0000 - loss: 0.0647 - val_accuracy: 0.9565 - val_auc: 0.9824 - val_loss: 0.2673 - learning_rate: 4.6875e-06
Epoch 57/150
416/416 ━━━━━━━━━━━━━━━━━━━━ 221s 532ms/step - accuracy: 0.9975 - auc: 1.0000 - loss: 0.0652 - val_accuracy: 0.9580 - val_auc: 0.9835 - val_loss: 0.2600 - learning_rate: 4.6875e-06
"""

log_exp12 = r"""
Epoch 1/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 253s 566ms/step - accuracy: 0.6944 - auc: 0.7706 - loss: 0.7391 - val_accuracy: 0.6742 - val_auc: 0.8325 - val_loss: 0.7305 - learning_rate: 3.0000e-04
Epoch 2/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 182s 467ms/step - accuracy: 0.7880 - auc: 0.8729 - loss: 0.5965 - val_accuracy: 0.8069 - val_auc: 0.8871 - val_loss: 0.5608 - learning_rate: 3.0000e-04
Epoch 3/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 172s 442ms/step - accuracy: 0.8321 - auc: 0.9139 - loss: 0.5139 - val_accuracy: 0.8307 - val_auc: 0.9172 - val_loss: 0.5142 - learning_rate: 3.0000e-04
Epoch 4/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 171s 439ms/step - accuracy: 0.8603 - auc: 0.9382 - loss: 0.4520 - val_accuracy: 0.8720 - val_auc: 0.9442 - val_loss: 0.4378 - learning_rate: 3.0000e-04
Epoch 5/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 168s 430ms/step - accuracy: 0.8839 - auc: 0.9563 - loss: 0.3976 - val_accuracy: 0.8724 - val_auc: 0.9543 - val_loss: 0.4135 - learning_rate: 3.0000e-04
Epoch 6/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 178s 455ms/step - accuracy: 0.9045 - auc: 0.9685 - loss: 0.3540 - val_accuracy: 0.8626 - val_auc: 0.9512 - val_loss: 0.4792 - learning_rate: 3.0000e-04
Epoch 7/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 176s 452ms/step - accuracy: 0.9162 - auc: 0.9749 - loss: 0.3247 - val_accuracy: 0.8943 - val_auc: 0.9640 - val_loss: 0.3668 - learning_rate: 3.0000e-04
Epoch 8/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 181s 465ms/step - accuracy: 0.9256 - auc: 0.9799 - loss: 0.3005 - val_accuracy: 0.8988 - val_auc: 0.9648 - val_loss: 0.3627 - learning_rate: 3.0000e-04
Epoch 9/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 181s 464ms/step - accuracy: 0.9341 - auc: 0.9839 - loss: 0.2777 - val_accuracy: 0.8859 - val_auc: 0.9631 - val_loss: 0.4169 - learning_rate: 3.0000e-04
Epoch 10/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 184s 472ms/step - accuracy: 0.9404 - auc: 0.9866 - loss: 0.2599 - val_accuracy: 0.9066 - val_auc: 0.9694 - val_loss: 0.3455 - learning_rate: 3.0000e-04
Epoch 11/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 206s 482ms/step - accuracy: 0.9487 - auc: 0.9896 - loss: 0.2398 - val_accuracy: 0.8923 - val_auc: 0.9619 - val_loss: 0.3970 - learning_rate: 3.0000e-04
Epoch 12/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 184s 471ms/step - accuracy: 0.9527 - auc: 0.9917 - loss: 0.2239 - val_accuracy: 0.9136 - val_auc: 0.9706 - val_loss: 0.3394 - learning_rate: 3.0000e-04
Epoch 13/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 199s 464ms/step - accuracy: 0.9552 - auc: 0.9922 - loss: 0.2177 - val_accuracy: 0.9194 - val_auc: 0.9760 - val_loss: 0.3253 - learning_rate: 3.0000e-04
Epoch 14/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 180s 461ms/step - accuracy: 0.9601 - auc: 0.9933 - loss: 0.2059 - val_accuracy: 0.9271 - val_auc: 0.9769 - val_loss: 0.3037 - learning_rate: 3.0000e-04
Epoch 15/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 180s 462ms/step - accuracy: 0.9641 - auc: 0.9944 - loss: 0.1941 - val_accuracy: 0.9223 - val_auc: 0.9749 - val_loss: 0.3286 - learning_rate: 3.0000e-04
Epoch 16/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 180s 461ms/step - accuracy: 0.9672 - auc: 0.9950 - loss: 0.1866 - val_accuracy: 0.8901 - val_auc: 0.9596 - val_loss: 0.4454 - learning_rate: 3.0000e-04
Epoch 17/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 201s 458ms/step - accuracy: 0.9656 - auc: 0.9954 - loss: 0.1838 - val_accuracy: 0.9131 - val_auc: 0.9701 - val_loss: 0.3631 - learning_rate: 3.0000e-04
Epoch 18/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 178s 455ms/step - accuracy: 0.9795 - auc: 0.9984 - loss: 0.1473 - val_accuracy: 0.9418 - val_auc: 0.9821 - val_loss: 0.2811 - learning_rate: 1.5000e-04
Epoch 19/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 181s 464ms/step - accuracy: 0.9837 - auc: 0.9988 - loss: 0.1365 - val_accuracy: 0.9395 - val_auc: 0.9803 - val_loss: 0.3004 - learning_rate: 1.5000e-04
Epoch 20/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 184s 472ms/step - accuracy: 0.9832 - auc: 0.9988 - loss: 0.1350 - val_accuracy: 0.9402 - val_auc: 0.9796 - val_loss: 0.3005 - learning_rate: 1.5000e-04
Epoch 21/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 186s 476ms/step - accuracy: 0.9838 - auc: 0.9990 - loss: 0.1311 - val_accuracy: 0.9239 - val_auc: 0.9752 - val_loss: 0.3809 - learning_rate: 1.5000e-04
Epoch 22/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 180s 462ms/step - accuracy: 0.9892 - auc: 0.9994 - loss: 0.1195 - val_accuracy: 0.9507 - val_auc: 0.9825 - val_loss: 0.2754 - learning_rate: 7.5000e-05
Epoch 23/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 173s 443ms/step - accuracy: 0.9911 - auc: 0.9997 - loss: 0.1126 - val_accuracy: 0.9503 - val_auc: 0.9822 - val_loss: 0.2852 - learning_rate: 7.5000e-05
Epoch 24/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 174s 445ms/step - accuracy: 0.9909 - auc: 0.9996 - loss: 0.1109 - val_accuracy: 0.9387 - val_auc: 0.9759 - val_loss: 0.3363 - learning_rate: 7.5000e-05
Epoch 25/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 171s 440ms/step - accuracy: 0.9908 - auc: 0.9995 - loss: 0.1115 - val_accuracy: 0.9436 - val_auc: 0.9804 - val_loss: 0.3005 - learning_rate: 7.5000e-05
Epoch 26/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 173s 444ms/step - accuracy: 0.9933 - auc: 0.9998 - loss: 0.1030 - val_accuracy: 0.9486 - val_auc: 0.9812 - val_loss: 0.2947 - learning_rate: 3.7500e-05
Epoch 27/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 171s 439ms/step - accuracy: 0.9930 - auc: 0.9998 - loss: 0.1024 - val_accuracy: 0.9477 - val_auc: 0.9823 - val_loss: 0.2962 - learning_rate: 3.7500e-05
Epoch 28/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 172s 442ms/step - accuracy: 0.9942 - auc: 0.9998 - loss: 0.0994 - val_accuracy: 0.9504 - val_auc: 0.9821 - val_loss: 0.2880 - learning_rate: 3.7500e-05
Epoch 29/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 176s 451ms/step - accuracy: 0.9948 - auc: 0.9998 - loss: 0.0978 - val_accuracy: 0.9502 - val_auc: 0.9819 - val_loss: 0.2948 - learning_rate: 1.8750e-05
Epoch 30/150
390/390 ━━━━━━━━━━━━━━━━━━━━ 175s 448ms/step - accuracy: 0.9947 - auc: 0.9999 - loss: 0.0980 - val_accuracy: 0.9460 - val_auc: 0.9814 - val_loss: 0.3048 - learning_rate: 1.8750e-05
"""

log_exp13 = r"""
Epoch 1/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 219s 267ms/step - accuracy: 0.6886 - auc: 0.7640 - loss: 0.7399 - val_accuracy: 0.7527 - val_auc: 0.8417 - val_loss: 0.6521 - learning_rate: 3.0000e-04
Epoch 2/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 173s 238ms/step - accuracy: 0.7811 - auc: 0.8660 - loss: 0.5998 - val_accuracy: 0.8161 - val_auc: 0.9046 - val_loss: 0.5274 - learning_rate: 3.0000e-04
Epoch 3/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 169s 232ms/step - accuracy: 0.8231 - auc: 0.9091 - loss: 0.5144 - val_accuracy: 0.8339 - val_auc: 0.9204 - val_loss: 0.4966 - learning_rate: 3.0000e-04
Epoch 4/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 185s 255ms/step - accuracy: 0.8487 - auc: 0.9308 - loss: 0.4594 - val_accuracy: 0.8415 - val_auc: 0.9273 - val_loss: 0.4628 - learning_rate: 3.0000e-04
Epoch 5/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 183s 252ms/step - accuracy: 0.8718 - auc: 0.9483 - loss: 0.4084 - val_accuracy: 0.8453 - val_auc: 0.9355 - val_loss: 0.4517 - learning_rate: 3.0000e-04
Epoch 6/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 182s 250ms/step - accuracy: 0.8918 - auc: 0.9612 - loss: 0.3652 - val_accuracy: 0.8834 - val_auc: 0.9562 - val_loss: 0.3805 - learning_rate: 3.0000e-04
Epoch 7/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 193s 238ms/step - accuracy: 0.9048 - auc: 0.9691 - loss: 0.3340 - val_accuracy: 0.8633 - val_auc: 0.9457 - val_loss: 0.4215 - learning_rate: 3.0000e-04
Epoch 8/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 175s 241ms/step - accuracy: 0.9131 - auc: 0.9739 - loss: 0.3130 - val_accuracy: 0.8918 - val_auc: 0.9629 - val_loss: 0.3626 - learning_rate: 3.0000e-04
Epoch 9/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 174s 239ms/step - accuracy: 0.9250 - auc: 0.9800 - loss: 0.2842 - val_accuracy: 0.8556 - val_auc: 0.9388 - val_loss: 0.4600 - learning_rate: 3.0000e-04
Epoch 10/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 175s 240ms/step - accuracy: 0.9311 - auc: 0.9832 - loss: 0.2665 - val_accuracy: 0.8983 - val_auc: 0.9698 - val_loss: 0.3553 - learning_rate: 3.0000e-04
Epoch 11/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 171s 234ms/step - accuracy: 0.9404 - auc: 0.9869 - loss: 0.2450 - val_accuracy: 0.8868 - val_auc: 0.9583 - val_loss: 0.4285 - learning_rate: 3.0000e-04
Epoch 12/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 174s 239ms/step - accuracy: 0.9459 - auc: 0.9890 - loss: 0.2307 - val_accuracy: 0.8945 - val_auc: 0.9577 - val_loss: 0.3921 - learning_rate: 3.0000e-04
Epoch 13/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 170s 233ms/step - accuracy: 0.9486 - auc: 0.9896 - loss: 0.2245 - val_accuracy: 0.9246 - val_auc: 0.9766 - val_loss: 0.2982 - learning_rate: 3.0000e-04
Epoch 14/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 204s 235ms/step - accuracy: 0.9536 - auc: 0.9913 - loss: 0.2116 - val_accuracy: 0.9156 - val_auc: 0.9709 - val_loss: 0.3255 - learning_rate: 3.0000e-04
Epoch 15/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 171s 235ms/step - accuracy: 0.9550 - auc: 0.9924 - loss: 0.2024 - val_accuracy: 0.9140 - val_auc: 0.9701 - val_loss: 0.3292 - learning_rate: 3.0000e-04
Epoch 16/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 182s 249ms/step - accuracy: 0.9592 - auc: 0.9931 - loss: 0.1943 - val_accuracy: 0.9056 - val_auc: 0.9724 - val_loss: 0.3514 - learning_rate: 3.0000e-04
Epoch 17/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 179s 246ms/step - accuracy: 0.9723 - auc: 0.9968 - loss: 0.1611 - val_accuracy: 0.9375 - val_auc: 0.9799 - val_loss: 0.2870 - learning_rate: 1.5000e-04
Epoch 18/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 190s 261ms/step - accuracy: 0.9773 - auc: 0.9978 - loss: 0.1459 - val_accuracy: 0.9223 - val_auc: 0.9758 - val_loss: 0.3338 - learning_rate: 1.5000e-04
Epoch 19/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 193s 264ms/step - accuracy: 0.9792 - auc: 0.9980 - loss: 0.1405 - val_accuracy: 0.9310 - val_auc: 0.9793 - val_loss: 0.3024 - learning_rate: 1.5000e-04
Epoch 20/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 200s 274ms/step - accuracy: 0.9808 - auc: 0.9984 - loss: 0.1337 - val_accuracy: 0.9402 - val_auc: 0.9800 - val_loss: 0.2813 - learning_rate: 1.5000e-04
Epoch 21/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 178s 245ms/step - accuracy: 0.9761 - auc: 0.9976 - loss: 0.1431 - val_accuracy: 0.9348 - val_auc: 0.9795 - val_loss: 0.2981 - learning_rate: 1.5000e-04
Epoch 22/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 242s 333ms/step - accuracy: 0.9833 - auc: 0.9988 - loss: 0.1242 - val_accuracy: 0.9443 - val_auc: 0.9807 - val_loss: 0.2806 - learning_rate: 1.5000e-04
Epoch 23/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 200s 274ms/step - accuracy: 0.9819 - auc: 0.9985 - loss: 0.1266 - val_accuracy: 0.9364 - val_auc: 0.9808 - val_loss: 0.2893 - learning_rate: 1.5000e-04
Epoch 24/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 197s 271ms/step - accuracy: 0.9817 - auc: 0.9985 - loss: 0.1265 - val_accuracy: 0.9398 - val_auc: 0.9823 - val_loss: 0.2824 - learning_rate: 1.5000e-04
Epoch 25/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 198s 272ms/step - accuracy: 0.9844 - auc: 0.9988 - loss: 0.1193 - val_accuracy: 0.9384 - val_auc: 0.9776 - val_loss: 0.2926 - learning_rate: 1.5000e-04
Epoch 26/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 200s 275ms/step - accuracy: 0.9894 - auc: 0.9994 - loss: 0.1059 - val_accuracy: 0.9461 - val_auc: 0.9816 - val_loss: 0.2712 - learning_rate: 7.5000e-05
Epoch 27/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 191s 262ms/step - accuracy: 0.9902 - auc: 0.9996 - loss: 0.1012 - val_accuracy: 0.9404 - val_auc: 0.9798 - val_loss: 0.2948 - learning_rate: 7.5000e-05
Epoch 28/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 189s 260ms/step - accuracy: 0.9906 - auc: 0.9997 - loss: 0.0982 - val_accuracy: 0.9467 - val_auc: 0.9807 - val_loss: 0.2807 - learning_rate: 7.5000e-05
Epoch 29/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 187s 257ms/step - accuracy: 0.9908 - auc: 0.9996 - loss: 0.0957 - val_accuracy: 0.9388 - val_auc: 0.9766 - val_loss: 0.3200 - learning_rate: 7.5000e-05
Epoch 30/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 190s 260ms/step - accuracy: 0.9907 - auc: 0.9997 - loss: 0.0953 - val_accuracy: 0.9478 - val_auc: 0.9801 - val_loss: 0.2847 - learning_rate: 7.5000e-05
Epoch 31/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 190s 261ms/step - accuracy: 0.9908 - auc: 0.9997 - loss: 0.0940 - val_accuracy: 0.9493 - val_auc: 0.9795 - val_loss: 0.2839 - learning_rate: 7.5000e-05
Epoch 32/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 193s 265ms/step - accuracy: 0.9919 - auc: 0.9997 - loss: 0.0905 - val_accuracy: 0.9469 - val_auc: 0.9793 - val_loss: 0.2828 - learning_rate: 7.5000e-05
Epoch 33/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 187s 257ms/step - accuracy: 0.9917 - auc: 0.9996 - loss: 0.0903 - val_accuracy: 0.9459 - val_auc: 0.9794 - val_loss: 0.2897 - learning_rate: 7.5000e-05
Epoch 34/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 189s 259ms/step - accuracy: 0.9914 - auc: 0.9997 - loss: 0.0887 - val_accuracy: 0.9441 - val_auc: 0.9749 - val_loss: 0.3100 - learning_rate: 7.5000e-05
Epoch 35/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 188s 258ms/step - accuracy: 0.9951 - auc: 0.9999 - loss: 0.0804 - val_accuracy: 0.9529 - val_auc: 0.9794 - val_loss: 0.2743 - learning_rate: 3.7500e-05
Epoch 36/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 188s 258ms/step - accuracy: 0.9946 - auc: 0.9999 - loss: 0.0800 - val_accuracy: 0.9465 - val_auc: 0.9781 - val_loss: 0.3005 - learning_rate: 3.7500e-05
Epoch 37/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 176s 241ms/step - accuracy: 0.9954 - auc: 0.9999 - loss: 0.0771 - val_accuracy: 0.9421 - val_auc: 0.9765 - val_loss: 0.3167 - learning_rate: 3.7500e-05
Epoch 38/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 174s 239ms/step - accuracy: 0.9949 - auc: 0.9999 - loss: 0.0786 - val_accuracy: 0.9513 - val_auc: 0.9782 - val_loss: 0.2893 - learning_rate: 3.7500e-05
Epoch 39/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 173s 237ms/step - accuracy: 0.9954 - auc: 0.9999 - loss: 0.0758 - val_accuracy: 0.9534 - val_auc: 0.9800 - val_loss: 0.2815 - learning_rate: 1.8750e-05
Epoch 40/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 178s 244ms/step - accuracy: 0.9958 - auc: 0.9999 - loss: 0.0748 - val_accuracy: 0.9519 - val_auc: 0.9785 - val_loss: 0.2910 - learning_rate: 1.8750e-05
Epoch 41/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 189s 260ms/step - accuracy: 0.9956 - auc: 0.9999 - loss: 0.0751 - val_accuracy: 0.9506 - val_auc: 0.9777 - val_loss: 0.2919 - learning_rate: 1.8750e-05
Epoch 42/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 194s 266ms/step - accuracy: 0.9957 - auc: 0.9999 - loss: 0.0735 - val_accuracy: 0.9517 - val_auc: 0.9788 - val_loss: 0.2823 - learning_rate: 1.8750e-05
Epoch 43/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 190s 261ms/step - accuracy: 0.9963 - auc: 1.0000 - loss: 0.0720 - val_accuracy: 0.9486 - val_auc: 0.9777 - val_loss: 0.2923 - learning_rate: 9.3750e-06
Epoch 44/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 191s 262ms/step - accuracy: 0.9963 - auc: 1.0000 - loss: 0.0717 - val_accuracy: 0.9532 - val_auc: 0.9784 - val_loss: 0.2826 - learning_rate: 9.3750e-06
Epoch 45/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 192s 263ms/step - accuracy: 0.9963 - auc: 1.0000 - loss: 0.0710 - val_accuracy: 0.9543 - val_auc: 0.9786 - val_loss: 0.2818 - learning_rate: 9.3750e-06
Epoch 46/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 189s 260ms/step - accuracy: 0.9965 - auc: 1.0000 - loss: 0.0706 - val_accuracy: 0.9507 - val_auc: 0.9775 - val_loss: 0.2907 - learning_rate: 9.3750e-06
Epoch 47/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 173s 238ms/step - accuracy: 0.9969 - auc: 0.9999 - loss: 0.0706 - val_accuracy: 0.9529 - val_auc: 0.9781 - val_loss: 0.2861 - learning_rate: 9.3750e-06
Epoch 48/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 177s 243ms/step - accuracy: 0.9967 - auc: 1.0000 - loss: 0.0703 - val_accuracy: 0.9499 - val_auc: 0.9769 - val_loss: 0.2996 - learning_rate: 9.3750e-06
Epoch 49/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 177s 242ms/step - accuracy: 0.9971 - auc: 1.0000 - loss: 0.0692 - val_accuracy: 0.9537 - val_auc: 0.9780 - val_loss: 0.2844 - learning_rate: 4.6875e-06
Epoch 50/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 175s 241ms/step - accuracy: 0.9970 - auc: 1.0000 - loss: 0.0693 - val_accuracy: 0.9507 - val_auc: 0.9769 - val_loss: 0.2958 - learning_rate: 4.6875e-06
Epoch 51/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 177s 243ms/step - accuracy: 0.9972 - auc: 1.0000 - loss: 0.0689 - val_accuracy: 0.9536 - val_auc: 0.9782 - val_loss: 0.2850 - learning_rate: 4.6875e-06
Epoch 52/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 177s 243ms/step - accuracy: 0.9973 - auc: 1.0000 - loss: 0.0687 - val_accuracy: 0.9528 - val_auc: 0.9778 - val_loss: 0.2900 - learning_rate: 2.3438e-06
Epoch 53/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 178s 244ms/step - accuracy: 0.9971 - auc: 1.0000 - loss: 0.0691 - val_accuracy: 0.9535 - val_auc: 0.9781 - val_loss: 0.2857 - learning_rate: 2.3438e-06
"""


log_exp14 = r"""
Epoch 1/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 446ms/step - accuracy: 0.6083 - auc: 0.6531 - loss: 0.8187 - pr_auc: 0.6423
Epoch 1: val_auc improved from None to 0.81840, saving model to best_benchmark04.keras

Epoch 1: finished saving model to best_benchmark04.keras
728/728 ━━━━━━━━━━━━━━━━━━━━ 375s 483ms/step - accuracy: 0.6594 - auc: 0.7240 - loss: 0.7700 - pr_auc: 0.7108 - val_accuracy: 0.7351 - val_auc: 0.8184 - val_loss: 0.6741 - val_pr_auc: 0.8006 - learning_rate: 5.0000e-04
Epoch 2/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 434ms/step - accuracy: 0.7338 - auc: 0.8118 - loss: 0.6657 - pr_auc: 0.7941
Epoch 2: val_auc improved from 0.81840 to 0.82725, saving model to best_benchmark04.keras

Epoch 2: finished saving model to best_benchmark04.keras
728/728 ━━━━━━━━━━━━━━━━━━━━ 342s 469ms/step - accuracy: 0.7406 - auc: 0.8200 - loss: 0.6496 - pr_auc: 0.8053 - val_accuracy: 0.7323 - val_auc: 0.8272 - val_loss: 0.6370 - val_pr_auc: 0.8223 - learning_rate: 5.0000e-04
Epoch 3/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 462ms/step - accuracy: 0.7825 - auc: 0.8620 - loss: 0.5836 - pr_auc: 0.8488
Epoch 3: val_auc improved from 0.82725 to 0.89416, saving model to best_benchmark04.keras

Epoch 3: finished saving model to best_benchmark04.keras
728/728 ━━━━━━━━━━━━━━━━━━━━ 362s 498ms/step - accuracy: 0.7859 - auc: 0.8648 - loss: 0.5761 - pr_auc: 0.8525 - val_accuracy: 0.8051 - val_auc: 0.8942 - val_loss: 0.5283 - val_pr_auc: 0.8864 - learning_rate: 5.0000e-04
Epoch 4/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 463ms/step - accuracy: 0.7978 - auc: 0.8818 - loss: 0.5413 - pr_auc: 0.8730
Epoch 4: val_auc did not improve from 0.89416
728/728 ━━━━━━━━━━━━━━━━━━━━ 383s 500ms/step - accuracy: 0.8034 - auc: 0.8867 - loss: 0.5334 - pr_auc: 0.8781 - val_accuracy: 0.7963 - val_auc: 0.8926 - val_loss: 0.5345 - val_pr_auc: 0.8890 - learning_rate: 5.0000e-04
Epoch 5/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 478ms/step - accuracy: 0.8202 - auc: 0.9028 - loss: 0.5048 - pr_auc: 0.8967
Epoch 5: val_auc improved from 0.89416 to 0.90485, saving model to best_benchmark04.keras

Epoch 5: finished saving model to best_benchmark04.keras
728/728 ━━━━━━━━━━━━━━━━━━━━ 374s 514ms/step - accuracy: 0.8245 - auc: 0.9070 - loss: 0.4975 - pr_auc: 0.9016 - val_accuracy: 0.8222 - val_auc: 0.9049 - val_loss: 0.4970 - val_pr_auc: 0.8938 - learning_rate: 5.0000e-04
Epoch 6/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 487ms/step - accuracy: 0.8409 - auc: 0.9198 - loss: 0.4731 - pr_auc: 0.9139
Epoch 6: val_auc improved from 0.90485 to 0.91848, saving model to best_benchmark04.keras

Epoch 6: finished saving model to best_benchmark04.keras
728/728 ━━━━━━━━━━━━━━━━━━━━ 389s 523ms/step - accuracy: 0.8427 - auc: 0.9212 - loss: 0.4703 - pr_auc: 0.9155 - val_accuracy: 0.8221 - val_auc: 0.9185 - val_loss: 0.5136 - val_pr_auc: 0.9127 - learning_rate: 5.0000e-04
Epoch 7/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 492ms/step - accuracy: 0.8589 - auc: 0.9347 - loss: 0.4445 - pr_auc: 0.9321
Epoch 7: val_auc did not improve from 0.91848
728/728 ━━━━━━━━━━━━━━━━━━━━ 384s 528ms/step - accuracy: 0.8596 - auc: 0.9356 - loss: 0.4421 - pr_auc: 0.9321 - val_accuracy: 0.7952 - val_auc: 0.9107 - val_loss: 0.5385 - val_pr_auc: 0.9094 - learning_rate: 5.0000e-04
Epoch 8/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 505ms/step - accuracy: 0.8708 - auc: 0.9456 - loss: 0.4204 - pr_auc: 0.9447
Epoch 8: val_auc did not improve from 0.91848
728/728 ━━━━━━━━━━━━━━━━━━━━ 392s 539ms/step - accuracy: 0.8674 - auc: 0.9430 - loss: 0.4257 - pr_auc: 0.9415 - val_accuracy: 0.8297 - val_auc: 0.9176 - val_loss: 0.4735 - val_pr_auc: 0.9137 - learning_rate: 5.0000e-04
Epoch 9/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 494ms/step - accuracy: 0.8809 - auc: 0.9506 - loss: 0.4100 - pr_auc: 0.9506
Epoch 9: val_auc improved from 0.91848 to 0.92465, saving model to best_benchmark04.keras

Epoch 9: finished saving model to best_benchmark04.keras
728/728 ━━━━━━━━━━━━━━━━━━━━ 386s 530ms/step - accuracy: 0.8817 - auc: 0.9522 - loss: 0.4065 - pr_auc: 0.9518 - val_accuracy: 0.8128 - val_auc: 0.9246 - val_loss: 0.5054 - val_pr_auc: 0.9246 - learning_rate: 5.0000e-04
Epoch 10/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 524ms/step - accuracy: 0.8897 - auc: 0.9582 - loss: 0.3921 - pr_auc: 0.9573
Epoch 10: val_auc improved from 0.92465 to 0.95420, saving model to best_benchmark04.keras

Epoch 10: finished saving model to best_benchmark04.keras
728/728 ━━━━━━━━━━━━━━━━━━━━ 409s 561ms/step - accuracy: 0.8891 - auc: 0.9587 - loss: 0.3914 - pr_auc: 0.9583 - val_accuracy: 0.8803 - val_auc: 0.9542 - val_loss: 0.4087 - val_pr_auc: 0.9536 - learning_rate: 5.0000e-04
Epoch 11/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 528ms/step - accuracy: 0.9022 - auc: 0.9655 - loss: 0.3740 - pr_auc: 0.9645
Epoch 11: val_auc did not improve from 0.95420
728/728 ━━━━━━━━━━━━━━━━━━━━ 446s 567ms/step - accuracy: 0.9010 - auc: 0.9643 - loss: 0.3770 - pr_auc: 0.9639 - val_accuracy: 0.8578 - val_auc: 0.9486 - val_loss: 0.4457 - val_pr_auc: 0.9481 - learning_rate: 5.0000e-04
Epoch 12/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 523ms/step - accuracy: 0.9058 - auc: 0.9682 - loss: 0.3671 - pr_auc: 0.9677
Epoch 12: val_auc improved from 0.95420 to 0.96300, saving model to best_benchmark04.keras

Epoch 12: finished saving model to best_benchmark04.keras
728/728 ━━━━━━━━━━━━━━━━━━━━ 408s 561ms/step - accuracy: 0.9093 - auc: 0.9697 - loss: 0.3626 - pr_auc: 0.9696 - val_accuracy: 0.8983 - val_auc: 0.9630 - val_loss: 0.3791 - val_pr_auc: 0.9638 - learning_rate: 5.0000e-04
Epoch 13/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 509ms/step - accuracy: 0.9140 - auc: 0.9730 - loss: 0.3532 - pr_auc: 0.9722
Epoch 13: val_auc improved from 0.96300 to 0.96490, saving model to best_benchmark04.keras

Epoch 13: finished saving model to best_benchmark04.keras
728/728 ━━━━━━━━━━━━━━━━━━━━ 445s 566ms/step - accuracy: 0.9116 - auc: 0.9719 - loss: 0.3563 - pr_auc: 0.9716 - val_accuracy: 0.8972 - val_auc: 0.9649 - val_loss: 0.3889 - val_pr_auc: 0.9669 - learning_rate: 5.0000e-04
Epoch 14/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 513ms/step - accuracy: 0.9239 - auc: 0.9776 - loss: 0.3388 - pr_auc: 0.9768
Epoch 14: val_auc did not improve from 0.96490
728/728 ━━━━━━━━━━━━━━━━━━━━ 432s 551ms/step - accuracy: 0.9227 - auc: 0.9764 - loss: 0.3421 - pr_auc: 0.9761 - val_accuracy: 0.8957 - val_auc: 0.9629 - val_loss: 0.3898 - val_pr_auc: 0.9631 - learning_rate: 5.0000e-04
Epoch 15/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 510ms/step - accuracy: 0.9271 - auc: 0.9803 - loss: 0.3298 - pr_auc: 0.9807
Epoch 15: val_auc improved from 0.96490 to 0.96690, saving model to best_benchmark04.keras

Epoch 15: finished saving model to best_benchmark04.keras
728/728 ━━━━━━━━━━━━━━━━━━━━ 401s 551ms/step - accuracy: 0.9233 - auc: 0.9777 - loss: 0.3376 - pr_auc: 0.9778 - val_accuracy: 0.9019 - val_auc: 0.9669 - val_loss: 0.3687 - val_pr_auc: 0.9680 - learning_rate: 5.0000e-04
Epoch 16/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 521ms/step - accuracy: 0.9345 - auc: 0.9826 - loss: 0.3218 - pr_auc: 0.9825
Epoch 16: val_auc did not improve from 0.96690
728/728 ━━━━━━━━━━━━━━━━━━━━ 450s 562ms/step - accuracy: 0.9312 - auc: 0.9812 - loss: 0.3267 - pr_auc: 0.9812 - val_accuracy: 0.9025 - val_auc: 0.9669 - val_loss: 0.3797 - val_pr_auc: 0.9673 - learning_rate: 5.0000e-04
Epoch 17/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 522ms/step - accuracy: 0.9315 - auc: 0.9823 - loss: 0.3221 - pr_auc: 0.9819
Epoch 17: val_auc did not improve from 0.96690
728/728 ━━━━━━━━━━━━━━━━━━━━ 409s 563ms/step - accuracy: 0.9327 - auc: 0.9824 - loss: 0.3215 - pr_auc: 0.9821 - val_accuracy: 0.8490 - val_auc: 0.9470 - val_loss: 0.5077 - val_pr_auc: 0.9443 - learning_rate: 5.0000e-04
Epoch 18/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 516ms/step - accuracy: 0.9390 - auc: 0.9843 - loss: 0.3135 - pr_auc: 0.9845
Epoch 18: val_auc improved from 0.96690 to 0.97252, saving model to best_benchmark04.keras

Epoch 18: finished saving model to best_benchmark04.keras
728/728 ━━━━━━━━━━━━━━━━━━━━ 406s 557ms/step - accuracy: 0.9360 - auc: 0.9837 - loss: 0.3166 - pr_auc: 0.9838 - val_accuracy: 0.9152 - val_auc: 0.9725 - val_loss: 0.3608 - val_pr_auc: 0.9718 - learning_rate: 5.0000e-04
Epoch 19/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 504ms/step - accuracy: 0.9406 - auc: 0.9867 - loss: 0.3065 - pr_auc: 0.9866
Epoch 19: val_auc did not improve from 0.97252
728/728 ━━━━━━━━━━━━━━━━━━━━ 395s 543ms/step - accuracy: 0.9398 - auc: 0.9859 - loss: 0.3092 - pr_auc: 0.9859 - val_accuracy: 0.9026 - val_auc: 0.9628 - val_loss: 0.3818 - val_pr_auc: 0.9618 - learning_rate: 5.0000e-04
Epoch 20/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 506ms/step - accuracy: 0.9441 - auc: 0.9869 - loss: 0.3044 - pr_auc: 0.9871
Epoch 20: val_auc did not improve from 0.97252
728/728 ━━━━━━━━━━━━━━━━━━━━ 398s 546ms/step - accuracy: 0.9419 - auc: 0.9864 - loss: 0.3070 - pr_auc: 0.9866 - val_accuracy: 0.9183 - val_auc: 0.9723 - val_loss: 0.3521 - val_pr_auc: 0.9729 - learning_rate: 5.0000e-04
Epoch 21/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 509ms/step - accuracy: 0.9498 - auc: 0.9885 - loss: 0.2978 - pr_auc: 0.9884
Epoch 21: val_auc did not improve from 0.97252
728/728 ━━━━━━━━━━━━━━━━━━━━ 401s 550ms/step - accuracy: 0.9474 - auc: 0.9879 - loss: 0.2998 - pr_auc: 0.9878 - val_accuracy: 0.9081 - val_auc: 0.9677 - val_loss: 0.3668 - val_pr_auc: 0.9689 - learning_rate: 5.0000e-04
Epoch 22/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 512ms/step - accuracy: 0.9491 - auc: 0.9893 - loss: 0.2943 - pr_auc: 0.9891
Epoch 22: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.

Epoch 22: val_auc did not improve from 0.97252
728/728 ━━━━━━━━━━━━━━━━━━━━ 402s 552ms/step - accuracy: 0.9486 - auc: 0.9888 - loss: 0.2961 - pr_auc: 0.9887 - val_accuracy: 0.9098 - val_auc: 0.9704 - val_loss: 0.3719 - val_pr_auc: 0.9684 - learning_rate: 5.0000e-04
Epoch 23/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 513ms/step - accuracy: 0.9639 - auc: 0.9940 - loss: 0.2678 - pr_auc: 0.9938
Epoch 23: val_auc improved from 0.97252 to 0.97619, saving model to best_benchmark04.keras

Epoch 23: finished saving model to best_benchmark04.keras
728/728 ━━━━━━━━━━━━━━━━━━━━ 403s 553ms/step - accuracy: 0.9666 - auc: 0.9947 - loss: 0.2627 - pr_auc: 0.9947 - val_accuracy: 0.9247 - val_auc: 0.9762 - val_loss: 0.3411 - val_pr_auc: 0.9772 - learning_rate: 2.5000e-04
Epoch 24/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 511ms/step - accuracy: 0.9711 - auc: 0.9959 - loss: 0.2520 - pr_auc: 0.9957
Epoch 24: val_auc improved from 0.97619 to 0.98006, saving model to best_benchmark04.keras

Epoch 24: finished saving model to best_benchmark04.keras
728/728 ━━━━━━━━━━━━━━━━━━━━ 441s 551ms/step - accuracy: 0.9713 - auc: 0.9960 - loss: 0.2497 - pr_auc: 0.9960 - val_accuracy: 0.9341 - val_auc: 0.9801 - val_loss: 0.3228 - val_pr_auc: 0.9812 - learning_rate: 2.5000e-04
Epoch 25/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 517ms/step - accuracy: 0.9729 - auc: 0.9964 - loss: 0.2441 - pr_auc: 0.9963
Epoch 25: val_auc did not improve from 0.98006
728/728 ━━━━━━━━━━━━━━━━━━━━ 408s 560ms/step - accuracy: 0.9715 - auc: 0.9963 - loss: 0.2447 - pr_auc: 0.9962 - val_accuracy: 0.9230 - val_auc: 0.9773 - val_loss: 0.3384 - val_pr_auc: 0.9776 - learning_rate: 2.5000e-04
Epoch 26/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 522ms/step - accuracy: 0.9761 - auc: 0.9971 - loss: 0.2353 - pr_auc: 0.9970
Epoch 26: val_auc did not improve from 0.98006
728/728 ━━━━━━━━━━━━━━━━━━━━ 445s 564ms/step - accuracy: 0.9744 - auc: 0.9964 - loss: 0.2386 - pr_auc: 0.9964 - val_accuracy: 0.9307 - val_auc: 0.9778 - val_loss: 0.3252 - val_pr_auc: 0.9791 - learning_rate: 2.5000e-04
Epoch 27/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 519ms/step - accuracy: 0.9747 - auc: 0.9967 - loss: 0.2361 - pr_auc: 0.9966
Epoch 27: val_auc did not improve from 0.98006
728/728 ━━━━━━━━━━━━━━━━━━━━ 407s 560ms/step - accuracy: 0.9740 - auc: 0.9966 - loss: 0.2368 - pr_auc: 0.9965 - val_accuracy: 0.9314 - val_auc: 0.9771 - val_loss: 0.3263 - val_pr_auc: 0.9772 - learning_rate: 2.5000e-04
Epoch 28/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 509ms/step - accuracy: 0.9767 - auc: 0.9972 - loss: 0.2306 - pr_auc: 0.9971
Epoch 28: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814.

Epoch 28: val_auc did not improve from 0.98006
728/728 ━━━━━━━━━━━━━━━━━━━━ 401s 551ms/step - accuracy: 0.9758 - auc: 0.9971 - loss: 0.2307 - pr_auc: 0.9970 - val_accuracy: 0.9266 - val_auc: 0.9772 - val_loss: 0.3335 - val_pr_auc: 0.9784 - learning_rate: 2.5000e-04
Epoch 29/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 512ms/step - accuracy: 0.9808 - auc: 0.9981 - loss: 0.2193 - pr_auc: 0.9981
Epoch 29: val_auc improved from 0.98006 to 0.98225, saving model to best_benchmark04.keras

Epoch 29: finished saving model to best_benchmark04.keras
728/728 ━━━━━━━━━━━━━━━━━━━━ 403s 553ms/step - accuracy: 0.9828 - auc: 0.9983 - loss: 0.2170 - pr_auc: 0.9983 - val_accuracy: 0.9355 - val_auc: 0.9823 - val_loss: 0.3061 - val_pr_auc: 0.9839 - learning_rate: 1.2500e-04
Epoch 30/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 508ms/step - accuracy: 0.9864 - auc: 0.9990 - loss: 0.2090 - pr_auc: 0.9989
Epoch 30: val_auc did not improve from 0.98225
728/728 ━━━━━━━━━━━━━━━━━━━━ 438s 548ms/step - accuracy: 0.9858 - auc: 0.9989 - loss: 0.2088 - pr_auc: 0.9988 - val_accuracy: 0.9337 - val_auc: 0.9819 - val_loss: 0.3168 - val_pr_auc: 0.9832 - learning_rate: 1.2500e-04
Epoch 31/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 507ms/step - accuracy: 0.9853 - auc: 0.9987 - loss: 0.2072 - pr_auc: 0.9985
Epoch 31: val_auc did not improve from 0.98225
728/728 ━━━━━━━━━━━━━━━━━━━━ 399s 548ms/step - accuracy: 0.9855 - auc: 0.9987 - loss: 0.2070 - pr_auc: 0.9985 - val_accuracy: 0.9352 - val_auc: 0.9797 - val_loss: 0.3040 - val_pr_auc: 0.9814 - learning_rate: 1.2500e-04
Epoch 32/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 503ms/step - accuracy: 0.9877 - auc: 0.9991 - loss: 0.2034 - pr_auc: 0.9991
Epoch 32: val_auc improved from 0.98225 to 0.98299, saving model to best_benchmark04.keras

Epoch 32: finished saving model to best_benchmark04.keras
728/728 ━━━━━━━━━━━━━━━━━━━━ 397s 545ms/step - accuracy: 0.9875 - auc: 0.9991 - loss: 0.2030 - pr_auc: 0.9990 - val_accuracy: 0.9391 - val_auc: 0.9830 - val_loss: 0.2953 - val_pr_auc: 0.9850 - learning_rate: 1.2500e-04
Epoch 33/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 505ms/step - accuracy: 0.9879 - auc: 0.9989 - loss: 0.2016 - pr_auc: 0.9988
Epoch 33: val_auc did not improve from 0.98299
728/728 ━━━━━━━━━━━━━━━━━━━━ 443s 546ms/step - accuracy: 0.9879 - auc: 0.9989 - loss: 0.2008 - pr_auc: 0.9989 - val_accuracy: 0.9329 - val_auc: 0.9817 - val_loss: 0.3123 - val_pr_auc: 0.9831 - learning_rate: 1.2500e-04
Epoch 34/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 509ms/step - accuracy: 0.9887 - auc: 0.9992 - loss: 0.1973 - pr_auc: 0.9991
Epoch 34: val_auc did not improve from 0.98299
728/728 ━━━━━━━━━━━━━━━━━━━━ 400s 550ms/step - accuracy: 0.9881 - auc: 0.9991 - loss: 0.1979 - pr_auc: 0.9991 - val_accuracy: 0.9422 - val_auc: 0.9806 - val_loss: 0.2944 - val_pr_auc: 0.9817 - learning_rate: 1.2500e-04
Epoch 35/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 518ms/step - accuracy: 0.9873 - auc: 0.9992 - loss: 0.1962 - pr_auc: 0.9991
Epoch 35: val_auc did not improve from 0.98299
728/728 ━━━━━━━━━━━━━━━━━━━━ 407s 559ms/step - accuracy: 0.9880 - auc: 0.9991 - loss: 0.1959 - pr_auc: 0.9991 - val_accuracy: 0.9205 - val_auc: 0.9784 - val_loss: 0.3339 - val_pr_auc: 0.9791 - learning_rate: 1.2500e-04
Epoch 36/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 508ms/step - accuracy: 0.9875 - auc: 0.9990 - loss: 0.1959 - pr_auc: 0.9990
Epoch 36: ReduceLROnPlateau reducing learning rate to 6.25000029685907e-05.

Epoch 36: val_auc did not improve from 0.98299
728/728 ━━━━━━━━━━━━━━━━━━━━ 435s 549ms/step - accuracy: 0.9890 - auc: 0.9992 - loss: 0.1935 - pr_auc: 0.9991 - val_accuracy: 0.9390 - val_auc: 0.9795 - val_loss: 0.2979 - val_pr_auc: 0.9817 - learning_rate: 1.2500e-04
Epoch 37/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 505ms/step - accuracy: 0.9914 - auc: 0.9993 - loss: 0.1882 - pr_auc: 0.9992
Epoch 37: val_auc improved from 0.98299 to 0.98325, saving model to best_benchmark04.keras

Epoch 37: finished saving model to best_benchmark04.keras
728/728 ━━━━━━━━━━━━━━━━━━━━ 440s 546ms/step - accuracy: 0.9913 - auc: 0.9995 - loss: 0.1870 - pr_auc: 0.9995 - val_accuracy: 0.9379 - val_auc: 0.9833 - val_loss: 0.2984 - val_pr_auc: 0.9851 - learning_rate: 6.2500e-05
Epoch 38/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 508ms/step - accuracy: 0.9916 - auc: 0.9996 - loss: 0.1850 - pr_auc: 0.9996
Epoch 38: val_auc improved from 0.98325 to 0.98352, saving model to best_benchmark04.keras

Epoch 38: finished saving model to best_benchmark04.keras
728/728 ━━━━━━━━━━━━━━━━━━━━ 400s 549ms/step - accuracy: 0.9922 - auc: 0.9996 - loss: 0.1842 - pr_auc: 0.9996 - val_accuracy: 0.9434 - val_auc: 0.9835 - val_loss: 0.2867 - val_pr_auc: 0.9851 - learning_rate: 6.2500e-05
Epoch 39/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 517ms/step - accuracy: 0.9923 - auc: 0.9996 - loss: 0.1830 - pr_auc: 0.9996
Epoch 39: val_auc did not improve from 0.98352
728/728 ━━━━━━━━━━━━━━━━━━━━ 406s 558ms/step - accuracy: 0.9923 - auc: 0.9996 - loss: 0.1830 - pr_auc: 0.9996 - val_accuracy: 0.9297 - val_auc: 0.9826 - val_loss: 0.3091 - val_pr_auc: 0.9841 - learning_rate: 6.2500e-05
Epoch 40/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 504ms/step - accuracy: 0.9933 - auc: 0.9997 - loss: 0.1807 - pr_auc: 0.9997
Epoch 40: val_auc did not improve from 0.98352
728/728 ━━━━━━━━━━━━━━━━━━━━ 397s 545ms/step - accuracy: 0.9934 - auc: 0.9997 - loss: 0.1808 - pr_auc: 0.9997 - val_accuracy: 0.9431 - val_auc: 0.9829 - val_loss: 0.2837 - val_pr_auc: 0.9847 - learning_rate: 6.2500e-05
Epoch 41/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 504ms/step - accuracy: 0.9924 - auc: 0.9997 - loss: 0.1806 - pr_auc: 0.9997
Epoch 41: val_auc did not improve from 0.98352
728/728 ━━━━━━━━━━━━━━━━━━━━ 398s 546ms/step - accuracy: 0.9925 - auc: 0.9996 - loss: 0.1809 - pr_auc: 0.9996 - val_accuracy: 0.9374 - val_auc: 0.9824 - val_loss: 0.2934 - val_pr_auc: 0.9845 - learning_rate: 6.2500e-05
Epoch 42/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 502ms/step - accuracy: 0.9918 - auc: 0.9996 - loss: 0.1803 - pr_auc: 0.9996
Epoch 42: ReduceLROnPlateau reducing learning rate to 3.125000148429535e-05.

Epoch 42: val_auc did not improve from 0.98352
728/728 ━━━━━━━━━━━━━━━━━━━━ 395s 543ms/step - accuracy: 0.9927 - auc: 0.9997 - loss: 0.1791 - pr_auc: 0.9997 - val_accuracy: 0.9366 - val_auc: 0.9818 - val_loss: 0.2965 - val_pr_auc: 0.9832 - learning_rate: 6.2500e-05
Epoch 43/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 501ms/step - accuracy: 0.9939 - auc: 0.9998 - loss: 0.1766 - pr_auc: 0.9998
Epoch 43: val_auc did not improve from 0.98352
728/728 ━━━━━━━━━━━━━━━━━━━━ 395s 543ms/step - accuracy: 0.9946 - auc: 0.9998 - loss: 0.1756 - pr_auc: 0.9998 - val_accuracy: 0.9358 - val_auc: 0.9831 - val_loss: 0.2959 - val_pr_auc: 0.9847 - learning_rate: 3.1250e-05
Epoch 44/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 511ms/step - accuracy: 0.9940 - auc: 0.9998 - loss: 0.1753 - pr_auc: 0.9998
Epoch 44: val_auc improved from 0.98352 to 0.98387, saving model to best_benchmark04.keras

Epoch 44: finished saving model to best_benchmark04.keras
728/728 ━━━━━━━━━━━━━━━━━━━━ 402s 552ms/step - accuracy: 0.9946 - auc: 0.9998 - loss: 0.1743 - pr_auc: 0.9998 - val_accuracy: 0.9429 - val_auc: 0.9839 - val_loss: 0.2802 - val_pr_auc: 0.9858 - learning_rate: 3.1250e-05
Epoch 45/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 505ms/step - accuracy: 0.9947 - auc: 0.9997 - loss: 0.1741 - pr_auc: 0.9997
Epoch 45: val_auc did not improve from 0.98387
728/728 ━━━━━━━━━━━━━━━━━━━━ 439s 547ms/step - accuracy: 0.9950 - auc: 0.9998 - loss: 0.1738 - pr_auc: 0.9997 - val_accuracy: 0.9442 - val_auc: 0.9837 - val_loss: 0.2795 - val_pr_auc: 0.9859 - learning_rate: 3.1250e-05
Epoch 46/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 499ms/step - accuracy: 0.9947 - auc: 0.9998 - loss: 0.1738 - pr_auc: 0.9998
Epoch 46: val_auc did not improve from 0.98387
728/728 ━━━━━━━━━━━━━━━━━━━━ 437s 540ms/step - accuracy: 0.9946 - auc: 0.9998 - loss: 0.1734 - pr_auc: 0.9998 - val_accuracy: 0.9406 - val_auc: 0.9835 - val_loss: 0.2838 - val_pr_auc: 0.9857 - learning_rate: 3.1250e-05
Epoch 47/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 501ms/step - accuracy: 0.9946 - auc: 0.9999 - loss: 0.1723 - pr_auc: 0.9999
Epoch 47: val_auc did not improve from 0.98387
728/728 ━━━━━━━━━━━━━━━━━━━━ 395s 542ms/step - accuracy: 0.9946 - auc: 0.9999 - loss: 0.1721 - pr_auc: 0.9999 - val_accuracy: 0.9372 - val_auc: 0.9826 - val_loss: 0.2928 - val_pr_auc: 0.9844 - learning_rate: 3.1250e-05
Epoch 48/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 499ms/step - accuracy: 0.9953 - auc: 0.9998 - loss: 0.1709 - pr_auc: 0.9998
Epoch 48: ReduceLROnPlateau reducing learning rate to 1.5625000742147677e-05.

Epoch 48: val_auc did not improve from 0.98387
728/728 ━━━━━━━━━━━━━━━━━━━━ 393s 540ms/step - accuracy: 0.9951 - auc: 0.9998 - loss: 0.1712 - pr_auc: 0.9998 - val_accuracy: 0.9369 - val_auc: 0.9823 - val_loss: 0.2924 - val_pr_auc: 0.9841 - learning_rate: 3.1250e-05
Epoch 49/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 506ms/step - accuracy: 0.9951 - auc: 0.9999 - loss: 0.1706 - pr_auc: 0.9999
Epoch 49: val_auc did not improve from 0.98387
728/728 ━━━━━━━━━━━━━━━━━━━━ 398s 547ms/step - accuracy: 0.9957 - auc: 0.9999 - loss: 0.1695 - pr_auc: 0.9999 - val_accuracy: 0.9359 - val_auc: 0.9827 - val_loss: 0.2935 - val_pr_auc: 0.9845 - learning_rate: 1.5625e-05
Epoch 50/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 504ms/step - accuracy: 0.9963 - auc: 0.9998 - loss: 0.1688 - pr_auc: 0.9998
Epoch 50: val_auc did not improve from 0.98387
728/728 ━━━━━━━━━━━━━━━━━━━━ 397s 545ms/step - accuracy: 0.9963 - auc: 0.9998 - loss: 0.1687 - pr_auc: 0.9998 - val_accuracy: 0.9382 - val_auc: 0.9830 - val_loss: 0.2874 - val_pr_auc: 0.9850 - learning_rate: 1.5625e-05
Epoch 51/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 513ms/step - accuracy: 0.9960 - auc: 0.9998 - loss: 0.1692 - pr_auc: 0.9998
Epoch 51: val_auc did not improve from 0.98387
728/728 ━━━━━━━━━━━━━━━━━━━━ 404s 554ms/step - accuracy: 0.9961 - auc: 0.9999 - loss: 0.1688 - pr_auc: 0.9999 - val_accuracy: 0.9419 - val_auc: 0.9838 - val_loss: 0.2795 - val_pr_auc: 0.9863 - learning_rate: 1.5625e-05
Epoch 52/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 507ms/step - accuracy: 0.9959 - auc: 0.9998 - loss: 0.1681 - pr_auc: 0.9996
Epoch 52: ReduceLROnPlateau reducing learning rate to 7.812500371073838e-06.

Epoch 52: val_auc did not improve from 0.98387
728/728 ━━━━━━━━━━━━━━━━━━━━ 399s 547ms/step - accuracy: 0.9959 - auc: 0.9999 - loss: 0.1680 - pr_auc: 0.9998 - val_accuracy: 0.9409 - val_auc: 0.9830 - val_loss: 0.2824 - val_pr_auc: 0.9850 - learning_rate: 1.5625e-05
Epoch 53/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 531ms/step - accuracy: 0.9966 - auc: 0.9999 - loss: 0.1673 - pr_auc: 0.9999
Epoch 53: val_auc did not improve from 0.98387
728/728 ━━━━━━━━━━━━━━━━━━━━ 416s 572ms/step - accuracy: 0.9966 - auc: 0.9999 - loss: 0.1674 - pr_auc: 0.9999 - val_accuracy: 0.9407 - val_auc: 0.9830 - val_loss: 0.2825 - val_pr_auc: 0.9849 - learning_rate: 7.8125e-06
Epoch 54/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 561ms/step - accuracy: 0.9958 - auc: 0.9999 - loss: 0.1674 - pr_auc: 0.9999
Epoch 54: val_auc did not improve from 0.98387
728/728 ━━━━━━━━━━━━━━━━━━━━ 439s 603ms/step - accuracy: 0.9960 - auc: 0.9999 - loss: 0.1673 - pr_auc: 0.9999 - val_accuracy: 0.9394 - val_auc: 0.9828 - val_loss: 0.2872 - val_pr_auc: 0.9845 - learning_rate: 7.8125e-06
Epoch 55/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 555ms/step - accuracy: 0.9962 - auc: 1.0000 - loss: 0.1665 - pr_auc: 1.0000
Epoch 55: val_auc did not improve from 0.98387
728/728 ━━━━━━━━━━━━━━━━━━━━ 435s 597ms/step - accuracy: 0.9965 - auc: 0.9999 - loss: 0.1667 - pr_auc: 0.9999 - val_accuracy: 0.9402 - val_auc: 0.9830 - val_loss: 0.2837 - val_pr_auc: 0.9846 - learning_rate: 7.8125e-06
Epoch 56/150
728/728 ━━━━━━━━━━━━━━━━━━━━ 0s 561ms/step - accuracy: 0.9966 - auc: 1.0000 - loss: 0.1662 - pr_auc: 1.0000
Epoch 56: ReduceLROnPlateau reducing learning rate to 3.906250185536919e-06.

Epoch 56: val_auc did not improve from 0.98387
728/728 ━━━━━━━━━━━━━━━━━━━━ 439s 603ms/step - accuracy: 0.9966 - auc: 0.9999 - loss: 0.1663 - pr_auc: 0.9999 - val_accuracy: 0.9435 - val_auc: 0.9833 - val_loss: 0.2787 - val_pr_auc: 0.9853 - learning_rate: 7.8125e-06
"""




# =========================================================
# SECTION 2: THE DASHBOARD CODE
# =========================================================

st.set_page_config(page_title="EEG Dashboard", layout="wide")

# Theme Colors
FIG_BG = "#f8fafc"
AX_BG = "#eef4fb"
GRID_COLOR = "#cbd5e1"
TEXT_COLOR = "#1e293b"
EXPERIMENT_COLORS = ["#1e3a8a", "#92400e", "#4d7c0f", "#7e22ce", "#b91c1c", "#0f766e", "#374151", "#a16207", "#be185d", "#166534", "#475569", "#9a3412"]

def parse_log(log_text, name):
    data = {"epochs": [], "train_acc": [], "val_acc": [], "train_auc": [], "val_auc": [], "train_loss": [], "val_loss": []}

    # Flexible parsing logic to ignore extra fields like pr_auc or learning_rate
    current_epoch = None
    for line in log_text.splitlines():
        line = line.strip()
        if not line: continue

        # Detect Epoch line
        e_match = re.search(r"Epoch\s+(\d+)", line)
        if e_match:
            current_epoch = int(e_match.group(1))
            continue

        # Detect Metrics line (looks for the numeric values following the labels)
        if "accuracy:" in line and "val_accuracy:" in line:
            try:
                data["epochs"].append(current_epoch)
                data["train_acc"].append(float(re.search(r"accuracy:\s*([0-9.]+)", line).group(1)))
                data["train_auc"].append(float(re.search(r"auc:\s*([0-9.]+)", line).group(1)))
                data["train_loss"].append(float(re.search(r"loss:\s*([0-9.]+)", line).group(1)))
                data["val_acc"].append(float(re.search(r"val_accuracy:\s*([0-9.]+)", line).group(1)))
                data["val_auc"].append(float(re.search(r"val_auc:\s*([0-9.]+)", line).group(1)))
                data["val_loss"].append(float(re.search(r"val_loss:\s*([0-9.]+)", line).group(1)))
            except:
                continue # Skip malformed lines

    for k in data: data[k] = np.array(data[k])
    return data if len(data["epochs"]) > 0 else None

def smooth_curve(values, window=1):
    if window <= 1 or len(values) < window: return values
    return np.convolve(np.pad(values, (window//2, window-1-window//2), mode="edge"), np.ones(window)/window, mode="valid")

def plot_metric(selected_exps, metric_label, smooth_window):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)

    configs = {
        "Accuracy": ("train_acc", "val_acc", "Accuracy", [0.0, 1.0]),
        "AUC": ("train_auc", "val_auc", "ROC-AUC", [0.0, 1.0]),
        "Loss": ("train_loss", "val_loss", "Loss", None)
    }
    t_key, v_key, ylabel, ylim = configs[metric_label]

    for i, exp in enumerate(selected_exps):
        color = EXPERIMENT_COLORS[i % len(EXPERIMENT_COLORS)]
        ax.plot(exp["epochs"], smooth_curve(exp[t_key], smooth_window), color=color, linestyle="-", label=f"{exp['name']} Train")
        ax.plot(exp["epochs"], smooth_curve(exp[v_key], smooth_window), color=color, linestyle="--", label=f"{exp['name']} Val")

    ax.set_title(f"Performance: {metric_label}", fontweight="bold", color=TEXT_COLOR)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    if ylim: ax.set_ylim(ylim)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    return fig

def main():
    st.title(" EEG Training Analysis Dashboard")

    raw_logs = {
        "Exp 1": log_exp1, "Exp 2": log_exp2, "Exp 3": log_exp3, "Exp 4": log_exp4,
        "Exp 7": log_exp7, "Exp 8": log_exp8, "Exp 9": log_exp9, "Exp 10": log_exp10,
        "Exp 11": log_exp11, "Exp 12": log_exp12, "Exp 13": log_exp13, "Exp 14": log_exp14
    }

    # Pre-parse all available data
    available_data = []
    for name, text in raw_logs.items():
        parsed = parse_log(text, name)
        if parsed:
            parsed["name"] = name
            available_data.append(parsed)

    if not available_data:
        st.warning(" No logs detected. Please paste your logs into the code variables at the top of the script.")
        return

    # Sidebar Navigation
    st.sidebar.header("Navigation")
    all_names = [d["name"] for d in available_data]

    # Individual Experiment Buttons
    st.sidebar.write("### Quick Select")
    if 'current_selection' not in st.session_state:
        st.session_state.current_selection = [all_names[0]]

    for name in all_names:
        if st.sidebar.button(f" {name}", use_container_width=True):
            st.session_state.current_selection = [name]

    st.sidebar.divider()

    # Comparison Mode
    if st.sidebar.toggle("Compare Multiple Experiments"):
        st.session_state.current_selection = st.sidebar.multiselect("Select Overlays", all_names, default=st.session_state.current_selection)

    smooth_val = st.sidebar.slider("Smoothing Window", 1, 10, 1)

    # Filter data for display
    display_list = [d for d in available_data if d["name"] in st.session_state.current_selection]

    # Main Layout
    col_plots, col_metrics = st.columns([3, 1])

    with col_plots:
        st.pyplot(plot_metric(display_list, "Accuracy", smooth_val))
        st.pyplot(plot_metric(display_list, "AUC", smooth_val))
        st.pyplot(plot_metric(display_list, "Loss", smooth_val))

    with col_metrics:
        st.write("### Best Results")
        for exp in display_list:
            with st.container(border=True):
                st.write(f"**{exp['name']}**")
                st.metric("Peak Val Acc", f"{np.max(exp['val_acc']):.4f}")
                st.metric("Peak Val AUC", f"{np.max(exp['val_auc']):.4f}")
                st.metric("Min Val Loss", f"{np.min(exp['val_loss']):.4f}")

if __name__ == "__main__":
    main()